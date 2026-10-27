"""Paired 24-H20 experiment: output budget, training, grading, and mediation."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import random
import time
from pathlib import Path

import torch

from .contracts import response_mask, token_loss
from .pilot import adapter_state, attach, event, load, restore_state, save_state, write
from .scale_contracts import ARMS, grade, matched_random, choose_sites


def prompt_tokens(tokenizer, row):
    prompt = tokenizer.apply_chat_template([
        {"role": "system", "content": "Solve the problem. Return only the final answer as #### <number>."},
        {"role": "user", "content": row["question"]},
    ], tokenize=True, return_dict=False, add_generation_prompt=True, enable_thinking=False)
    if not isinstance(prompt, list) or not all(isinstance(token, int) for token in prompt):
        raise TypeError("Prompt must be a flat integer list")
    if len(prompt) + 256 > 768:
        raise ValueError("Prompt exceeds frozen 768-token total budget")
    return prompt


def example(tokenizer, row, eos):
    prompt = prompt_tokens(tokenizer, row)
    answer = tokenizer.encode("#### " + row["final_answer"], add_special_tokens=False) + [eos]
    return prompt, answer


def forward(model, prompt, response):
    ids = torch.tensor([prompt + response], device=model.device, dtype=torch.long)
    positions = torch.arange(len(prompt) - 1, ids.shape[1] - 1, device=model.device)
    result = model(input_ids=ids, attention_mask=torch.ones_like(ids),
                   use_cache=False, logits_to_keep=positions).logits
    if not bool(torch.isfinite(result).all()):
        raise FloatingPointError("Nonfinite logits")
    return result


def generate(model, tokenizer, row, eos_ids, budget, sample):
    prompt = prompt_tokens(tokenizer, row)
    ids = torch.tensor([prompt], device=model.device, dtype=torch.long)
    options = dict(input_ids=ids, attention_mask=torch.ones_like(ids), do_sample=sample,
                   max_new_tokens=budget, use_cache=True, eos_token_id=eos_ids,
                   pad_token_id=tokenizer.pad_token_id)
    if sample:
        options.update(temperature=1., top_p=1., top_k=0)
    started = time.monotonic()
    with torch.no_grad():
        sequence = model.generate(**options)
    tokens = sequence[0, len(prompt):].tolist()
    if not tokens:
        raise RuntimeError("Empty generation")
    text = tokenizer.decode(tokens, skip_special_tokens=True)
    truncated = tokens[-1] not in eos_ids
    return (prompt, tokens), {
        "id": row["id"], "text": text, "token_ids": tokens,
        "output_tokens": len(tokens), "elapsed_seconds": time.monotonic() - started,
        **grade(text, row["final_answer"], truncated)}


def evaluate(student, tokenizer, rows, eos_ids, path):
    records = []
    for row in rows:
        _, record = generate(student, tokenizer, row, eos_ids, 256, False)
        records.append(record)
        with path.open("a") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    return summarize(records)


def summarize(records):
    return {"n": len(records), **{
        key: sum(float(record[key]) for record in records) / len(records)
        for key in ("strict_correct", "numeric_correct", "complete_numeric_correct",
                    "format_valid", "truncated", "output_tokens")}}


def common_prefix_metrics(student, teacher, tokenizer, rows, eos_ids, path):
    """Gold-prefix NLL and teacher KL are diagnostics, not generation accuracy."""
    records = []
    with torch.no_grad():
        for row in rows:
            pair = example(tokenizer, row, eos_ids[0])
            reference = forward(teacher, *pair).to(student.device)
            prediction = forward(student, *pair)
            target = torch.tensor([pair[1]], device=student.device)
            mask = response_mask(target, eos_ids)
            record = {"id": row["id"], "tokens": int(mask.sum()),
                      "gold_nll": float(token_loss(prediction, target, mask)),
                      "teacher_kl": float(token_loss(prediction, target, mask, reference))}
            with path.open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            records.append(record)
    tokens = sum(row["tokens"] for row in records)
    return {"n": len(records), "tokens": tokens, **{
        key: sum(row[key] * row["tokens"] for row in records) / tokens
        for key in ("gold_nll", "teacher_kl")}}


def train_step(student, teacher, optimizer, pair, arm, eos_ids):
    prompt, tokens = pair
    optimizer.zero_grad(set_to_none=True)
    target = torch.tensor([tokens], device=student.device)
    mask = response_mask(target, eos_ids)
    reference = None
    if arm != "sft":
        with torch.no_grad():
            reference = forward(teacher, prompt, tokens).to(student.device)
    predicted = forward(student, prompt, tokens)
    loss = token_loss(predicted, target, mask, reference)
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("Nonfinite loss")
    loss.backward()
    parameters = [p for p in student.parameters() if p.requires_grad]
    if any(p.grad is None for p in parameters):
        raise RuntimeError("An attachment site has no gradient")
    norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
    if float(norm) <= 0:
        raise FloatingPointError("Zero gradient")
    optimizer.step()
    return {"loss": float(loss.detach()), "grad_norm": float(norm),
            "response_tokens": int(mask.sum()), "prefix_tokens": len(prompt)}


def sites(model):
    result = {}
    for index, layer in enumerate(model.get_base_model().model.layers):
        if hasattr(layer, "linear_attn"):
            result[f"layer{index:02d}.gdn"] = layer.linear_attn
        else:
            result[f"layer{index:02d}.attention"] = layer.self_attn
        result[f"layer{index:02d}.ffn"] = layer.mlp
    return result


@contextlib.contextmanager
def capture_sites(modules, captured):
    handles = []
    def hook(name):
        def capture(_module, _inputs, output):
            tensor = output[0] if isinstance(output, tuple) else output
            captured[name] = tensor.detach().cpu().clone()
        return capture
    try:
        for name, module in modules.items():
            handles.append(module.register_forward_hook(hook(name)))
        yield
    finally:
        for handle in handles:
            handle.remove()


@contextlib.contextmanager
def patch_site(module, replacement):
    def hook(_module, _inputs, output):
        tensor = output[0] if isinstance(output, tuple) else output
        if tensor.shape != replacement.shape:
            raise ValueError("Activation replacement shape differs")
        value = replacement.to(tensor.device, tensor.dtype)
        return (value, *output[1:]) if isinstance(output, tuple) else value
    handle = module.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def gold_logp(values, targets):
    return float(values.float().log_softmax(-1).gather(-1, targets[..., None]).mean())


def intervention_scan(student, tokenizer, rows, eos_ids, selected, output, seed):
    modules = sites(student)
    names = list(modules) if selected is None else selected
    records = []
    for row_index, row in enumerate(rows):
        pair = example(tokenizer, row, eos_ids[0])
        target = torch.tensor([pair[1]], device=student.device)
        donors, recipient = {}, {}
        with torch.no_grad():
            with student.disable_adapter(), capture_sites(modules, donors):
                base_logits = forward(student, *pair)
                base_score = gold_logp(base_logits, target)
            with capture_sites(modules, recipient):
                final_logits = forward(student, *pair)
                final_score = gold_logp(final_logits, target)
            for site_index, name in enumerate(names):
                modes = [("base", donors[name]), ("random", matched_random(
                    recipient[name], donors[name], seed + 1000 * row_index + site_index))]
                if site_index == 0:
                    modes.append(("identity", recipient[name]))
                for mode, replacement in modes:
                    with patch_site(modules[name], replacement):
                        patched = forward(student, *pair)
                    if mode == "identity":
                        torch.testing.assert_close(patched, final_logits, atol=1e-6, rtol=1e-6)
                    record = {
                        "id": row["id"], "site": name, "mode": mode,
                        "base_gold_logp": base_score, "trained_gold_logp": final_score,
                        "patched_gold_logp": gold_logp(patched, target),
                        "gold_logp_drop": final_score - gold_logp(patched, target),
                        "logits_max_abs": float((patched - final_logits).abs().max()),
                        "intervention_l2": float((replacement - recipient[name]).float().norm())}
                    with output.open("a") as stream:
                        stream.write(json.dumps(record) + "\n")
                    records.append(record)
    return records


def main():
    parser = argparse.ArgumentParser()
    for name in ("student", "teacher", "data", "output", "protocol"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if torch.cuda.device_count() != 2:
        raise RuntimeError("Each experimental cell owns two GPUs")
    if not all("H20" in torch.cuda.get_device_name(i) for i in range(2)):
        raise RuntimeError("The frozen experiment requires H20")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.student, local_files_only=True)
    teacher_tokenizer = AutoTokenizer.from_pretrained(args.teacher, local_files_only=True)
    if tokenizer.get_vocab() != teacher_tokenizer.get_vocab():
        raise ValueError("Full tokenizer vocabularies differ")
    if tokenizer.all_special_ids != teacher_tokenizer.all_special_ids:
        raise ValueError("Special token IDs differ")
    for text in ("Solve 3+4.", "计算三个加四个。", "3と4を足してください。"):
        messages = [{"role": "user", "content": text}]
        options = dict(tokenize=True, return_dict=False, enable_thinking=False, add_generation_prompt=True)
        if tokenizer.apply_chat_template(messages, **options) != teacher_tokenizer.apply_chat_template(
                messages, **options):
            raise ValueError("Actual templates differ")
    event(root, event="cell_loading", arm=args.arm, seed=args.seed)
    student = attach(load(args.student, "cuda:0"))
    teacher = load(args.teacher, "cuda:1").requires_grad_(False)
    if student.config.vocab_size != teacher.config.vocab_size:
        raise ValueError("Output vocabulary sizes differ")
    eos_ids = student.generation_config.eos_token_id
    if isinstance(eos_ids, int):
        eos_ids = [eos_ids]
    if not eos_ids:
        eos_ids = [tokenizer.eos_token_id]
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    manifest = json.loads((args.data / "manifest.json").read_text())
    rows = {}
    for split in ("train", "dev", "verification"):
        path = args.data / (split + ".jsonl")
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["files"][path.name]["sha256"]:
            raise RuntimeError("Frozen dataset hash differs")
        rows[split] = [json.loads(line) for line in path.read_text().splitlines()]
        for row in rows[split]:
            prompt_tokens(tokenizer, row)
    ids = {split: {r["id"] for r in values} for split, values in rows.items()}
    if ids["train"] & ids["dev"] or ids["train"] & ids["verification"] or ids["dev"] & ids["verification"]:
        raise RuntimeError("Data splits overlap")
    random.Random(args.seed).shuffle(rows["train"])
    train_order = [row["id"] for row in rows["train"]]
    budget = 32 if args.arm == "opd32" else 256
    provenance = {
        "seed": args.seed, "arm": args.arm,
        "protocol_sha256": hashlib.sha256(args.protocol.read_bytes()).hexdigest(),
        "data_manifest_sha256": hashlib.sha256((args.data / "manifest.json").read_bytes()).hexdigest(),
        "models_manifest_sha256": hashlib.sha256(
            (Path(__file__).parent / "assets/models.json").read_bytes()).hexdigest(),
        "train_order_sha256": hashlib.sha256(json.dumps(train_order).encode()).hexdigest(),
        "lora_sites": [name for name, m in student.named_modules() if hasattr(m, "lora_A")],
        "rollout_budget": budget, "learning_rate": 1e-4,
        "student_dtype": "float32", "teacher_dtype": "bfloat16",
    }
    initial_digest = hashlib.sha256()
    for name, tensor in sorted(adapter_state(student).items()):
        initial_digest.update(name.encode())
        initial_digest.update(tensor.numpy().tobytes())
    write(root / "contract.json", {
        **provenance, "versions": {name: importlib.metadata.version(name) for name in
                                 ("torch", "transformers", "peft", "tokenizers")},
        "split_counts": {k: len(v) for k, v in rows.items()}, "train_order": train_order,
        "benchmark_test_accessed": False, "evaluation_budget": 256,
        "student_parameters": sum(p.numel() for p in student.parameters()),
        "trainable_parameters": sum(p.numel() for p in student.parameters() if p.requires_grad),
        "initial_adapter_sha256": initial_digest.hexdigest(),
        "gpu_names": [torch.cuda.get_device_name(i) for i in range(2)]})
    optimizer = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad],
                                 lr=1e-4, weight_decay=0.)
    initial = root / "initial.pt"
    save_state(initial, student, optimizer, 0, provenance)

    # Baseline uses common deterministic protocol; it is not an independent seed sample.
    metrics = {"base_dev": evaluate(student, tokenizer, rows["dev"], eos_ids, root / "base-dev.jsonl")}
    metrics["base_dev_prefix"] = common_prefix_metrics(
        student, teacher, tokenizer, rows["dev"], eos_ids, root / "base-dev-prefix.jsonl")
    write(root / "metrics.json", metrics)
    event(root, event="baseline_complete", **metrics["base_dev"])
    restore_state(initial, student, optimizer, provenance)
    gates = []
    for step, row in enumerate(rows["train"]):
        pair = example(tokenizer, row, eos_ids[0])
        if args.arm.startswith("opd"):
            pair, rollout = generate(student, tokenizer, row, eos_ids, budget, True)
            with (root / "rollouts.jsonl").open("a") as stream:
                stream.write(json.dumps({"step": step, **rollout}, ensure_ascii=False) + "\n")
        timestamp = time.monotonic()
        record = train_step(student, teacher, optimizer, pair, args.arm, eos_ids)
        event(root, event="train_step", step=step + 1, id=row["id"], arm=args.arm,
              elapsed_seconds=time.monotonic() - timestamp, **record)
        if step + 1 in (64, 128, 256):
            save_state(root / f"step{step+1}.pt", student, optimizer, step + 1, provenance)
        if step == 127:
            path = root / "step128.pt"
            next_pair = example(tokenizer, rows["train"][128], eos_ids[0])
            # Both paths use the same fixed next prefix; replay does not consume training data twice.
            reference = train_step(student, teacher, optimizer, next_pair, args.arm, eos_ids)
            reference_state = adapter_state(student)
            restore_state(path, student, optimizer, provenance)
            replay = train_step(student, teacher, optimizer, next_pair, args.arm, eos_ids)
            replay_state = adapter_state(student)
            maximum = 0.
            for key, value in reference_state.items():
                torch.testing.assert_close(replay_state[key], value, atol=1e-7, rtol=1e-6)
                maximum = max(maximum, float((replay_state[key] - value).abs().max()))
            if abs(reference["loss"] - replay["loss"]) > 1e-6:
                raise RuntimeError("Next-step replay differs")
            gates.append({"step": 128, "adapter_max_abs": maximum,
                          "reference": reference, "replay": replay,
                          "scope": "same-process fixed-prefix replay"})
            write(root / "recovery.json", gates)
            restore_state(path, student, optimizer, provenance)
    student.save_pretrained(root / "adapter", safe_serialization=True)
    metrics["trained_dev"] = evaluate(student, tokenizer, rows["dev"], eos_ids, root / "trained-dev.jsonl")
    metrics["trained_dev_prefix"] = common_prefix_metrics(
        student, teacher, tokenizer, rows["dev"], eos_ids, root / "trained-dev-prefix.jsonl")
    write(root / "metrics.json", metrics)
    event(root, event="dev_evaluation_complete", **metrics["trained_dev"])
    dev_scan = intervention_scan(student, tokenizer, rows["dev"][:8], eos_ids, None,
                                 root / "dev-interventions.jsonl", args.seed)
    selected = choose_sites(dev_scan)
    write(root / "selected-sites.json", {"selection_split": "dev", "n_search": 8,
                                        "criterion": "largest gold logp drop under base donor",
                                        "sites": selected})
    # Verification outputs are computed only after site selection is fixed.
    metrics["trained_verification"] = evaluate(
        student, tokenizer, rows["verification"], eos_ids, root / "trained-verification.jsonl")
    with student.disable_adapter():
        metrics["base_verification"] = evaluate(
            student, tokenizer, rows["verification"], eos_ids, root / "base-verification.jsonl")
    metrics["trained_verification_prefix"] = common_prefix_metrics(
        student, teacher, tokenizer, rows["verification"], eos_ids,
        root / "trained-verification-prefix.jsonl")
    intervention_scan(student, tokenizer, rows["verification"], eos_ids, selected,
                      root / "verification-interventions.jsonl", args.seed + 100000)
    write(root / "metrics.json", metrics)
    write(root / "completed.json", {
        "complete": True, "arm": args.arm, "seed": args.seed,
        "steps": len(rows["train"]), "metrics": metrics, "selected_sites": selected,
        "recovery_passed": True, "elapsed_seconds": time.monotonic() - started,
        "peak_allocated_bytes": [torch.cuda.max_memory_allocated(i) for i in range(2)],
        "scope": "paired training and fixed-prefix activation mediation; no train-signal repair"})
    event(root, event="cell_complete", arm=args.arm, seed=args.seed)


if __name__ == "__main__":
    main()
