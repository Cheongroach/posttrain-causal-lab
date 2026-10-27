"""Two-device A0: real updates, GDN interventions and exact state replay."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import random
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, Qwen3_5ForCausalLM, Qwen3_5TextConfig
from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen_modeling
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    torch_causal_conv1d_update,
    torch_chunk_gated_delta_rule,
    torch_recurrent_gated_delta_rule,
)

from .contracts import output_intervention, response_mask, token_loss


def write(path: Path, value: object):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def event(output: Path, **value):
    value["unix_seconds"] = time.time()
    with (output / "events.jsonl").open("a") as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + "\n")
    print(json.dumps(value, ensure_ascii=False), flush=True)


def force_torch_delta(model):
    for layer in model.model.layers:
        attention = getattr(layer, "linear_attn", None)
        if attention is not None:
            attention.chunk_gated_delta_rule = torch_chunk_gated_delta_rule
            attention.recurrent_gated_delta_rule = torch_recurrent_gated_delta_rule
            attention.causal_conv1d_fn = None
            attention.causal_conv1d_update = torch_causal_conv1d_update


def load(path: Path, device: str):
    # Make normalization independent of optional FLA installations in the image.
    qwen_modeling.FusedRMSNormGated = None
    config = Qwen3_5TextConfig.from_pretrained(path, local_files_only=True)
    model, info = Qwen3_5ForCausalLM.from_pretrained(
        path, config=config, local_files_only=True, trust_remote_code=False,
        dtype=torch.bfloat16, attn_implementation="sdpa", device_map={"": device},
        output_loading_info=True)
    errors = {key: list(info[key]) for key in
              ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs") if info.get(key)}
    if errors:
        raise RuntimeError(f"Language weights did not load exactly: {errors}")
    force_torch_delta(model)
    model.eval()
    return model


def attach(model):
    model = get_peft_model(model, LoraConfig(
        task_type="CAUSAL_LM", r=16, lora_alpha=32, lora_dropout=0.,
        target_modules=["o_proj", "out_proj", "down_proj"]))
    model.float()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.eval()
    return model


def adapter_state(model):
    return {name: parameter.detach().cpu().clone()
            for name, parameter in model.named_parameters() if parameter.requires_grad}


def restore_adapter(model, state):
    parameters = dict(model.named_parameters())
    if set(state) != {name for name, p in parameters.items() if p.requires_grad}:
        raise ValueError("Checkpoint attachment sites differ")
    with torch.no_grad():
        for name, value in state.items():
            parameters[name].copy_(value)


def save_state(path, model, optimizer, cursor, provenance):
    temporary = path.with_suffix(".tmp")
    torch.save(dict(adapter=adapter_state(model), optimizer=optimizer.state_dict(),
                    torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
                    python_rng=random.getstate(), cursor=cursor, provenance=provenance), temporary)
    temporary.replace(path)
    write(path.with_suffix(".sha256.json"), {
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size})


def restore_state(path, model, optimizer, provenance):
    expected = json.loads(path.with_suffix(".sha256.json").read_text())["sha256"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise RuntimeError("Checkpoint hash differs")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state["provenance"] != provenance:
        raise RuntimeError("Checkpoint data/config provenance differs")
    restore_adapter(model, state["adapter"])
    optimizer.load_state_dict(state["optimizer"])
    torch.set_rng_state(state["torch_rng"])
    torch.cuda.set_rng_state_all(state["cuda_rng"])
    random.setstate(state["python_rng"])
    return state["cursor"]


def make_example(tokenizer, row, eos_id, max_sequence=256):
    prompt = tokenizer.apply_chat_template([
        {"role": "system", "content": "Solve the problem. Return only the final answer as #### <number>."},
        {"role": "user", "content": row["question"]},
    ], tokenize=True, return_dict=False, add_generation_prompt=True, enable_thinking=False)
    if not isinstance(prompt, list) or not all(isinstance(token, int) for token in prompt):
        raise TypeError("Rendered prompt must be a flat token-ID list")
    answer = tokenizer.encode("#### " + row["final_answer"], add_special_tokens=False) + [eos_id]
    if len(prompt) + len(answer) > max_sequence:
        raise ValueError("Example exceeds frozen sequence budget")
    if len(prompt) + 32 > max_sequence:
        raise ValueError("Student rollout would exceed frozen sequence budget")
    return prompt, answer


def logits(model, prompt, response):
    device = model.device
    ids = torch.tensor([prompt + response], dtype=torch.long, device=device)
    positions = torch.arange(len(prompt) - 1, ids.shape[1] - 1, device=device)
    result = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False,
                   logits_to_keep=positions).logits
    if not bool(torch.isfinite(result).all()):
        raise FloatingPointError("Nonfinite logits")
    return result


def update(student, teacher, optimizer, example, objective, eos_ids):
    prompt, response = example
    optimizer.zero_grad(set_to_none=True)
    reference = None
    if objective != "sft":
        with torch.no_grad():
            reference = logits(teacher, prompt, response).to(student.device)
    predicted = logits(student, prompt, response)
    targets = torch.tensor([response], device=student.device)
    mask = response_mask(targets, eos_ids)
    loss = token_loss(predicted, targets, mask, reference)
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("Nonfinite loss")
    loss.backward()
    gradients = [p.grad for p in student.parameters() if p.requires_grad and p.grad is not None]
    if not gradients or not all(bool(torch.isfinite(g).all()) for g in gradients):
        raise FloatingPointError("Missing or nonfinite adapter gradients")
    norm = torch.nn.utils.clip_grad_norm_(
        [p for p in student.parameters() if p.requires_grad], 1., error_if_nonfinite=True)
    if not float(norm) > 0:
        raise FloatingPointError("Zero adapter gradient")
    optimizer.step()
    torch.cuda.synchronize(student.device)
    return {"loss": float(loss.detach()), "grad_norm_before_clip": float(norm),
            "response_tokens": int(mask.sum()), "prefix_tokens": len(prompt)}


def rollout(student, tokenizer, row, eos_ids):
    prompt, _ = make_example(tokenizer, row, eos_ids[0])
    ids = torch.tensor([prompt], dtype=torch.long, device=student.device)
    with torch.no_grad():
        sequence = student.generate(
            input_ids=ids, attention_mask=torch.ones_like(ids), do_sample=True,
            temperature=1., top_p=1., top_k=0, max_new_tokens=32,
            eos_token_id=eos_ids, pad_token_id=tokenizer.pad_token_id,
            use_cache=True)
    response = sequence[0, len(prompt):].tolist()
    if not response:
        raise RuntimeError("Empty student rollout")
    return (prompt, response), {
        "text": tokenizer.decode(response, skip_special_tokens=True),
        "token_ids": response, "truncated": response[-1] not in eos_ids}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--student", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    seed = 20261027
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if torch.cuda.device_count() != 2 or not all(
            "H20" in torch.cuda.get_device_name(i) for i in range(2)):
        raise RuntimeError("A0 requires exactly two H20 devices")
    write(output / "environment.json", {
        "versions": {name: importlib.metadata.version(name) for name in
                     ("torch", "transformers", "peft", "accelerate", "tokenizers")},
        "cuda": torch.version.cuda,
        "devices": [torch.cuda.get_device_name(i) for i in range(2)],
        "seed": seed, "student_device": 0, "teacher_device": 1})
    event(output, event="loading_student")
    tokenizer = AutoTokenizer.from_pretrained(args.student, local_files_only=True)
    teacher_tokenizer = AutoTokenizer.from_pretrained(args.teacher, local_files_only=True)
    if tokenizer.get_vocab() != teacher_tokenizer.get_vocab():
        raise ValueError("Full token-to-ID mapping differs")
    if tokenizer.all_special_ids != teacher_tokenizer.all_special_ids:
        raise ValueError("Special token IDs differ")
    student = attach(load(args.student, "cuda:0"))
    event(output, event="loading_teacher")
    teacher = load(args.teacher, "cuda:1").requires_grad_(False)
    if student.config.vocab_size != teacher.config.vocab_size:
        raise ValueError("Output vocabulary dimensions differ")
    eos_ids = student.generation_config.eos_token_id
    if isinstance(eos_ids, int):
        eos_ids = [eos_ids]
    if not eos_ids:
        eos_ids = [tokenizer.eos_token_id]
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    rows = [json.loads(line) for line in (args.data / "train.jsonl").read_text().splitlines()]
    eligible, skipped = [], []
    for row in rows:
        try:
            example = make_example(tokenizer, row, eos_ids[0])
        except ValueError:
            skipped.append(row["id"])
            continue
        # Compare the actual rendered protocol.
        messages = [{"role": "system", "content": "Solve the problem. Return only the final answer as #### <number>."},
                    {"role": "user", "content": row["question"]}]
        if tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False, add_generation_prompt=True,
                                         enable_thinking=False) != teacher_tokenizer.apply_chat_template(
                messages, tokenize=True, return_dict=False, add_generation_prompt=True, enable_thinking=False):
            raise ValueError("Rendered teacher/student prompts differ")
        eligible.append((row, example))
    if len(eligible) < 8:
        raise RuntimeError("Insufficient examples within sequence budget")
    base = student.get_base_model()
    sites = [name for name, module in student.named_modules() if hasattr(module, "lora_A")]
    provenance = {
        "data_manifest_sha256": hashlib.sha256((args.data / "manifest.json").read_bytes()).hexdigest(),
        "models_manifest_sha256": hashlib.sha256(
            (Path(__file__).parent / "assets/models.json").read_bytes()).hexdigest(),
        "seed": seed, "lora_sites": sites, "lora_rank": 16, "lora_alpha": 32,
        "student_precision": "float32", "teacher_precision": "bfloat16",
        "loss": "CE or full-vocabulary KL(teacher || student), response-token mean, T=1",
        "learning_rate": 1e-4, "max_sequence": 256, "max_new_tokens": 32,
        "kernel": "transformers torch GDN; SDPA full attention; TF32 disabled",
        "thinking": False, "eos_ids": eos_ids,
        "tokenizer_vocab_sha256": hashlib.sha256(
            json.dumps(tokenizer.get_vocab(), sort_keys=True).encode()).hexdigest(),
    }
    write(output / "contract.json", {
        **provenance, "student_config": student.config.to_dict(),
        "teacher_config": teacher.config.to_dict(),
        "trainable_parameters": sum(p.numel() for p in student.parameters() if p.requires_grad),
        "loading": "native text class, zero missing/unexpected/mismatched language keys",
        "selected_ids": [row["id"] for row, _ in eligible[:8]], "skipped_long_ids": skipped})
    optimizer = torch.optim.AdamW(
        [p for p in student.parameters() if p.requires_grad], lr=1e-4, weight_decay=0.)
    initial = output / "initial.pt"
    save_state(initial, student, optimizer, 0, provenance)
    probe = eligible[0][1]
    gdn = next(layer.linear_attn for layer in base.model.layers if hasattr(layer, "linear_attn"))
    with torch.no_grad():
        before = logits(student, *probe).cpu()
        with output_intervention(gdn, "capture") as captured:
            logits(student, *probe)
        with output_intervention(gdn, "identity"):
            identity = logits(student, *probe).cpu()
        with output_intervention(gdn, "zero"):
            ablated = logits(student, *probe).cpu()
        with output_intervention(gdn, "replace", captured[0]):
            restored = logits(student, *probe).cpu()
    torch.testing.assert_close(identity, before, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(restored, before, atol=1e-6, rtol=1e-6)
    intervention = {
        "layer": 0, "site": "linear_attn output, all prompt and response positions",
        "identity_max_abs": float((identity - before).abs().max()),
        "restore_max_abs": float((restored - before).abs().max()),
        "zero_max_abs": float((ablated - before).abs().max())}
    if intervention["zero_max_abs"] <= 1e-7:
        raise RuntimeError("GDN zero intervention had no measurable effect")
    write(output / "intervention.json", intervention)
    event(output, event="gdn_gates_passed", **intervention)

    # Recovery is tested at the same next minibatch with optimizer and RNG.
    first = update(student, teacher, optimizer, probe, "sft", eos_ids)
    checkpoint = output / "resume-step1.pt"
    save_state(checkpoint, student, optimizer, 1, provenance)
    reference_step = update(student, teacher, optimizer, eligible[1][1], "sft", eos_ids)
    reference_state = adapter_state(student)
    cursor = restore_state(checkpoint, student, optimizer, provenance)
    if cursor != 1:
        raise RuntimeError("Sampler cursor was not restored")
    replay_step = update(student, teacher, optimizer, eligible[cursor][1], "sft", eos_ids)
    replay_state = adapter_state(student)
    max_error = max(float((replay_state[k] - reference_state[k]).abs().max()) for k in reference_state)
    for key in reference_state:
        torch.testing.assert_close(replay_state[key], reference_state[key], atol=1e-7, rtol=1e-6)
    if abs(replay_step["loss"] - reference_step["loss"]) > 1e-6:
        raise RuntimeError("Resumed loss differs from uninterrupted next step")
    write(output / "recovery.json", {
        "first_step": first, "reference_next_step": reference_step, "replayed_next_step": replay_step,
        "adapter_max_abs": max_error, "cursor": cursor, "optimizer_and_rng_restored": True,
        "scope": "same-process state replay; excludes cross-machine fault recovery"})
    event(output, event="recovery_gates_passed", adapter_max_abs=max_error)

    # All arms start from the identical adapter, optimizer and RNG state.
    results = {}
    for objective in ("sft", "kd", "opd"):
        restore_state(initial, student, optimizer, provenance)
        events = []
        for step in range(4):
            row, example = eligible[step]
            if objective == "opd":
                example, generation = rollout(student, tokenizer, row, eos_ids)
                write(output / f"opd-rollout-{step}.json", {"id": row["id"], **generation})
            timestamp = time.monotonic()
            record = update(student, teacher, optimizer, example, objective, eos_ids)
            record.update(objective=objective, step=step, id=row["id"],
                          elapsed_seconds=time.monotonic() - timestamp)
            event(output, event="training_step", **record)
            events.append(record)
        state = adapter_state(student)
        initial_state = torch.load(initial, map_location="cpu", weights_only=True)["adapter"]
        update_max_abs = max(float((state[k] - initial_state[k]).abs().max()) for k in state)
        if update_max_abs <= 0:
            raise RuntimeError("No adapter parameter changed")
        with torch.no_grad():
            trained_logits = logits(student, *probe).cpu()
            with student.disable_adapter():
                disabled_logits = logits(student, *probe).cpu()
        torch.testing.assert_close(disabled_logits, before, atol=1e-6, rtol=1e-6)
        save_state(output / f"{objective}-step4.pt", student, optimizer, 4, provenance)
        student.save_pretrained(output / f"{objective}-adapter", safe_serialization=True)
        results[objective] = {
            "steps": events, "adapter_update_max_abs": update_max_abs,
            "trained_logit_change_max_abs": float((trained_logits - before).abs().max()),
            "disabled_adapter_max_abs": float((disabled_logits - before).abs().max())}
        write(output / "short-arms.json", results)
    write(output / "completed.json", {
        "a0_passed": True, "short_arms": list(results), "elapsed_seconds": time.monotonic() - started,
        "peak_allocated_bytes": [torch.cuda.max_memory_allocated(i) for i in range(2)],
        "quality_conclusion": None, "scope": "one seed, four updates per arm, training split only"})
    event(output, event="a0_complete")


if __name__ == "__main__":
    main()
