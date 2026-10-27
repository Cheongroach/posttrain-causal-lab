"""Audit terminal artifacts before describing the frozen paired experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import statistics
import tarfile
import urllib.request
from datetime import datetime
from pathlib import Path

import torch

from .scale_contracts import ARMS, SEEDS, cell_layout, choose_sites, grade


METRICS = ("strict_correct", "numeric_correct", "complete_numeric_correct",
           "format_valid", "truncated", "output_tokens")
PACKAGE_SHA256 = "aa32b34506aa9be8d1f7d6255f1300469e79fcc83a911965d2f4dedbaac57783"


class AuditError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise AuditError(message)


def reject_constant(value):
    raise AuditError("Nonfinite JSON constant: " + value)


def read(path):
    return json.loads(path.read_text(), parse_constant=reject_constant)


def records(path):
    return [json.loads(line, parse_constant=reject_constant)
            for line in path.read_text().splitlines() if line.strip()]


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def close(left, right, message, atol=1e-8, rtol=1e-6):
    require(math.isfinite(left) and math.isfinite(right)
            and math.isclose(left, right, abs_tol=atol, rel_tol=rtol), message)


def summary(rows):
    require(bool(rows), "Empty metric records")
    return {"n": len(rows), **{key: statistics.mean(float(r[key]) for r in rows)
                              for key in METRICS}}


def check_summary(actual, expected, label):
    require(actual.keys() == expected.keys(), label + ": metric fields differ")
    for key, value in expected.items():
        close(actual[key], value, label + ": " + key)


def verify_generations(rows, gold, tokenizer, eos_ids, budget):
    require([r["id"] for r in rows] == [r["id"] for r in gold],
            "Generation IDs/order differ or have duplicates")
    for row, answer in zip(rows, gold):
        tokens = row["token_ids"]
        require(tokens and len(tokens) <= budget
                and all(type(t) is int and 0 <= t < len(tokenizer) for t in tokens),
                "Invalid generation tokens")
        require(row["output_tokens"] == len(tokens), "Output token count differs")
        require(not any(t in eos_ids for t in tokens[:-1]), "Generation continued after EOS")
        truncated = tokens[-1] not in eos_ids
        require(not truncated or len(tokens) == budget, "Short generation without EOS")
        text = tokenizer.decode(tokens, skip_special_tokens=True)
        require(text == row["text"], "Stored text differs from raw token decoding")
        for key, value in grade(text, answer["final_answer"], truncated).items():
            require(row[key] == value, "Reproduced grading differs: " + key)
        require(math.isfinite(row["elapsed_seconds"]) and row["elapsed_seconds"] >= 0,
                "Invalid generation elapsed time")
    return summary(rows)


def verify_interventions(rows, ids, names):
    expected = {(identifier, site, mode) for identifier in ids for site in names
                for mode in ("base", "random")}
    expected |= {(identifier, names[0], "identity") for identifier in ids}
    grouped = {(r["id"], r["site"], r["mode"]): r for r in rows}
    require(len(grouped) == len(rows) and set(grouped) == expected,
            "Missing, duplicate, or unexpected intervention records")
    identity_max = 0.
    for identifier in ids:
        reference = grouped[identifier, names[0], "identity"]
        for site in names:
            base, noise = (grouped[identifier, site, mode] for mode in ("base", "random"))
            close(base["intervention_l2"], noise["intervention_l2"],
                  "Random perturbation L2 differs from donor L2", atol=1e-5, rtol=1e-5)
        for key, row in grouped.items():
            if key[0] != identifier:
                continue
            require(all(math.isfinite(row[field]) for field in (
                "base_gold_logp", "trained_gold_logp", "patched_gold_logp",
                "gold_logp_drop", "logits_max_abs", "intervention_l2")),
                "Nonfinite intervention value")
            require(row["logits_max_abs"] >= 0 and row["intervention_l2"] >= 0,
                    "Negative intervention magnitude")
            close(row["gold_logp_drop"], row["trained_gold_logp"] - row["patched_gold_logp"],
                  "Intervention score arithmetic differs")
            for field in ("base_gold_logp", "trained_gold_logp"):
                close(row[field], reference[field], "Unpatched score differs across modes")
        require(reference["intervention_l2"] == 0., "Identity activation changed")
        close(reference["gold_logp_drop"], 0., "Identity gold score changed", atol=1e-6)
        # The worker asserts tensorwise atol=rtol=1e-6. Full logits are not saved,
        # so an absolute maximum alone cannot independently recheck its rtol.
        identity_max = max(identity_max, reference["logits_max_abs"])
    result = {"n_rows": len(ids), "n_records": len(rows), "identity_max_abs": identity_max,
              "sites": {}}
    for site in names:
        base = statistics.mean(grouped[i, site, "base"]["gold_logp_drop"] for i in ids)
        noise = statistics.mean(grouped[i, site, "random"]["gold_logp_drop"] for i in ids)
        result["sites"][site] = {"base_drop": base, "random_drop": noise,
                                 "base_minus_random": base - noise}
    for field in ("base_drop", "random_drop", "base_minus_random"):
        result[field] = statistics.mean(r[field] for r in result["sites"].values())
    return result


def finite_tree(value):
    if isinstance(value, torch.Tensor):
        require(bool(torch.isfinite(value).all()), "Nonfinite checkpoint tensor")
    elif isinstance(value, dict):
        for child in value.values():
            finite_tree(child)
    elif isinstance(value, (tuple, list)):
        for child in value:
            finite_tree(child)
    elif isinstance(value, float):
        require(math.isfinite(value), "Nonfinite checkpoint scalar")


def adapter_digest(adapter):
    digest = hashlib.sha256()
    for name, tensor in sorted(adapter.items()):
        digest.update(name.encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def verify_checkpoints(cell, contract, provenance):
    initial_adapter, final_adapter = None, None
    result = {}
    for step, name in ((0, "initial"), (64, "step64"), (128, "step128"), (256, "step256")):
        path = cell / (name + ".pt")
        metadata = read(path.with_suffix(".sha256.json"))
        require(path.stat().st_size == metadata["bytes"] and sha(path) == metadata["sha256"],
                "Checkpoint bytes/hash differ: " + name)
        state = torch.load(path, map_location="cpu", weights_only=True)
        require(state["cursor"] == step and state["provenance"] == provenance,
                "Checkpoint cursor/provenance differ")
        finite_tree(state)
        adapter = state["adapter"]
        require(sum(t.numel() for t in adapter.values()) == contract["trainable_parameters"],
                "Trainable parameter count differs")
        require(all(t.dtype == torch.float32 for t in adapter.values()),
                "Adapter is not FP32")
        require(len(state["cuda_rng"]) == 2 and state["torch_rng"].dtype == torch.uint8,
                "Missing CPU/CUDA RNG state")
        random.Random().setstate(state["python_rng"])
        groups = state["optimizer"]["param_groups"]
        require(len(groups) == 1 and groups[0]["lr"] == 1e-4
                and groups[0]["weight_decay"] == 0., "Optimizer configuration differs")
        optimizer = state["optimizer"]["state"]
        require(len(optimizer) == (0 if step == 0 else len(adapter)),
                "Optimizer does not cover the trained adapter")
        for item in optimizer.values():
            require(int(item["step"].item()) == step, "Optimizer step differs from cursor")
        if step == 0:
            require(adapter_digest(adapter) == contract["initial_adapter_sha256"],
                    "Initial adapter digest differs")
            initial_adapter = adapter
            rng = hashlib.sha256(state["torch_rng"].numpy().tobytes())
            for value in state["cuda_rng"]:
                rng.update(value.numpy().tobytes())
            rng.update(repr(state["python_rng"]).encode())
            result["initial_rng_sha256"] = rng.hexdigest()
        else:
            require(adapter.keys() == initial_adapter.keys(), "Checkpoint attachment keys differ")
            require(all(t.shape == initial_adapter[k].shape for k, t in adapter.items()),
                    "Checkpoint adapter shape differs")
        result[name] = metadata
        if step == 256:
            final_adapter = adapter
    require(any(not torch.equal(t, initial_adapter[k]) for k, t in final_adapter.items()),
            "Final adapter did not change")
    config = read(cell / "adapter/adapter_config.json")
    require(config["r"] == 16 and config["lora_alpha"] == 32 and config["lora_dropout"] == 0.
            and set(config["target_modules"]) == {"o_proj", "out_proj", "down_proj"},
            "Exported LoRA configuration differs")
    from safetensors.torch import load_file
    exported = load_file(str(cell / "adapter/adapter_model.safetensors"))
    normalized = {k.replace(".default.", "."): t for k, t in final_adapter.items()}
    require(exported.keys() == normalized.keys()
            and all(torch.equal(t, normalized[k]) for k, t in exported.items()),
            "Exported adapter differs from step256 checkpoint")
    result["export_sha256"] = sha(cell / "adapter/adapter_model.safetensors")
    return result


def paired_stats(values):
    """Uncertainty across paired seeds, conditional on the same fixed cases."""
    require(len(values) == 3, "Exactly three paired seed effects are required")
    mean = statistics.mean(values)
    sd = statistics.stdev(values)
    half = 4.302652729911275 * sd / math.sqrt(3)  # t(2), two-sided 95%
    return {"values": values, "mean": mean, "seed_sd": sd,
            "t95_across_seeds": [mean - half, mean + half], "paired_seeds": 3,
            "unique_verification_cases": 64,
            "uncertainty_scope": "across three seeds, conditional on the fixed 64 cases; df=2"}


def frozen_package(root):
    require(sha(root / "code.tar.gz") == PACKAGE_SHA256, "Frozen package hash differs")
    manifest = read(root / "source_manifest.json")
    with tarfile.open(root / "code.tar.gz") as archive:
        members = archive.getmembers()
        require(len(members) == 16 and len(manifest) == 16
                and {m.name for m in members} == set(manifest),
                "Frozen package members differ")
        blobs = {}
        for member in members:
            require(member.isfile(), "Non-file frozen package member")
            blob = archive.extractfile(member).read()
            require(hashlib.sha256(blob).hexdigest() == manifest[member.name],
                    "Frozen package member hash differs")
            blobs[member.name] = blob
    require(hashlib.sha256((root / "protocol.json").read_bytes()).hexdigest()
            == manifest["protocol.json"], "Local frozen protocol changed")
    return manifest, blobs


def tokenizer_for_audit(root, blobs):
    """Download only three pinned tokenizer/config files, never model weights."""
    from transformers import AutoTokenizer
    models = json.loads(blobs["posttrain_causal_lab/assets/models.json"])
    student = next(model for model in models["models"] if model["model"] == "Qwen/Qwen3.5-4B")
    directory = root / "analysis-tokenizer"
    directory.mkdir(exist_ok=True)
    for item in student["files"]:
        if item["path"] not in {"config.json", "tokenizer.json", "tokenizer_config.json"}:
            continue
        path = directory / item["path"]
        if not path.exists() or sha(path) != item["sha256"]:
            temporary = path.with_suffix(".download")
            with urllib.request.urlopen(item["url"], timeout=60) as response, temporary.open("wb") as stream:
                while chunk := response.read(1024 * 1024):
                    stream.write(chunk)
            require(temporary.stat().st_size == item["size"] and sha(temporary) == item["sha256"],
                    "Pinned tokenizer download hash differs")
            temporary.replace(path)
    tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
    config = read(directory / "config.json")
    eos = config.get("text_config", config)["eos_token_id"]
    return tokenizer, [eos] if isinstance(eos, int) else eos


def verify_cell(cell, spec, manifest, data, tokenizer, eos_ids):
    contract = read(cell / "contract.json")
    completed = read(cell / "completed.json")
    require(completed["complete"] is True and completed["recovery_passed"] is True
            and completed["steps"] == 256, "Cell did not complete all gates/steps")
    for record in (contract, completed):
        require(record["arm"] == spec["arm"] and record["seed"] == spec["seed"],
                "Cell identity differs from worker layout")
    require(contract["split_counts"] == {"train": 256, "dev": 64, "verification": 64}
            and contract["benchmark_test_accessed"] is False
            and contract["evaluation_budget"] == 256, "Data/evaluation contract differs")
    require(contract["student_dtype"] == "float32" and contract["teacher_dtype"] == "bfloat16"
            and contract["learning_rate"] == 1e-4
            and len(contract["gpu_names"]) == 2
            and all("H20" in name for name in contract["gpu_names"]),
            "Hardware/precision/optimizer contract differs")
    for field, name in (("protocol_sha256", "protocol.json"),
                        ("data_manifest_sha256", "posttrain_causal_lab/data/manifest.json"),
                        ("models_manifest_sha256", "posttrain_causal_lab/assets/models.json")):
        require(contract[field] == manifest[name], "Cell source provenance differs: " + field)
    train = list(data["train"])
    random.Random(spec["seed"]).shuffle(train)
    order = [r["id"] for r in train]
    require(contract["train_order"] == order and contract["train_order_sha256"]
            == hashlib.sha256(json.dumps(order).encode()).hexdigest(), "Train order differs")
    budget = 32 if spec["arm"] == "opd32" else 256
    require(contract["rollout_budget"] == budget, "Rollout budget differs")
    lora_sites = contract["lora_sites"]
    require(len(lora_sites) == 64 and len(set(lora_sites)) == 64
            and all(name.endswith((".o_proj", ".out_proj", ".down_proj")) for name in lora_sites),
            "LoRA attachment sites differ")
    provenance_fields = ("seed", "arm", "protocol_sha256", "data_manifest_sha256",
                         "models_manifest_sha256", "train_order_sha256", "lora_sites",
                         "rollout_budget", "learning_rate", "student_dtype", "teacher_dtype")
    checkpoints = verify_checkpoints(cell, contract, {k: contract[k] for k in provenance_fields})
    events = records(cell / "events.jsonl")
    steps = [r for r in events if r["event"] == "train_step"]
    require([r["step"] for r in steps] == list(range(1, 257))
            and [r["id"] for r in steps] == order
            and all(r["arm"] == spec["arm"] for r in steps), "Training trajectory differs")
    for row in steps:
        require(all(math.isfinite(row[k]) for k in ("loss", "grad_norm", "elapsed_seconds"))
                and row["grad_norm"] > 0 and row["elapsed_seconds"] >= 0
                and 0 < row["response_tokens"] <= 256
                and row["prefix_tokens"] + row["response_tokens"] <= 768,
                "Invalid training step")
    require(events[-1]["event"] == "cell_complete", "Missing terminal cell event")
    recovery = read(cell / "recovery.json")
    require(len(recovery) == 1 and recovery[0]["step"] == 128
            and recovery[0]["scope"] == "same-process fixed-prefix replay",
            "Recovery gate scope differs")
    gate = recovery[0]
    require(math.isfinite(gate["adapter_max_abs"]) and gate["adapter_max_abs"] >= 0,
            "Invalid replay adapter error")
    close(gate["reference"]["loss"], gate["replay"]["loss"], "Replay loss differs",
          atol=1e-6, rtol=0)
    metrics = read(cell / "metrics.json")
    computed = {}
    generations = {}
    for phase in ("base_dev", "trained_dev", "trained_verification", "base_verification"):
        split = "verification" if phase.endswith("verification") else "dev"
        values = records(cell / (phase.replace("_", "-") + ".jsonl"))
        computed[phase] = verify_generations(values, data[split], tokenizer, eos_ids, 256)
        check_summary(metrics[phase], computed[phase], phase)
        generations[phase] = values
    prefix = {}
    for phase in ("base_dev_prefix", "trained_dev_prefix", "trained_verification_prefix"):
        split = "verification" if "verification" in phase else "dev"
        values = records(cell / (phase.replace("_", "-") + ".jsonl"))
        require([r["id"] for r in values] == [r["id"] for r in data[split]],
                "Prefix diagnostics IDs differ")
        require(all(type(r["tokens"]) is int and r["tokens"] > 0
                    and math.isfinite(r["gold_nll"]) and r["gold_nll"] >= 0
                    and math.isfinite(r["teacher_kl"]) and r["teacher_kl"] >= -1e-5
                    for r in values), "Invalid prefix diagnostic")
        tokens = sum(r["tokens"] for r in values)
        prefix[phase] = {"n": len(values), "tokens": tokens, **{
            k: sum(r[k] * r["tokens"] for r in values) / tokens for k in ("gold_nll", "teacher_kl")}}
        check_summary(metrics[phase], prefix[phase], phase)
    require(metrics == completed["metrics"], "Completion metrics differ from metric file")
    rollout_summary = None
    if spec["arm"].startswith("opd"):
        rollouts = records(cell / "rollouts.jsonl")
        require([r["step"] for r in rollouts] == list(range(256)), "Rollout step sequence differs")
        rollout_summary = verify_generations(rollouts, train, tokenizer, eos_ids, budget)
        require([r["response_tokens"] for r in steps] == [r["output_tokens"] for r in rollouts],
                "Rollout lengths differ from supervised response lengths")
    else:
        require(not (cell / "rollouts.jsonl").exists(), "Unexpected SFT/KD rollouts")
    dev = records(cell / "dev-interventions.jsonl")
    names = list(dict.fromkeys(r["site"] for r in dev if r["mode"] == "base"))
    require(len(names) == 64
            and all(sum(n.startswith(f"layer{i:02d}.") for n in names) == 2 for i in range(32)),
            "Intervention search does not cover all 32 layers")
    dev_result = verify_interventions(dev, [r["id"] for r in data["dev"][:8]], names)
    selection = read(cell / "selected-sites.json")
    selected = choose_sites(dev)
    require(selection["selection_split"] == "dev" and selection["n_search"] == 8
            and selection["sites"] == selected and len(selected) == 3
            and completed["selected_sites"] == selected, "Dev-only site selection differs")
    verification = verify_interventions(
        records(cell / "verification-interventions.jsonl"),
        [r["id"] for r in data["verification"]], selected)
    require(math.isfinite(completed["elapsed_seconds"]) and completed["elapsed_seconds"] > 0,
            "Invalid cell elapsed time")
    return {"arm": spec["arm"], "seed": spec["seed"], "metrics": computed,
            "prefix_metrics": prefix, "rollouts": rollout_summary,
            "dev_interventions": dev_result, "verification_interventions": verification,
            "recovery": gate, "checkpoints": checkpoints,
            "initial_adapter_sha256": contract["initial_adapter_sha256"],
            "train_order_sha256": contract["train_order_sha256"],
            "versions": contract["versions"], "elapsed_seconds": completed["elapsed_seconds"],
            "peak_allocated_bytes": completed["peak_allocated_bytes"]}, generations


def cost_accounting(root):
    """Use recorded Pod schedule→stop intervals; never substitute cell runtime."""
    trials_path = root / "trials.json"
    if not trials_path.exists():
        return {"complete": False, "gpu_hours": None, "reason": "All-trial evidence missing"}
    trials = read(trials_path)
    rows, complete, total = [], True, 0.
    for trial in trials["data"]:
        directory = root / "trial-evidence" / str(trial["sid"])
        detail = read(directory / "trial.json")["data"]
        pods = read(directory / "pods.json")
        require(pods.get("pagination", {}).get("total", 0) <= len(pods["data"]),
                "Paginated Pod cost evidence")
        if not pods["data"]:
            # Null started_at, a terminal state, and no pods together support
            # zero recorded allocation. "stopped" alone would not.
            zero = detail.get("started_at") is None and detail.get("status") in (4, 5, 6, 7)
            complete &= zero
            rows.append({"trial": str(trial["sid"]), "pods": 0, "gpu_hours": 0. if zero else None,
                         "basis": "terminal, no started_at, no recorded pods" if zero
                         else "No closed allocation evidence"})
        for pod in pods["data"]:
            start = pod.get("schedule_at") or pod.get("scheduled_at")
            end = pod.get("stop_at") or pod.get("stopped_at")
            if start and end:
                seconds = (datetime.fromisoformat(end.replace("Z", "+00:00"))
                           - datetime.fromisoformat(start.replace("Z", "+00:00"))).total_seconds()
                require(seconds >= 0, "Negative Pod allocation interval")
                hours = seconds * 8 / 3600
                total += hours
            else:
                complete, hours = False, None
            rows.append({"trial": str(trial["sid"]), "pod": pod.get("sid"),
                         "schedule_at": start, "stop_at": end, "gpu_hours": hours})
    return {"complete": complete, "gpu_hours": total if complete else None,
            "known_closed_gpu_hours": total, "trials": rows,
            "basis": "8 GPUs per Pod, schedule to stop; all attempts; no queue cost"}


def analyze(root):
    manifest, blobs = frozen_package(root)
    tokenizer, eos_ids = tokenizer_for_audit(root, blobs)
    data = {split: [json.loads(line) for line in blobs[
        "posttrain_causal_lab/data/" + split + ".jsonl"].decode().splitlines()]
        for split in ("train", "dev", "verification")}
    sets = [set(r["id"] for r in data[split]) for split in data]
    require([len(s) for s in sets] == [256, 64, 64]
            and len(set.union(*sets)) == 384, "Frozen dataset IDs overlap or repeat")
    artifacts = read(root / "artifacts.json")
    require(len(artifacts) == 3 and {r["worker"] for r in artifacts} == {0, 1, 2}
            and all(r["complete"] for r in artifacts), "All three worker archives are required")
    run = read(root / "run.json")["data"]
    trial = str(run["runtime_context"]["trial_id"])
    require(run["status"] in ("done", "success")
            and all(r["trial"] == trial for r in artifacts), "Run/archive terminal identity differs")
    cells, raw = [], {}
    for archive in sorted(artifacts, key=lambda r: r["worker"]):
        worker = root / archive["directory"]
        require(worker.resolve().is_relative_to(root.resolve()), "Archive directory escapes experiment")
        path = root / f"trial-{trial}-trialrun-{trial}-worker-{archive['worker']}.tar.gz"
        require(path.stat().st_size == archive["bytes"] and sha(path) == archive["sha256"],
                "Collected archive bytes/hash differ")
        require(read(worker / "extracted.json")["sha256"] == archive["sha256"],
                "Extraction marker differs from archive hash")
        with tarfile.open(path) as stream:
            for member in stream:
                target = worker / member.name
                require(target.resolve().is_relative_to(worker.resolve())
                        and (member.isfile() or member.isdir()), "Unsafe archived output path")
                if member.isfile():
                    source = stream.extractfile(member)
                    require(hashlib.file_digest(source, "sha256").hexdigest() == sha(target),
                            "Extracted output differs from original archive: " + member.name)
        require(read(worker / "results/source_manifest.json") == manifest,
                "Worker source manifest differs")
        for name, digest in manifest.items():
            require(sha(worker / "code" / name) == digest, "Worker source hash differs: " + name)
        require((worker / "results/exit_code.txt").read_text().strip() == "0",
                "Worker entrypoint failed")
        contracts_log = (worker / "results/contracts.log").read_text()
        require("Ran 13 tests" in contracts_log and "\nOK" in contracts_log,
                "Worker CPU contract gates missing")
        directory = worker / "results/cells"
        layout = cell_layout(archive["worker"])
        require(read(directory / "layout.json") == layout, "Worker cell layout differs")
        require(read(directory / "worker-status.json")
                == [{**spec, "returncode": 0} for spec in layout], "Experimental child failed")
        for spec in layout:
            name = f"{spec['arm']}-seed{spec['seed']}"
            result, generations = verify_cell(directory / name, spec, manifest, data, tokenizer, eos_ids)
            cells.append(result)
            raw[spec["seed"], spec["arm"]] = generations
    require(len({json.dumps(c["versions"], sort_keys=True) for c in cells}) == 1,
            "Runtime dependency versions differ across cells")
    for seed in SEEDS:
        group = [c for c in cells if c["seed"] == seed]
        for key in ("initial_adapter_sha256", "train_order_sha256"):
            require(len({c[key] for c in group}) == 1, "Within-seed pairing differs: " + key)
        require(len({c["checkpoints"]["initial_rng_sha256"] for c in group}) == 1,
                "Within-seed initial CPU/CUDA/Python RNG differs")
        for split in ("base_dev", "base_verification"):
            signatures = [json.dumps([(r["id"], r["token_ids"]) for r in raw[seed, arm][split]])
                          for arm in ARMS]
            require(len(set(signatures)) == 1, "Within-seed baseline generation differs")
    indexed = {(c["seed"], c["arm"]): c for c in cells}
    pooled = {}
    for arm in ARMS:
        values = [indexed[seed, arm] for seed in SEEDS]
        pooled[arm] = {"seed_count": 3, "unique_verification_cases": 64,
                       "evaluations": 192, "verification": {
                           key: statistics.mean(c["metrics"]["trained_verification"][key] for c in values)
                           for key in METRICS},
                       "interventions": {key: statistics.mean(c["verification_interventions"][key]
                                                             for c in values)
                                         for key in ("base_drop", "random_drop", "base_minus_random")}}
        if arm.startswith("opd"):
            pooled[arm]["rollouts"] = {key: statistics.mean(c["rollouts"][key] for c in values)
                                       for key in METRICS}
    paired = {key: paired_stats([
        indexed[seed, "opd256"]["metrics"]["trained_verification"][key]
        - indexed[seed, "opd32"]["metrics"]["trained_verification"][key] for seed in SEEDS])
        for key in METRICS}
    paired["rollout_truncated"] = paired_stats([
        indexed[seed, "opd256"]["rollouts"]["truncated"]
        - indexed[seed, "opd32"]["rollouts"]["truncated"] for seed in SEEDS])
    paired["rollout_truncated"]["unique_training_cases"] = 256
    paired["rollout_truncated"].pop("unique_verification_cases")
    paired["rollout_truncated"]["uncertainty_scope"] = "three seeds, fixed 256 training cases"
    for arm in ARMS:
        paired[arm + "_intervention_base_minus_random"] = paired_stats([
            indexed[seed, arm]["verification_interventions"]["base_minus_random"] for seed in SEEDS])
    case_pairs = [{"id": row["id"], "effects_by_seed": [
        int(raw[seed, "opd256"]["trained_verification"][i]["complete_numeric_correct"])
        - int(raw[seed, "opd32"]["trained_verification"][i]["complete_numeric_correct"])
        for seed in SEEDS]} for i, row in enumerate(data["verification"])]
    return {"audit_passed": True, "source_sha256": PACKAGE_SHA256, "trial": trial,
            "cells": cells, "pooled_descriptives": pooled, "paired_effects": paired,
            "primary_case_pairs": case_pairs, "cost": cost_accounting(root),
            "scope": "Three paired seeds, fixed 64 verification cases; no human reasoning audit, "
                     "no historical train-signal intervention, no cross-machine recovery"}


def report(result):
    effect = result["paired_effects"]["complete_numeric_correct"]
    lo, hi = effect["t95_across_seeds"]
    text = [
        "# 24-H20 配对实验结果",
        "",
        f"十二个 cell 的产物与数值门禁通过。OPD-256 相对 OPD-32 的完整数字正确率变化为"
        f" {100 * effect['mean']:+.2f} 个百分点；三个 seed 的差值为 "
        + "、".join(f"{100 * x:+.2f}" for x in effect["values"]) + " 个百分点。",
        "",
        "![输出预算、验证质量和固定前缀干预](results.png)",
        "",
        "图：每点代表一个 seed。质量评测统一使用贪心生成、256-token 上限；"
        "干预分数为在 dev 选出的三个位置上，verification 平均 gold log-prob 下降，"
        "单位 nats/token。右图 donor 与等 L2 随机扰动相减，不能解释为自由生成正确率差。",
        "",
        "| 条件 | 完整数字正确率 | 数字正确率 | 格式有效率 | 评测截断率 | 平均输出 token |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        m = result["pooled_descriptives"][arm]["verification"]
        text.append(f"| {arm} | {100*m['complete_numeric_correct']:.2f}% | "
                    f"{100*m['numeric_correct']:.2f}% | {100*m['format_valid']:.2f}% | "
                    f"{100*m['truncated']:.2f}% | {m['output_tokens']:.2f} |")
    text += [
        "",
        f"按三个配对 seed 计算的均值 t 区间为 [{100*lo:+.2f}, {100*hi:+.2f}] 个百分点"
        "（95%，df=2）。它只描述固定 64 题上的 seed 不确定性。"
        "各条件有 192 次评测，但独立 verification 题目只有 64 道；"
        "小 seed 数使该区间对分布假设敏感。",
        "",
        "该主对照衡量 32→256 输出预算改变后的完整 OPD 配方效果。"
        "采样路径和监督 token 数也随之改变，不能据此隔离等 token 的优化收益。"
        "评分按最后一个 `####` 数字、否则最后一个数字与官方答案比较；"
        "主要指标还要求 EOS 完成。它是 parser 代理，没有人工审查推理过程。",
        "",
        "| Seed | 条件 | 训练 rollout 截断率 | 验证 donor−random，nats/token | dev 选出的模块 |",
        "|---|---|---:|---:|---|",
    ]
    for cell in result["cells"]:
        rollout = f"{100*cell['rollouts']['truncated']:.2f}%" if cell["rollouts"] else "—"
        sites = ", ".join(cell["verification_interventions"]["sites"])
        text.append(f"| {cell['seed']} | {cell['arm']} | {rollout} | "
                    f"{cell['verification_interventions']['base_minus_random']:+.6f} | {sites} |")
    cost = result["cost"]
    cost_text = (f"{cost['gpu_hours']:.3f} GPU·h" if cost["complete"]
                 else f"尚不能完整确定；已闭合区间合计 {cost.get('known_closed_gpu_hours', 0):.3f} GPU·h")
    text += [
        "",
        "固定前缀 patching 测量的是指定内部计算对 gold token 概率的作用。"
        "底座 donor 与随机扰动在激活 L2 强度上匹配，位置只在 dev 搜索，"
        "verification 不参与选址。即使 donor 效果超过随机，也不能证明自由生成质量"
        "由相同模块介导，更不能识别训练信号的历史原因。",
        "",
        "每 cell 完成 256 步、64 dev 和64 verification；"
        "dev 干预 1032 条、verification 干预448条；"
        "三个 Worker 各自重新运行13项 CPU门禁。"
        "复核包含16个冻结源文件、初始 adapter与样本顺序配对、checkpoint hash／有限性／游标、"
        "导出 adapter一致性、原始 token重新解码与评分，以及 dev-only top3选址。",
        "",
        "恢复为同进程固定前缀重放。完整 logits未归档，因此 identity与重放的逐元素"
        "相对容差依赖冻结源码中的运行时断言；离线复核只能再次检查保存的最大误差和标量。"
        "恢复门禁不构成跨 Worker故障接管验证。",
        "",
        f"实际分配成本：{cost_text}。成本使用所有 trial的 Pod schedule→stop区间，"
        "包含准备、诊断额外计算和归档；排队不计入。入口上限72 GPU·h不代表实际成本。",
        "",
        "完整逐 seed指标、配对题目差值、checkpoint哈希和成本证据索引见 [analysis.json](analysis.json)。"
        "这些结果适用于 Qwen3.5-4B FP32学生、9B BF16教师与当前 LoRA／Torch后端；"
        "尚未完成更大模型、低精度后端或训练信号修复验证。",
        "",
    ]
    return "\n".join(text)


def render(result, root):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"sft": "#587ca6", "kd": "#7d799f", "opd32": "#ca7850", "opd256": "#428878"}
    indexed = {(c["seed"], c["arm"]): c for c in result["cells"]}
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), layout="constrained")
    panels = (
        ("Training rollout truncation", "Percent", ("opd32", "opd256"),
         lambda c: 100 * c["rollouts"]["truncated"]),
        ("Verification complete numeric accuracy", "Percent", ARMS,
         lambda c: 100 * c["metrics"]["trained_verification"]["complete_numeric_correct"]),
        ("Selected sites: donor minus random", "Gold log-prob drop (nats/token)", ARMS,
         lambda c: c["verification_interventions"]["base_minus_random"]),
    )
    for ax, (title, ylabel, arms, value) in zip(axes, panels):
        for x, arm in enumerate(arms):
            ys = [value(indexed[seed, arm]) for seed in SEEDS]
            ax.scatter([x-.08, x, x+.08], ys, color=colors[arm], s=35, zorder=3)
            ax.plot([x-.18, x+.18], [statistics.mean(ys)]*2, color=colors[arm], lw=2)
        ax.set_xticks(range(len(arms)), [arm.upper() for arm in arms])
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.18)
    axes[0].set_ylim(-3, 103)
    axes[1].set_ylim(-3, 103)
    axes[2].axhline(0, color="#666", lw=.7)
    fig.suptitle("Qwen3.5 4B/9B · 256 updates · 3 paired seeds · 64 fixed verification cases", fontsize=11)
    for extension in ("png", "svg"):
        fig.savefig(root / ("results." + extension), dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    root = args.experiment.resolve()
    try:
        result = analyze(root)
    except (AuditError, OSError, KeyError, TypeError, RuntimeError) as error:
        write_json(root / "audit.json", {"audit_passed": False,
                                        "error": f"{type(error).__name__}: {error}"})
        raise
    write_json(root / "analysis.json", result)
    write_json(root / "audit.json", {"audit_passed": True, "cells": 12, "source_files": 16})
    if not args.no_plot:
        render(result, root)
    (root / "REPORT.md").write_text(report(result))
    print(json.dumps({"audit_passed": True, "cells": 12,
                      "primary_effect": result["paired_effects"]["complete_numeric_correct"],
                      "cost": result["cost"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
