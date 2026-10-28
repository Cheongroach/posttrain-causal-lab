"""Re-evaluate immutable OPD checkpoints on dev with paired output budgets."""
import argparse
import hashlib
import importlib.metadata
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import torch

from .case_rng import case_sampling
from .pilot import adapter_state, attach, event, load, restore_adapter, write
from .scale24 import generate, prompt_tokens, summarize


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require_hash(path, expected):
    if digest(path) != expected:
        raise ValueError("Frozen input hash differs: " + path.name)


def prefix_gate(short, long):
    if short["id"] != long["id"] or short["token_ids"] != long["token_ids"][:len(short["token_ids"])]:
        raise ValueError("Budget changes the shared greedy prefix")


def rng_probe(student, tokenizer, row, eos_ids, seed):
    cpu = torch.get_rng_state().clone()
    gpu = torch.cuda.get_rng_state(0).clone()
    python = random.getstate()
    with case_sampling(seed, row["id"], cuda_devices=[0]) as derived:
        _, short = generate(student, tokenizer, row, eos_ids, 8, True)
    with case_sampling(seed, "unused-extra-case", cuda_devices=[0]):
        torch.rand(53, device="cuda:0")
        torch.rand(37)
        random.random()
    with case_sampling(seed, row["id"], cuda_devices=[0]):
        _, long = generate(student, tokenizer, row, eos_ids, 16, True)
    prefix_gate(short, long)
    if (not torch.equal(cpu, torch.get_rng_state())
            or not torch.equal(gpu, torch.cuda.get_rng_state(0))
            or python != random.getstate()):
        raise ValueError("Per-case sampling changed the surrounding RNG stream")
    return {"passed": True, "example_id": row["id"], "derived_seed": derived,
            "budgets": [8, 16], "shared_prefix_equal": True, "outer_rng_restored": True,
            "short_token_ids": short["token_ids"], "long_token_ids": long["token_ids"],
            "scope": "real-model CUDA sampling contract; no paired RNG retraining yet"}


def run_cell(args):
    from safetensors.torch import load_file
    from transformers import AutoTokenizer
    if torch.cuda.device_count() != 1 or "H20" not in torch.cuda.get_device_name(0):
        raise ValueError("Each length-probe cell requires one H20")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    package = Path(__file__).parent
    protocol = json.loads(args.protocol.read_text())
    manifest = json.loads((args.snapshots / "snapshot-manifest.json").read_text())
    spec = manifest["cells"][f"{args.arm}-seed{args.seed}"]
    require_hash(args.data, protocol["dev_sha256"])
    rows = [json.loads(line) for line in args.data.read_text().splitlines()]
    if len(rows) != 64 or len({r["id"] for r in rows}) != 64:
        raise ValueError("Expected the frozen 64 dev cases")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    student = attach(load(args.model, "cuda:0"))
    eos_ids = student.generation_config.eos_token_id
    eos_ids = [eos_ids] if isinstance(eos_ids, int) else eos_ids
    if eos_ids != [248044]:
        raise ValueError("Preserve the original model EOS contract")
    for row in rows:
        if len(prompt_tokens(tokenizer, row)) + 512 > protocol["max_total_tokens"]:
            raise ValueError("Prompt exceeds the extended total context allowance")
    initial_shapes = {k: tuple(t.shape) for k, t in adapter_state(student).items()}
    contract = {"arm": args.arm, "seed": args.seed, "rows": 64, "split": "dev",
                "checkpoints": protocol["checkpoints"], "budgets": protocol["budgets"],
                "protocol_sha256": digest(args.protocol), "snapshot_spec": spec,
                "dev_sha256": digest(args.data), "eos_ids": eos_ids,
                "tokenizer_eos_id": tokenizer.eos_token_id, "dtype": str(student.dtype),
                "gpu": torch.cuda.get_device_name(0),
                "model_manifest_sha256": digest(package / "assets/models.json"),
                "versions": {name: importlib.metadata.version(name) for name in
                             ("torch", "transformers", "peft", "tokenizers")},
                "optimizer_updates": 0, "verification_accessed": False,
                "benchmark_test_accessed": False}
    write(args.output / "contract.json", contract)
    metrics = {}
    reference_path = args.snapshots / spec["original_dev"]["path"]
    require_hash(reference_path, spec["original_dev"]["sha256"])
    reference = {r["id"]: r for r in (
        json.loads(line) for line in reference_path.read_text().splitlines())}
    for step in protocol["checkpoints"]:
        item = spec["snapshots"][str(step)]
        path = args.snapshots / item["path"]
        require_hash(path, item["sha256"])
        adapter = load_file(str(path))
        if (set(adapter) != set(initial_shapes)
                or any(tuple(t.shape) != initial_shapes[k] or not bool(torch.isfinite(t).all())
                       for k, t in adapter.items())):
            raise ValueError("Invalid adapter snapshot")
        restore_adapter(student, adapter)
        values = {budget: [] for budget in protocol["budgets"]}
        for row in rows:
            pair = []
            for budget in protocol["budgets"]:
                _, result = generate(student, tokenizer, row, eos_ids, budget, False)
                record = {"step": step, "budget": budget, **result}
                with (args.output / "generations.jsonl").open("a") as stream:
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                values[budget].append(result)
                pair.append(result)
            prefix_gate(*pair)
            if step == 256 and pair[0]["token_ids"] != reference[row["id"]]["token_ids"]:
                raise ValueError("Step256 / budget256 no longer reproduces the original dev output")
        metrics[str(step)] = {str(budget): summarize(records) for budget, records in values.items()}
        write(args.output / "metrics.json", metrics)
        event(args.output, event="checkpoint_evaluation_complete", step=step,
              arm=args.arm, seed=args.seed, metrics=metrics[str(step)])
    rng = rng_probe(student, tokenizer, rows[0], eos_ids, args.seed)
    write(args.output / "rng-probe.json", rng)
    write(args.output / "completed.json", {
        "complete": True, "arm": args.arm, "seed": args.seed, "metrics": metrics,
        "greedy_prefix_passed": True, "original_step256_dev_reproduced": True,
        "rng_probe_passed": True, "optimizer_updates": 0,
        "elapsed_seconds": time.monotonic() - started,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(0)})


def main():
    parser = argparse.ArgumentParser()
    for name in ("model", "data", "snapshots", "protocol", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--worker-index", type=int, choices=range(3))
    parser.add_argument("--arm", choices=("opd32", "opd256"))
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    if args.worker_index is None:
        if args.arm is None or args.seed is None:
            parser.error("Cell mode requires --arm and --seed")
        run_cell(args)
        return
    if torch.cuda.device_count() != 2:
        raise ValueError("Worker mode requires exactly two H20 GPUs")
    args.output.mkdir(parents=True, exist_ok=True)
    seed = 20261027 + args.worker_index
    children = []
    layout = [{"arm": arm, "seed": seed, "gpu": index}
              for index, arm in enumerate(("opd32", "opd256"))]
    write(args.output / "layout.json", layout)
    for spec in layout:
        name = f"{spec['arm']}-seed{seed}"
        stream = (args.output / (name + ".log")).open("wb")
        command = [sys.executable, "-u", "-m", "posttrain_causal_lab.length_probe",
                   "--arm", spec["arm"], "--seed", str(seed),
                   "--model", str(args.model), "--data", str(args.data),
                   "--snapshots", str(args.snapshots), "--protocol", str(args.protocol),
                   "--output", str(args.output / name)]
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                   env={**os.environ, "CUDA_VISIBLE_DEVICES": str(spec["gpu"])})
        children.append((spec, process, stream))
    statuses = []
    for spec, process, stream in children:
        returncode = process.wait()
        stream.close()
        statuses.append({**spec, "returncode": returncode})
    write(args.output / "worker-status.json", statuses)
    if any(r["returncode"] for r in statuses):
        raise RuntimeError("An evaluation cell failed")


if __name__ == "__main__":
    main()
