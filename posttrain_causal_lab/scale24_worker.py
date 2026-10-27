"""Four disjoint two-GPU experimental cells on one eight-GPU worker."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from .pilot import write
from .scale_contracts import cell_layout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-index", type=int, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    args = parser.parse_args()
    import torch
    if torch.cuda.device_count() != 8:
        raise RuntimeError("Each worker must own exactly eight GPUs")
    specs = cell_layout(args.worker_index)
    args.output.mkdir(parents=True, exist_ok=True)
    write(args.output / "layout.json", specs)
    children = []
    for spec in specs:
        name = f"{spec['arm']}-seed{spec['seed']}"
        log = (args.output / f"{name}.log").open("wb")
        command = [sys.executable, "-u", "-m", "posttrain_causal_lab.scale24",
                   "--arm", spec["arm"], "--seed", str(spec["seed"]),
                   "--student", str(args.models / "Qwen3.5-4B"),
                   "--teacher", str(args.models / "Qwen3.5-9B"),
                   "--data", str(args.data), "--output", str(args.output / name),
                   "--protocol", str(args.protocol)]
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": ",".join(map(str, spec["gpus"]))}
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        children.append((spec, process, log))
    statuses = []
    for spec, process, log in children:
        status = process.wait()
        log.close()
        statuses.append({**spec, "returncode": status})
    write(args.output / "worker-status.json", statuses)
    if any(item["returncode"] for item in statuses):
        raise RuntimeError("An experimental cell failed; inspect per-cell logs")
    print(json.dumps({"event": "all_four_cells_completed", "worker": args.worker_index}), flush=True)


if __name__ == "__main__":
    main()
