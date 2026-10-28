"""Verify length-probe outputs and distinguish budget from checkpoint effects."""
import argparse
import hashlib
import json
import statistics
import tarfile
from datetime import datetime
from pathlib import Path

from .analyze_scale24 import (check_summary, require, read, records, sha,
                              verify_generations, write_json)
from .length_probe import prefix_gate
from .case_rng import sampling_seed


def transitions(short, long):
    require(len(short) == len(long), "Paired generation count differs")
    result = {"n": len(short), "rescued_correct": 0, "finished_wrong": 0,
              "still_truncated": 0, "already_finished": 0, "regressed_correct": 0}
    for before, after in zip(short, long):
        prefix_gate(before, after)
        if before["truncated"]:
            if after["truncated"]:
                result["still_truncated"] += 1
            elif after["complete_numeric_correct"]:
                result["rescued_correct"] += 1
            else:
                result["finished_wrong"] += 1
        else:
            require(before["token_ids"] == after["token_ids"],
                    "An already completed output changed with the budget")
            result["already_finished"] += 1
        result["regressed_correct"] += bool(
            before["complete_numeric_correct"] and not after["complete_numeric_correct"])
    require(result["regressed_correct"] == 0, "Longer greedy prefix regressed a completed answer")
    return result


def gpu_cost(root):
    totals, rows, complete = 0., [], True
    for trial in read(root / "trials.json")["data"]:
        directory = root / "trial-evidence" / str(trial["sid"])
        detail = read(directory / "trial.json")["data"]
        pods = read(directory / "pods.json")["data"]
        if not pods:
            zero = detail.get("started_at") is None and detail.get("status") in (4, 5, 6, 7)
            complete &= zero
            rows.append({"trial": str(trial["sid"]), "gpu_hours": 0. if zero else None})
        for pod in pods:
            start, end = pod.get("schedule_at"), pod.get("stop_at")
            if not start or not end:
                complete = False
                rows.append({"trial": str(trial["sid"]), "pod": pod["sid"], "gpu_hours": None})
                continue
            seconds = (datetime.fromisoformat(end.replace("Z", "+00:00"))
                       - datetime.fromisoformat(start.replace("Z", "+00:00"))).total_seconds()
            require(seconds >= 0, "Negative allocation interval")
            hours = seconds * 2 / 3600
            totals += hours
            rows.append({"trial": str(trial["sid"]), "pod": pod["sid"],
                         "schedule_at": start, "stop_at": end, "gpu_hours": hours})
    return {"complete": complete, "gpu_hours": totals if complete else None,
            "known_closed_gpu_hours": totals, "intervals": rows, "gpus_per_pod": 2}


def analyze(root):
    from transformers import AutoTokenizer
    protocol = read(root / "protocol.json")
    manifest = read(root / "source_manifest.json")
    artifacts = read(root / "artifacts.json")
    run = read(root / "run.json")["data"]
    trial = str(run["runtime_context"]["trial_id"])
    require(run["status"] in ("done", "success"), "Run did not finish successfully")
    require(len(artifacts) == 3 and {r["worker"] for r in artifacts} == {0, 1, 2}
            and all(r["complete"] and r["trial"] == trial for r in artifacts),
            "Three matching worker archives required")
    tokenizer = AutoTokenizer.from_pretrained(root / "analysis-tokenizer", local_files_only=True)
    cells = []
    for item in artifacts:
        worker = root / item["directory"]
        archive = root / f"trial-{trial}-trialrun-{trial}-worker-{item['worker']}.tar.gz"
        require(sha(archive) == item["sha256"], "Archive checksum differs")
        with tarfile.open(archive) as stream:
            for member in stream:
                path = worker / member.name
                require(path.resolve().is_relative_to(worker.resolve())
                        and (member.isfile() or member.isdir()), "Unsafe result archive path")
                if member.isfile():
                    require(hashlib.file_digest(stream.extractfile(member), "sha256").hexdigest()
                            == sha(path), "Extracted result differs from archived bytes")
        require(read(worker / "results/source_manifest.json") == manifest,
                "Worker source manifest differs")
        for name, expected in manifest.items():
            require(sha(worker / "code" / name) == expected, "Worker source bytes differ")
        require((worker / "results/exit_code.txt").read_text().strip() == "0",
                "Worker exited nonzero")
        log = (worker / "results/contracts.log").read_text()
        require("Ran 18 tests" in log and "\nOK" in log, "Worker CPU contracts missing")
        require(sha(worker / "code/protocol.json") == sha(root / "protocol.json"),
                "Frozen protocol differs")
        gold = records(worker / "code/posttrain_causal_lab/data/dev.jsonl")
        require(len(gold) == 64 and sha(worker / "code/posttrain_causal_lab/data/dev.jsonl")
                == protocol["dev_sha256"], "Dev data differs")
        inputs = root / f"input-worker-{item['worker']}"
        snapshot = read(worker / "results/snapshot-manifest.json")
        require(sha(worker / "results/snapshot-manifest.json")
                == protocol["input_archives"][item["worker"]]["snapshot_manifest_sha256"],
                "Adapter snapshot provenance differs")
        layout = [{"arm": arm, "seed": 20261027+item["worker"], "gpu": slot}
                  for slot, arm in enumerate(("opd32", "opd256"))]
        require(read(worker / "results/cells/layout.json") == layout, "Worker layout differs")
        require(read(worker / "results/cells/worker-status.json")
                == [{**spec, "returncode": 0} for spec in layout], "Cell failure")
        for spec in layout:
            name = f"{spec['arm']}-seed{spec['seed']}"
            cell = worker / "results/cells" / name
            contract, completed = read(cell / "contract.json"), read(cell / "completed.json")
            require(contract["arm"] == spec["arm"] and contract["seed"] == spec["seed"]
                    and contract["snapshot_spec"] == snapshot["cells"][name]
                    and contract["optimizer_updates"] == 0 and contract["eos_ids"] == [248044]
                    and contract["verification_accessed"] is False
                    and contract["benchmark_test_accessed"] is False, "Cell contract differs")
            require(completed["complete"] and completed["optimizer_updates"] == 0
                    and completed["original_step256_dev_reproduced"]
                    and completed["greedy_prefix_passed"] and completed["rng_probe_passed"],
                    "Cell gate did not pass")
            raw = records(cell / "generations.jsonl")
            require(len(raw) == 384, "Expected 3 checkpoints x 2 budgets x 64 records")
            indexed = {(r["step"], r["budget"], r["id"]): r for r in raw}
            expected = {(step, budget, row["id"]) for step in (64, 128, 256)
                        for budget in (256, 512) for row in gold}
            require(len(indexed) == len(raw) and set(indexed) == expected,
                    "Duplicate or missing generation records")
            metrics, moves = {}, {}
            for step in (64, 128, 256):
                values = {}
                metrics[str(step)] = {}
                for budget in (256, 512):
                    values[budget] = [indexed[step, budget, row["id"]] for row in gold]
                    actual = verify_generations(values[budget], gold, tokenizer, [248044], budget)
                    metrics[str(step)][str(budget)] = actual
                    check_summary(completed["metrics"][str(step)][str(budget)], actual, name)
                moves[str(step)] = transitions(values[256], values[512])
                if step == 256:
                    original_info = snapshot["cells"][name]["original_dev"]
                    original_path = inputs / original_info["path"]
                    require(sha(original_path) == original_info["sha256"], "Original dev reference differs")
                    original = records(original_path)
                    require([r["token_ids"] for r in original]
                            == [r["token_ids"] for r in values[256]],
                            "Original dev output not reproduced offline")
            rng = read(cell / "rng-probe.json")
            require(rng["passed"] and rng["outer_rng_restored"] and rng["shared_prefix_equal"]
                    and rng["derived_seed"] == sampling_seed(spec["seed"], rng["example_id"])
                    and rng["short_token_ids"] == rng["long_token_ids"][:len(rng["short_token_ids"])],
                    "Real-model case RNG probe differs")
            cells.append({"arm": spec["arm"], "seed": spec["seed"], "metrics": metrics,
                          "transitions": moves, "rng_probe": rng,
                          "elapsed_seconds": completed["elapsed_seconds"]})
    require(len(cells) == 6, "Six cells required")
    aggregate = {}
    for arm in ("opd32", "opd256"):
        selected = [c for c in cells if c["arm"] == arm]
        aggregate[arm] = {str(step): {str(budget): {
            key: statistics.mean(c["metrics"][str(step)][str(budget)][key] for c in selected)
            for key in ("complete_numeric_correct", "numeric_correct", "truncated", "output_tokens")}
            for budget in (256, 512)} for step in (64, 128, 256)}
    return {"audit_passed": True, "cells": cells, "aggregate": aggregate, "cost": gpu_cost(root),
            "generations": 2304, "unique_cases": 64, "split": "dev", "optimizer_updates": 0,
            "scope": "post-hoc diagnostic; no fresh quality confirmation or RNG retraining"}


def report(result):
    lines = ["# OPD checkpoint 长度敏感性诊断", "",
             "对已有checkpoint仅改变推理预算，六个cell通过独立复核；"
             "第256步的旧256-token输出逐token复现，256／512共同前缀一致。"
             "数据为原dev 64题，属于已观察原实验后的诊断。", "",
             "![checkpoint与预算比较](length-results.png)", "",
             "| Seed | 条件 | 步数 | 正确且完成／64：256→512 | 截断数：256→512 | 额外正确完成 |",
             "|---|---|---:|---:|---:|---:|"]
    for cell in sorted(result["cells"], key=lambda c: (c["seed"], c["arm"])):
        for step in (64, 128, 256):
            m = cell["metrics"][str(step)]
            short, long = m["256"], m["512"]
            lines.append(f"| {cell['seed']} | {cell['arm']} | {step} | "
                         f"{round(short['complete_numeric_correct']*64)}→"
                         f"{round(long['complete_numeric_correct']*64)} | "
                         f"{round(short['truncated']*64)}→{round(long['truncated']*64)} | "
                         f"{cell['transitions'][str(step)]['rescued_correct']} |")
    lines += ["", "每个点来自同一64道题；跨seed复用不增加独立题目数。预算增加后"
              "原已完成的回答必须保持不变，增加的正确完成只来自原先截断的题目。"
              "同checkpoint的推理预算效应可以直接比较；不同checkpoint之间的变化"
              "同时包含其训练历史，不能据此隔离某一步更新的因果贡献。", "",
              "真实CUDA逐题RNG探针全部通过：同题同模型8／16-token采样前缀一致，"
              "插入无关随机采样不影响结果，外层CPU／CUDA／Python状态恢复。"
              "这验证采样实现，没有重新训练，不能称为质量或稳定性收益。", "",
              f"本轮优化更新0，实际分配成本 "
              f"{result['cost']['gpu_hours'] if result['cost']['complete'] else '未完整确定'} GPU·h。"
              "配置见 [PROTOCOL.md](PROTOCOL.md)，逐组记录见 [analysis.json](analysis.json)。", ""]
    return "\n".join(lines)


def render(result, root):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4), layout="constrained")
    for ax, arm in zip(axes, ("opd32", "opd256")):
        for budget, color in (("256", "#ca7850"), ("512", "#428878")):
            selected = [c for c in result["cells"] if c["arm"] == arm]
            for cell in selected:
                ys = [100*cell["metrics"][str(step)][budget]["complete_numeric_correct"]
                      for step in (64, 128, 256)]
                ax.plot([64, 128, 256], ys, color=color, alpha=.25, lw=1, marker=".")
            mean = [100*result["aggregate"][arm][str(step)][budget]["complete_numeric_correct"]
                    for step in (64, 128, 256)]
            ax.plot([64, 128, 256], mean, color=color, lw=2, marker="o", label=budget+" tokens")
        ax.set_title(arm.upper())
        ax.set_xlabel("Training checkpoint (updates)")
        ax.set_xticks([64, 128, 256])
        ax.set_ylim(0, 100)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.2)
        ax.legend(frameon=False)
    axes[0].set_ylabel("Dev complete numeric accuracy (%)")
    fig.suptitle("Same 64 dev cases · 3 paired seeds · thin lines: seeds; thick lines: means", fontsize=10)
    for suffix in ("png", "svg"):
        fig.savefig(root / ("length-results." + suffix), dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True, type=Path)
    args = parser.parse_args()
    root = args.experiment.resolve()
    try:
        result = analyze(root)
    except Exception as error:
        write_json(root / "audit.json", {"audit_passed": False, "error": str(error)})
        raise
    write_json(root / "analysis.json", result)
    write_json(root / "audit.json", {"audit_passed": True, "cells": 6, "generations": 2304})
    render(result, root)
    (root / "REPORT.md").write_text(report(result))
    print(json.dumps({"audit_passed": True, "aggregate": result["aggregate"], "cost": result["cost"]}))


if __name__ == "__main__":
    main()
