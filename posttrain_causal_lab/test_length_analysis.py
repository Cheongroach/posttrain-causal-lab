"""Budget rescue accounting must preserve the paired greedy prefix."""
import json
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from .analyze_length_probe import analyze, transitions
from .analyze_scale24 import sha, summary, write_json
from .case_rng import sampling_seed
from .scale_contracts import grade


class Tokens:
    def __len__(self):
        return 3

    def decode(self, tokens, skip_special_tokens=True):
        return "".join({0: "", 1: ".", 2: "#### 1"}.get(t, "") for t in tokens)


def fixture(root):
    gold = [{"id": f"dev-{i}", "final_answer": "1"} for i in range(64)]
    gold_blob = "".join(json.dumps(row) + "\n" for row in gold)
    import hashlib
    protocol = {"dev_sha256": hashlib.sha256(gold_blob.encode()).hexdigest(),
                "input_archives": []}
    artifacts = []
    for index in range(3):
        worker = root / f"worker-{index}"
        cells_dir = worker / "results/cells"
        cells_dir.mkdir(parents=True)
        layout = [{"arm": arm, "seed": 20261027+index, "gpu": gpu}
                  for gpu, arm in enumerate(("opd32", "opd256"))]
        write_json(cells_dir / "layout.json", layout)
        write_json(cells_dir / "worker-status.json", [{**s, "returncode": 0} for s in layout])
        (worker / "results/exit_code.txt").write_text("0")
        (worker / "results/contracts.log").write_text("Ran 18 tests\n\nOK\n")
        input_dir = root / f"input-worker-{index}"
        input_dir.mkdir()
        snapshots = {"cells": {}}
        for spec in layout:
            name = f"{spec['arm']}-seed{spec['seed']}"
            cell = cells_dir / name
            cell.mkdir()
            raw, metrics, original = [], {}, []
            for step, correct in ((64, 32), (128, 40), (256, 48)):
                metrics[str(step)] = {}
                for budget in (256, 512):
                    rows = []
                    for i, row in enumerate(gold):
                        if i < correct:
                            tokens = [2, 248044]
                        elif budget == 512 and i < correct+8:
                            tokens = [1]*256 + [2, 248044]
                        else:
                            tokens = [1]*budget
                        text = Tokens().decode(tokens)
                        value = {"step": step, "budget": budget, "id": row["id"],
                                 "text": text, "token_ids": tokens, "output_tokens": len(tokens),
                                 "elapsed_seconds": .1, **grade(text, "1", tokens[-1] != 248044)}
                        rows.append(value)
                    metrics[str(step)][str(budget)] = summary(rows)
                    raw.extend(rows)
                    if step == 256 and budget == 256:
                        original = rows
            original_file = input_dir / (name + ".jsonl")
            original_file.write_text("".join(json.dumps(r)+"\n" for r in original))
            snapshot = {"original_dev": {"path": original_file.name, "sha256": sha(original_file)}}
            snapshots["cells"][name] = snapshot
            write_json(cell / "contract.json", {
                "arm": spec["arm"], "seed": spec["seed"], "snapshot_spec": snapshot,
                "optimizer_updates": 0, "eos_ids": [248044],
                "verification_accessed": False, "benchmark_test_accessed": False})
            write_json(cell / "completed.json", {
                "complete": True, "optimizer_updates": 0, "original_step256_dev_reproduced": True,
                "greedy_prefix_passed": True, "rng_probe_passed": True,
                "metrics": metrics, "elapsed_seconds": 1.})
            (cell / "generations.jsonl").write_text("".join(json.dumps(r)+"\n" for r in raw))
            write_json(cell / "rng-probe.json", {
                "passed": True, "outer_rng_restored": True, "shared_prefix_equal": True,
                "example_id": gold[0]["id"], "derived_seed": sampling_seed(spec["seed"], gold[0]["id"]),
                "short_token_ids": [1]*8, "long_token_ids": [1]*16})
        write_json(worker / "results/snapshot-manifest.json", snapshots)
        protocol["input_archives"].append({
            "snapshot_manifest_sha256": sha(worker / "results/snapshot-manifest.json")})
    write_json(root / "protocol.json", protocol)
    manifest = {"protocol.json": sha(root / "protocol.json"),
                "posttrain_causal_lab/data/dev.jsonl": protocol["dev_sha256"]}
    write_json(root / "source_manifest.json", manifest)
    for index in range(3):
        worker = root / f"worker-{index}"
        code = worker / "code"
        (code / "posttrain_causal_lab/data").mkdir(parents=True)
        (code / "protocol.json").write_bytes((root / "protocol.json").read_bytes())
        (code / "posttrain_causal_lab/data/dev.jsonl").write_text(gold_blob)
        write_json(worker / "results/source_manifest.json", manifest)
        archive = root / f"trial-1-trialrun-1-worker-{index}.tar.gz"
        with tarfile.open(archive, "w:gz") as stream:
            stream.add(code, arcname="code")
            stream.add(worker / "results", arcname="results")
        artifacts.append({"worker": index, "trial": "1", "complete": True,
                          "directory": worker.name, "sha256": sha(archive)})
    write_json(root / "artifacts.json", artifacts)
    write_json(root / "run.json", {"data": {"status": "done", "runtime_context": {"trial_id": "1"}}})
    write_json(root / "trials.json", {"data": [{"sid": "1"}]})
    evidence = root / "trial-evidence/1"
    evidence.mkdir(parents=True)
    write_json(evidence / "trial.json", {"data": {"status": 4}})
    write_json(evidence / "pods.json", {"data": [
        {"sid": str(i), "schedule_at": "2026-10-28T00:00:00Z", "stop_at": "2026-10-28T01:00:00Z"}
        for i in range(3)]})


class LengthAnalysis(unittest.TestCase):
    def test_complete_archives_reproduce_eight_rescues_per_cell_and_checkpoint(self):
        class ValidTokens(Tokens):
            def __len__(self):
                return 248077
        with tempfile.TemporaryDirectory(prefix="posttrain-length-fixture-") as directory:
            root = Path(directory)
            fixture(root)
            with patch("transformers.AutoTokenizer.from_pretrained", return_value=ValidTokens()):
                result = analyze(root)
            self.assertTrue(result["audit_passed"])
            self.assertEqual(len(result["cells"]), 6)
            self.assertEqual(result["cost"]["gpu_hours"], 6.)
            self.assertTrue(all(c["transitions"][str(step)]["rescued_correct"] == 8
                                for c in result["cells"] for step in (64, 128, 256)))

    def test_rescue_wrong_and_still_truncated_are_distinct(self):
        short = [{"id": str(i), "token_ids": [1, 2], "truncated": True,
                  "complete_numeric_correct": False} for i in range(3)]
        long = [{"id": str(i), "token_ids": [1, 2, 3], "truncated": i == 2,
                 "complete_numeric_correct": i == 0} for i in range(3)]
        result = transitions(short, long)
        self.assertEqual(result["rescued_correct"], 1)
        self.assertEqual(result["finished_wrong"], 1)
        self.assertEqual(result["still_truncated"], 1)

    def test_completed_answers_cannot_be_changed_by_larger_budget(self):
        short = [{"id": "x", "token_ids": [1, 2], "truncated": False,
                  "complete_numeric_correct": True}]
        self.assertEqual(transitions(short, short)["already_finished"], 1)
        long = [{**short[0], "token_ids": [1, 2, 3]}]
        with self.assertRaisesRegex(ValueError, "already completed"):
            transitions(short, long)

    def test_answer_count_cannot_hide_prefix_mismatch(self):
        short = [{"id": "x", "token_ids": [1, 2], "truncated": True,
                  "complete_numeric_correct": False}]
        long = [{"id": "x", "token_ids": [1, 9, 3], "truncated": False,
                 "complete_numeric_correct": True}]
        with self.assertRaisesRegex(ValueError, "shared greedy prefix"):
            transitions(short, long)


if __name__ == "__main__":
    unittest.main()
