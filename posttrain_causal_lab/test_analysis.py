"""Synthetic audit failures and a complete matrix; fixtures live in temp dirs."""
import copy
import io
import json
import random
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from . import analyze_scale24 as analysis
from .analyze_scale24 import write_json as write
from .scale_contracts import cell_layout, grade


class FixtureTokenizer:
    def __init__(self):
        self.texts = [""]

    def encode(self, text):
        if text not in self.texts:
            self.texts.append(text)
        return [self.texts.index(text), 0]

    def __len__(self):
        return len(self.texts)

    def decode(self, tokens, skip_special_tokens=True):
        return "".join(self.texts[token] for token in tokens if token)


def make_generations(gold, tokenizer, correct_count):
    rows = []
    for index, row in enumerate(gold):
        answer = row["final_answer"] if index < correct_count else "987654321987654321"
        text = "#### " + answer
        tokens = tokenizer.encode(text)
        rows.append({"id": row["id"], "text": text, "token_ids": tokens, "output_tokens": 2,
                     "elapsed_seconds": .01, **grade(text, row["final_answer"], False)})
    return rows


def write_records(path, values):
    path.write_text("".join(json.dumps(value) + "\n" for value in values))


def make_interventions(ids, names):
    result = []
    for identifier in ids:
        for index, site in enumerate(names):
            for mode in ("base", "random") + (("identity",) if index == 0 else ()):
                drop = .1 + .001*index if mode == "base" else (.01 if mode == "random" else 0.)
                result.append({"id": identifier, "site": site, "mode": mode,
                               "base_gold_logp": -3., "trained_gold_logp": -2.,
                               "patched_gold_logp": -2. - drop, "gold_logp_drop": drop,
                               "logits_max_abs": 0. if mode == "identity" else .1,
                               "intervention_l2": 0. if mode == "identity" else .5})
    return result


def make_synthetic_package(root):
    names = ["pilot.py", "contracts.py", "assets.py", "scale_contracts.py", "scale24.py",
             "scale24_worker.py", "test_contracts.py", "test_model_contract.py", "test_scale24.py",
             "assets/models.json", "assets/wheels.json", "data/manifest.json",
             "data/train.jsonl", "data/dev.jsonl", "data/verification.jsonl"]
    blobs = {"posttrain_causal_lab/" + name: b"# synthetic audit fixture\n" for name in names}
    for split, count in (("train", 256), ("dev", 64), ("verification", 64)):
        values = [{"id": f"synthetic-{split}-{i}", "question": f"{i}+1?",
                   "final_answer": str(i + 1)} for i in range(count)]
        blobs["posttrain_causal_lab/data/" + split + ".jsonl"] = "".join(
            json.dumps(value) + "\n" for value in values).encode()
    blobs["protocol.json"] = b'{"synthetic": true}\n'
    blobs["posttrain_causal_lab/assets/models.json"] = b'{"models": []}\n'
    (root / "protocol.json").write_bytes(blobs["protocol.json"])
    write(root / "source_manifest.json", {
        name: analysis.hashlib.sha256(blob).hexdigest() for name, blob in blobs.items()})
    with tarfile.open(root / "code.tar.gz", "w:gz") as archive:
        for name, blob in sorted(blobs.items()):
            member = tarfile.TarInfo(name)
            member.size = len(blob)
            archive.addfile(member, io.BytesIO(blob))


def make_matrix(root, tokenizer):
    """All input and output records are synthetic; no GPU or dataset access."""
    manifest, blobs = analysis.frozen_package(root)
    data = {split: [json.loads(line) for line in blobs[
        "posttrain_causal_lab/data/" + split + ".jsonl"].decode().splitlines()]
        for split in ("train", "dev", "verification")}
    trial = "1"
    write(root / "run.json", {"data": {"status": "done", "runtime_context": {"trial_id": trial}}})
    archives = []
    names = [f"layer{i:02d}.{kind}" for i in range(32) for kind in ("gdn", "ffn")]
    attachments = [f"base_model.model.model.layers.{i}.{kind}" for i in range(32)
                   for kind in ("linear_attn.out_proj", "mlp.down_proj")]
    for worker_index in range(3):
        worker = root / "artifacts" / "trial-1" / f"worker-{worker_index}"
        for name, blob in blobs.items():
            destination = worker / "code" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(blob)
        results = worker / "results"
        results.mkdir()
        write(results / "source_manifest.json", manifest)
        (results / "exit_code.txt").write_text("0\n")
        (results / "contracts.log").write_text("Ran 13 tests\n\nOK\n")
        cells = results / "cells"
        cells.mkdir()
        layout = cell_layout(worker_index)
        write(cells / "layout.json", layout)
        write(cells / "worker-status.json", [{**s, "returncode": 0} for s in layout])
        for spec in layout:
            cell = cells / f"{spec['arm']}-seed{spec['seed']}"
            cell.mkdir()
            train = list(data["train"])
            random.Random(spec["seed"]).shuffle(train)
            order = [r["id"] for r in train]
            adapter = {site + ".lora_A.default.weight": torch.zeros(2, 2) for site in attachments}
            provenance = {"seed": spec["seed"], "arm": spec["arm"],
                          "protocol_sha256": manifest["protocol.json"],
                          "data_manifest_sha256": manifest["posttrain_causal_lab/data/manifest.json"],
                          "models_manifest_sha256": manifest["posttrain_causal_lab/assets/models.json"],
                          "train_order_sha256": analysis.hashlib.sha256(json.dumps(order).encode()).hexdigest(),
                          "lora_sites": attachments,
                          "rollout_budget": 32 if spec["arm"] == "opd32" else 256,
                          "learning_rate": 1e-4, "student_dtype": "float32", "teacher_dtype": "bfloat16"}
            contract = {**provenance, "versions": {"torch": "synthetic-fixture"},
                        "split_counts": {"train": 256, "dev": 64, "verification": 64},
                        "train_order": order, "benchmark_test_accessed": False,
                        "evaluation_budget": 256, "student_parameters": 256,
                        "trainable_parameters": 256, "initial_adapter_sha256": analysis.adapter_digest(adapter),
                        "gpu_names": ["synthetic H20", "synthetic H20"]}
            write(cell / "contract.json", contract)
            for step, name in ((0, "initial"), (64, "step64"), (128, "step128"), (256, "step256")):
                parameters = {k: t + step * .001 for k, t in adapter.items()}
                state = {"adapter": parameters, "cursor": step, "provenance": provenance,
                         "torch_rng": torch.zeros(8, dtype=torch.uint8),
                         "cuda_rng": [torch.zeros(8, dtype=torch.uint8)]*2,
                         "python_rng": random.Random(spec["seed"]).getstate(),
                         "optimizer": {"param_groups": [{"lr": 1e-4, "weight_decay": 0.}],
                                       "state": {i: {"step": torch.tensor(float(step)),
                                                     "exp_avg": torch.zeros(2, 2)}
                                                 for i in range(64)} if step else {}}}
                path = cell / (name + ".pt")
                torch.save(state, path)
                write(path.with_suffix(".sha256.json"), {"sha256": analysis.sha(path),
                                                        "bytes": path.stat().st_size})
            exported = {k.replace(".default.", "."): t + .256 for k, t in adapter.items()}
            (cell / "adapter").mkdir()
            save_file(exported, str(cell / "adapter/adapter_model.safetensors"))
            write(cell / "adapter/adapter_config.json", {
                "r": 16, "lora_alpha": 32, "lora_dropout": 0.,
                "target_modules": ["o_proj", "out_proj", "down_proj"]})
            events = [{"event": "train_step", "step": i + 1, "id": identifier,
                       "arm": spec["arm"], "loss": .2, "grad_norm": .3,
                       "elapsed_seconds": .01, "response_tokens": 2, "prefix_tokens": 20}
                      for i, identifier in enumerate(order)]
            events.append({"event": "cell_complete"})
            write_records(cell / "events.jsonl", events)
            write(cell / "recovery.json", [{
                "step": 128, "adapter_max_abs": 0., "reference": {"loss": .2},
                "replay": {"loss": .2}, "scope": "same-process fixed-prefix replay"}])
            metrics = {}
            for phase in ("base_dev", "trained_dev", "trained_verification", "base_verification"):
                split = "verification" if "verification" in phase else "dev"
                count = 8 if phase.startswith("trained") and spec["arm"] == "opd256" else 0
                generations = make_generations(data[split], tokenizer, count)
                write_records(cell / (phase.replace("_", "-") + ".jsonl"), generations)
                metrics[phase] = analysis.summary(generations)
            for phase in ("base_dev_prefix", "trained_dev_prefix", "trained_verification_prefix"):
                split = "verification" if "verification" in phase else "dev"
                values = [{"id": r["id"], "tokens": 2, "gold_nll": .2, "teacher_kl": .3}
                          for r in data[split]]
                write_records(cell / (phase.replace("_", "-") + ".jsonl"), values)
                metrics[phase] = {"n": 64, "tokens": 128, "gold_nll": .2, "teacher_kl": .3}
            if spec["arm"].startswith("opd"):
                rollout = make_generations(train, tokenizer, 0)
                write_records(cell / "rollouts.jsonl", [{"step": i, **r} for i, r in enumerate(rollout)])
            dev = make_interventions([r["id"] for r in data["dev"][:8]], names)
            write_records(cell / "dev-interventions.jsonl", dev)
            selected = analysis.choose_sites(dev)
            write(cell / "selected-sites.json", {"selection_split": "dev", "n_search": 8,
                                                 "sites": selected})
            write_records(cell / "verification-interventions.jsonl",
                          make_interventions([r["id"] for r in data["verification"]], selected))
            write(cell / "metrics.json", metrics)
            write(cell / "completed.json", {"complete": True, "arm": spec["arm"], "seed": spec["seed"],
                                            "steps": 256, "metrics": metrics, "selected_sites": selected,
                                            "recovery_passed": True, "elapsed_seconds": 100.,
                                            "peak_allocated_bytes": [1000, 1000]})
        archive = root / f"trial-1-trialrun-1-worker-{worker_index}.tar.gz"
        with tarfile.open(archive, "w:gz") as stream:
            stream.add(worker / "code", arcname="code")
            stream.add(worker / "results", arcname="results")
        digest = analysis.sha(archive)
        write(worker / "extracted.json", {"sha256": digest, "trial": "1"})
        archives.append({"worker": worker_index, "trial": trial, "complete": True,
                         "directory": str(worker.relative_to(root)), "sha256": digest,
                         "bytes": archive.stat().st_size})
    write(root / "artifacts.json", archives)


class AnalysisContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="posttrain-audit-synthetic-")
        cls.root = Path(cls.temporary.name)
        cls.tokenizer = FixtureTokenizer()
        make_synthetic_package(cls.root)
        cls.package_pin = patch.object(analysis, "PACKAGE_SHA256", analysis.sha(cls.root / "code.tar.gz"))
        cls.package_pin.start()
        make_matrix(cls.root, cls.tokenizer)

    @classmethod
    def tearDownClass(cls):
        cls.package_pin.stop()
        cls.temporary.cleanup()

    def test_complete_matrix_reproduces_known_paired_effect_and_counts(self):
        with patch.object(analysis, "tokenizer_for_audit", return_value=(self.tokenizer, [0])):
            result = analysis.analyze(self.root)
        self.assertTrue(result["audit_passed"])
        self.assertEqual(len(result["cells"]), 12)
        effect = result["paired_effects"]["complete_numeric_correct"]
        self.assertEqual(effect["values"], [.125]*3)
        self.assertEqual(effect["mean"], .125)
        self.assertEqual(effect["unique_verification_cases"], 64)
        self.assertEqual(len(result["primary_case_pairs"]), 64)
        self.assertEqual(result["cells"][0]["dev_interventions"]["n_records"], 1032)
        self.assertEqual(result["cells"][0]["verification_interventions"]["n_records"], 448)
        self.assertFalse(result["cost"]["complete"])
        self.assertIn("12.50", analysis.report(result))

    def test_missing_worker_never_becomes_a_partial_success(self):
        path = self.root / "artifacts.json"
        original = path.read_text()
        try:
            write(path, analysis.read(path)[:2])
            with patch.object(analysis, "tokenizer_for_audit", return_value=(self.tokenizer, [0])):
                with self.assertRaisesRegex(analysis.AuditError, "three worker"):
                    analysis.analyze(self.root)
        finally:
            path.write_text(original)

    def test_scoring_and_token_decoding_are_independently_checked(self):
        gold = [{"id": "x", "final_answer": "3"}]
        values = make_generations(gold, self.tokenizer, 1)
        analysis.verify_generations(values, gold, self.tokenizer, [0], 256)
        values[0]["numeric_correct"] = False
        with self.assertRaisesRegex(analysis.AuditError, "grading"):
            analysis.verify_generations(values, gold, self.tokenizer, [0], 256)
        values = make_generations(gold, self.tokenizer, 1)
        values[0]["text"] = "#### 4"
        with self.assertRaisesRegex(analysis.AuditError, "decoding"):
            analysis.verify_generations(values, gold, self.tokenizer, [0], 256)

    def test_vocabulary_size_is_cached_for_raw_token_validation(self):
        class CountingTokenizer(FixtureTokenizer):
            calls = 0

            def __len__(self):
                self.calls += 1
                return super().__len__()

        tokenizer = CountingTokenizer()
        gold = [{"id": "x", "final_answer": "3"}, {"id": "y", "final_answer": "4"}]
        values = make_generations(gold, tokenizer, 2)
        analysis.verify_generations(values, gold, tokenizer, [0], 256)
        self.assertEqual(tokenizer.calls, 1)
        values[0]["token_ids"][0] = 999
        with self.assertRaisesRegex(analysis.AuditError, "generation tokens"):
            analysis.verify_generations(values, gold, tokenizer, [0], 256)

    def test_interventions_reject_duplicate_rows_and_unmatched_treatment(self):
        values = make_interventions(["x"], ["layer00.gdn"])
        with self.assertRaisesRegex(analysis.AuditError, "duplicate"):
            analysis.verify_interventions(values + [values[0]], ["x"], ["layer00.gdn"])
        altered = copy.deepcopy(values)
        altered[1]["intervention_l2"] = .6
        with self.assertRaisesRegex(analysis.AuditError, "L2 differs"):
            analysis.verify_interventions(altered, ["x"], ["layer00.gdn"])

    def test_checksum_valid_nonfinite_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="posttrain-bad-checkpoint-") as directory:
            cell = Path(directory)
            path = cell / "initial.pt"
            torch.save({"cursor": 0, "provenance": {}, "adapter": {"w": torch.tensor(float("nan"))}}, path)
            write(path.with_suffix(".sha256.json"), {"sha256": analysis.sha(path),
                                                    "bytes": path.stat().st_size})
            with self.assertRaisesRegex(analysis.AuditError, "Nonfinite checkpoint"):
                analysis.verify_checkpoints(cell, {}, {})

    def test_cost_includes_stopped_trial_and_requires_closed_pod_interval(self):
        with tempfile.TemporaryDirectory(prefix="posttrain-cost-evidence-") as directory:
            root = Path(directory)
            write(root / "trials.json", {"data": [{"sid": "old"}, {"sid": "new"}]})
            for sid in ("old", "new"):
                path = root / "trial-evidence" / sid
                path.mkdir(parents=True)
                write(path / "trial.json", {"data": {"started_at": None, "status": 7}})
                write(path / "pods.json", {"data": []})
            new = root / "trial-evidence/new/pods.json"
            write(new, {"data": [{"sid": "p", "schedule_at": "2026-10-27T00:00:00Z",
                                 "stop_at": "2026-10-27T01:00:00Z"}]})
            value = analysis.cost_accounting(root)
            self.assertTrue(value["complete"])
            self.assertEqual(value["gpu_hours"], 8.)
            self.assertEqual(value["trials"][0]["gpu_hours"], 0.)
            write(new, {"data": [{"sid": "p", "schedule_at": "2026-10-27T00:00:00Z"}]})
            self.assertFalse(analysis.cost_accounting(root)["complete"])
            self.assertIsNone(analysis.cost_accounting(root)["gpu_hours"])


if __name__ == "__main__":
    unittest.main()
