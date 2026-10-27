"""Freeze independent GSM8K train subsets, published model pins and wheels."""
import argparse
import hashlib
import json
import random
from pathlib import Path


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-parquet", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--wheels", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    expected = "ea82612ea9582142387730c793eb67d3b12849002bc0b7fa6f8efafa7351419d"
    if sha(args.train_parquet) != expected:
        raise RuntimeError("Official GSM8K train mirror hash differs")
    import pyarrow.parquet as pq
    records = pq.read_table(args.train_parquet).to_pylist()
    if len(records) != 7473:
        raise ValueError("Unexpected GSM8K training row count")
    rows, seen = [], set()
    for index, row in enumerate(records):
        question = row["question"].strip()
        if question in seen:
            continue
        seen.add(question)
        if "#### " not in row["answer"]:
            raise ValueError("Missing official final answer")
        rows.append({
            "id": f"gsm8k-train-{index:05d}", "question": question,
            "answer": row["answer"],
            "final_answer": row["answer"].rsplit("#### ", 1)[1].strip(),
            "question_sha256": hashlib.sha256(question.encode()).hexdigest()})
    random.Random(20261027).shuffle(rows)
    data = root / "data"
    data.mkdir(exist_ok=True)
    files, offset = {}, 0
    for name, count in (("train", 256), ("dev", 64), ("verification", 64)):
        subset = rows[offset:offset + count]
        offset += count
        path = data / (name + ".jsonl")
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in subset))
        files[path.name] = {"sha256": sha(path), "rows": count,
                            "ids": [r["id"] for r in subset]}
    manifest = {
        "canonical_dataset": "openai/gsm8k, main, official training split",
        "mirror": "opencompass/gsm8k on ModelScope",
        "mirror_revision": "0aaec570272bcba74732c98af7686519774deaaa",
        "source_url": "https://modelscope.cn/datasets/opencompass/gsm8k/resolve/"
                      "0aaec570272bcba74732c98af7686519774deaaa/main/train-00000-of-00001.parquet",
        "source_sha256": expected, "source_rows": len(records),
        "unique_questions": len(rows), "seed": 20261027,
        "split_rule": "deduplicate exact question, seed shuffle, disjoint 256/64/64 from train",
        "benchmark_test_accessed": False, "pilot_access": ["train.jsonl"], "files": files}
    (data / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    model_source = json.loads(args.model_manifest.read_text())
    models = [m for m in model_source["models"] if m["model"] in
              ("Qwen/Qwen3.5-4B", "Qwen/Qwen3.5-9B")]
    if len(models) != 2:
        raise ValueError("Need exact published student/teacher manifests")
    assets = root / "assets"
    assets.mkdir(exist_ok=True)
    (assets / "models.json").write_text(json.dumps({"models": models}, indent=2) + "\n")
    wheels = {p.name: {"sha256": sha(p), "bytes": p.stat().st_size}
              for p in sorted(args.wheels.glob("*.whl"))}
    (assets / "wheels.json").write_text(json.dumps(wheels, indent=2) + "\n")
    print(json.dumps({"dataset_rows": 384, "models": 2, "wheels": len(wheels)}))


if __name__ == "__main__":
    main()
