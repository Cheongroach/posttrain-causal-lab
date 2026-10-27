# Frozen data

`manifest.json` records the canonical source, source SHA256, split seed, row IDs and output hashes for the 256／64／64 GSM8K-train-derived split.

The JSONL rows are not duplicated in this repository. Rebuild them with `ops/freeze_assets.py` from the pinned parquet source.

A0 only consumed `train.jsonl`; it did not access dev, intervention verification or the official GSM8K test split.
