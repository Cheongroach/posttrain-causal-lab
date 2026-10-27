"""Frozen experimental layout, numeric grading, and matched interventions."""
from __future__ import annotations

import decimal
import re

import torch


ARMS = ("sft", "kd", "opd32", "opd256")
SEEDS = (20261027, 20261028, 20261029)


def cell_layout(worker_index: int) -> list[dict]:
    if worker_index not in range(3):
        raise ValueError("This experiment has exactly three workers")
    return [{"seed": SEEDS[worker_index], "arm": arm, "gpus": [2 * slot, 2 * slot + 1]}
            for slot, arm in enumerate(ARMS)]


def grade(text: str, answer: str, truncated: bool) -> dict:
    """Report format and a final-number proxy separately, using no eval()."""
    marked = re.findall(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)", text)
    numbers = re.findall(r"(?<!\w)[-+]?\d[\d,]*(?:\.\d+)?", text)
    extracted = marked[-1] if marked else (numbers[-1] if numbers else None)
    correct = False
    if extracted is not None:
        try:
            correct = decimal.Decimal(extracted.replace(",", "")) == decimal.Decimal(
                answer.replace(",", "").strip())
        except decimal.InvalidOperation:
            correct = False
    return {
        "extracted_number": extracted, "format_valid": bool(marked),
        "numeric_correct": bool(correct),
        "complete_numeric_correct": bool(correct and not truncated),
        "strict_correct": bool(correct and marked and not truncated),
        "truncated": bool(truncated),
    }


def matched_random(original: torch.Tensor, donor: torch.Tensor, seed: int) -> torch.Tensor:
    """Random direction with exactly the donor intervention's L2 norm."""
    if original.shape != donor.shape:
        raise ValueError("Donor shape differs from the recipient")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(original.shape, generator=generator, dtype=torch.float32)
    magnitude = (donor.float() - original.float()).norm()
    if float(magnitude) == 0:
        return original.clone()
    return original + noise.to(original.device, original.dtype) * (
        magnitude.to(original.device) / noise.norm().to(original.device))


def choose_sites(records: list[dict], count: int = 3) -> list[str]:
    """Select on dev only: largest loss of gold log probability under base patch."""
    grouped = {}
    for record in records:
        if record["mode"] == "base":
            grouped.setdefault(record["site"], []).append(record["gold_logp_drop"])
    scores = {site: sum(values) / len(values) for site, values in grouped.items()}
    return sorted(scores, key=lambda site: (-scores[site], site))[:count]
