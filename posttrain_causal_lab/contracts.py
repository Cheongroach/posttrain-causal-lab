"""Response-token losses and interventions shared by CPU checks and the pilot."""
from __future__ import annotations

import contextlib

import torch
import torch.nn.functional as F


def response_mask(tokens: torch.Tensor, eos_ids: list[int]) -> torch.Tensor:
    """Include the first EOS action, excluding subsequent padding."""
    eos = torch.zeros_like(tokens, dtype=torch.bool)
    for token_id in eos_ids:
        eos |= tokens.eq(token_id)
    return (eos.long().cumsum(-1) - eos.long()).eq(0)


def token_loss(student: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor,
               teacher: torch.Tensor | None = None) -> torch.Tensor:
    """Token mean CE or full-vocabulary KL(teacher || student), temperature 1."""
    if student.shape[:-1] != targets.shape or targets.shape != mask.shape:
        raise ValueError("Logits, targets and response mask do not align")
    if not bool(mask.any()):
        raise ValueError("Empty response cannot define a training loss")
    selected = student[mask].float().log_softmax(-1)
    if teacher is None:
        return F.nll_loss(selected, targets[mask])
    if teacher.shape != student.shape:
        raise ValueError("Teacher and student event spaces differ")
    reference = teacher.detach()[mask].float().log_softmax(-1)
    return (reference.exp() * (reference - selected)).sum(-1).mean()


@contextlib.contextmanager
def output_intervention(module: torch.nn.Module, mode: str,
                        replacement: torch.Tensor | None = None):
    """Patch a module output without leaving hooks installed after an error."""
    if mode not in ("capture", "identity", "zero", "replace"):
        raise ValueError(mode)
    captured = []

    def hook(_module, _inputs, output):
        if not isinstance(output, torch.Tensor):
            raise TypeError("Intervention requires a tensor output")
        if mode == "capture":
            captured.append(output.detach().clone())
            return output
        if mode == "identity":
            return output.clone()
        if mode == "zero":
            return torch.zeros_like(output)
        if replacement is None or replacement.shape != output.shape:
            raise ValueError("Donor and recipient activation shapes differ")
        return replacement.to(output.device, output.dtype)

    handle = module.register_forward_hook(hook)
    try:
        yield captured
    finally:
        handle.remove()
