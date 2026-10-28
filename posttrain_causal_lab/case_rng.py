"""Stable per-example sampling streams, separate from initialization and order."""
import contextlib
import hashlib
import json
import random

import torch


def sampling_seed(rollout_seed, example_id, epoch=0, sample_index=0):
    if (type(rollout_seed) is not int or type(epoch) is not int or epoch < 0
            or type(sample_index) is not int or sample_index < 0
            or not isinstance(example_id, str) or not example_id):
        raise ValueError("Expected an integer seed, nonempty ID, and nonnegative indices")
    # Arm, output budget, and sample processing order deliberately do not enter.
    payload = json.dumps(["posttrain-case-rng-v1", rollout_seed, example_id, epoch, sample_index],
                         ensure_ascii=False, separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


@contextlib.contextmanager
def case_sampling(rollout_seed, example_id, epoch=0, sample_index=0, cuda_devices=()):
    seed = sampling_seed(rollout_seed, example_id, epoch, sample_index)
    devices = list(cuda_devices)
    if len(set(devices)) != len(devices) or any(type(i) is not int or i < 0 for i in devices):
        raise ValueError("CUDA devices must be distinct nonnegative indices")
    python_state = random.getstate()
    # fork_rng restores CPU and listed CUDA streams even if generation raises.
    with torch.random.fork_rng(devices=devices):
        torch.random.default_generator.manual_seed(seed)
        for device in devices:
            torch.cuda.default_generators[device].manual_seed(seed)
        random.seed(seed)
        try:
            yield seed
        finally:
            random.setstate(python_state)
