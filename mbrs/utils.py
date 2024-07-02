from typing import Any

import torch


def to_device(sample: Any, device: torch.device):
    def _to_device(x):
        if torch.is_tensor(x):
            return x.to(device=device, non_blocking=True)
        elif isinstance(x, dict):
            return {key: _to_device(value) for key, value in x.items()}
        elif isinstance(x, list):
            return [_to_device(x) for x in x]
        elif isinstance(x, tuple):
            return tuple(_to_device(x) for x in x)
        elif isinstance(x, set):
            return {_to_device(x) for x in x}
        else:
            return x

    return _to_device(sample)
