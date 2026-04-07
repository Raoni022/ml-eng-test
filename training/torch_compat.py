"""
torch_compat.py

Compatibility helper for PyTorch 2.6+ where `torch.load()` changed the default
`weights_only` behavior to `True`.

Some Ultralytics workflows still assume the older default when loading trusted
YOLO checkpoints. This helper patches `torch.load` only when `weights_only`
was not explicitly provided by the caller.

Use this only for trusted weights such as:
- official Ultralytics pretrained checkpoints
- your own locally trained checkpoints
"""

from __future__ import annotations

from functools import wraps


def patch_torch_load_for_trusted_weights() -> None:
    import torch

    if getattr(torch.load, "_truebuilt_patched", False):
        return

    original_torch_load = torch.load

    @wraps(original_torch_load)
    def patched_torch_load(*args, **kwargs):
        if "weights_only" not in kwargs:
            kwargs["weights_only"] = False
        return original_torch_load(*args, **kwargs)

    patched_torch_load._truebuilt_patched = True  # type: ignore[attr-defined]
    torch.load = patched_torch_load
