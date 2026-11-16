"""Hardware and dependency environment information for reproducible runs."""

import importlib.metadata
import platform
import time
from collections.abc import Callable
from typing import Any, TypeVar

T = TypeVar("T")


def package_versions() -> dict[str, str]:
    """Capture versions without importing heavy dependencies."""
    result = {"python": platform.python_version(), "platform": platform.platform()}
    for name in (
        "torch",
        "sentence-transformers",
        "transformers",
        "datasets",
        "numpy",
        "pydantic",
    ):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "not installed"
    return result


def time_call(
    function: Callable[[], T], *, device: str = "cpu"
) -> tuple[T, dict[str, Any]]:
    """Time inference, synchronizing CUDA to avoid timing only launches."""
    if device.startswith("cuda"):
        import torch

        torch.cuda.synchronize(device)
    start = time.perf_counter()
    result = function()
    if device.startswith("cuda"):
        import torch

        torch.cuda.synchronize(device)
    return result, {"seconds": time.perf_counter() - start, "device": device}
