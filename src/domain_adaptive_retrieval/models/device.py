"""Validate requested devices and verify actual model parameter placement."""

from typing import Any


def normalize_device(device: str) -> str:
    """Accept CPU or an explicit CUDA index without probing hardware."""
    value = device.strip().lower()
    if value == "cpu":
        return value
    if value == "cuda":
        return "cuda:0"
    if value.startswith("cuda:") and value[5:].isdecimal():
        return f"cuda:{int(value[5:])}"
    raise ValueError(f"Unsupported device {device!r}; expected cpu, cuda or cuda:N")


def check_device(device: str, *, require_gpu: bool = False) -> dict[str, Any]:
    """Probe CUDA with a real allocation; keep CPU checks independent of Torch."""
    normalized = normalize_device(device)
    if normalized == "cpu":
        if require_gpu:
            raise RuntimeError("require_gpu=true requires an NVIDIA CUDA device")
        return {"device": normalized}

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("CUDA requires Torch: uv sync --extra train") from exc

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; select an NVIDIA GPU runtime")
    index = int(normalized[5:])
    if index >= torch.cuda.device_count():
        raise RuntimeError(f"CUDA device index unavailable: {normalized}")
    properties = torch.cuda.get_device_properties(index)
    probe = torch.ones(1, device=normalized)
    torch.cuda.synchronize(normalized)
    if str(probe.device) != normalized:
        raise RuntimeError(
            f"CUDA allocation is on {probe.device}, expected {normalized}"
        )
    return {
        "device": normalized,
        "name": properties.name,
        "total_memory_bytes": properties.total_memory,
    }


def verify_model_device(model: Any, device: str) -> None:
    """Check every parameter, including CrossEncoder's wrapped Torch module."""
    expected = normalize_device(device)
    parameters = getattr(model, "parameters", None)
    if parameters is None:
        parameters = getattr(getattr(model, "model", None), "parameters", None)
    if parameters is None:
        raise RuntimeError("Model does not expose parameters for device verification")
    devices = {str(parameter.device) for parameter in parameters()}
    if devices != {expected}:
        raise RuntimeError(
            f"Model parameters are on {sorted(devices)}, expected {expected}"
        )
