"""Device preflight must reject unavailable CUDA without falling back to CPU."""

import sys
from types import SimpleNamespace

import pytest

from domain_adaptive_retrieval.models.device import (
    check_device,
    normalize_device,
    verify_model_device,
)


@pytest.mark.parametrize("device", ["", "cuda:-1", "cuda:abc", "cuda:", "gpu"])
def test_invalid_device_rejected(device: str) -> None:
    with pytest.raises(ValueError, match="Unsupported device"):
        normalize_device(device)


def test_cpu_preflight_does_not_require_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", None)
    assert check_device(" CPU ") == {"device": "cpu"}
    with pytest.raises(RuntimeError, match="require_gpu"):
        check_device("cpu", require_gpu=True)
    with pytest.raises(RuntimeError, match="CUDA requires Torch"):
        check_device("cuda", require_gpu=True)


def test_cuda_preflight_checks_index_and_allocates_on_requested_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def allocate(size: int, *, device: str) -> SimpleNamespace:
        calls.append(device)
        return SimpleNamespace(device=device)

    fake = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 1,
            get_device_properties=lambda index: SimpleNamespace(
                name="unit-test device", total_memory=1024
            ),
            synchronize=lambda device: calls.append(device),
        ),
        ones=allocate,
    )
    monkeypatch.setitem(sys.modules, "torch", fake)
    with pytest.raises(RuntimeError, match="index unavailable"):
        check_device("cuda:1")
    assert calls == []
    assert check_device("cuda", require_gpu=True) == {
        "device": "cuda:0",
        "name": "unit-test device",
        "total_memory_bytes": 1024,
    }
    assert calls == ["cuda:0", "cuda:0"]
    monkeypatch.setattr(
        fake, "ones", lambda *args, **kwargs: SimpleNamespace(device="cpu")
    )
    with pytest.raises(RuntimeError, match="expected cuda:0"):
        check_device("cuda:0")


def test_model_verification_rejects_missing_and_mixed_parameters() -> None:
    with pytest.raises(RuntimeError, match="does not expose parameters"):
        verify_model_device(object(), "cpu")
    model = SimpleNamespace(
        parameters=lambda: iter(
            [SimpleNamespace(device="cpu"), SimpleNamespace(device="cuda:0")]
        )
    )
    with pytest.raises(RuntimeError, match="expected cuda:0"):
        verify_model_device(model, "cuda:0")
