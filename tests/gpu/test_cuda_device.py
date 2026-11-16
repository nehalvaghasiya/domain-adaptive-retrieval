"""Run manually on a real T4; skipped on CPU CI."""

import pytest
import torch
from domain_adaptive_retrieval.models.device import check_device


@pytest.mark.gpu
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="NVIDIA CUDA hardware required"
)
def test_cuda_real_allocation() -> None:
    info = check_device("cuda", require_gpu=True)
    assert info["device"] == "cuda:0"
    tensor = torch.rand((3, 3), device="cuda:0")
    result = tensor @ tensor
    assert result.device.type == "cuda"
