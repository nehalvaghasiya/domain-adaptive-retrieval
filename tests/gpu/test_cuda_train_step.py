"""Actual CUDA forward/backward smoke for model libraries after installation."""

import pytest
import torch


@pytest.mark.gpu
@pytest.mark.network
@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_real_sentence_transformer_gpu_train_step() -> None:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cuda:0")
    features = model.tokenize(["finance question", "finance answer"])
    features = {k: v.to("cuda:0") for k, v in features.items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    model.train()
    vectors = model(features)["sentence_embedding"]
    loss = ((vectors[0] - vectors[1]) ** 2).mean()
    loss.backward()
    optimizer.step()
    assert loss.item() >= 0
    assert next(model.parameters()).device.type == "cuda"
