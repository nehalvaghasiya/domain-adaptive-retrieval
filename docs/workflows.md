# Legal embedding workflow

The legal encoder maps questions and passages to normalized 384-dimensional vectors. Similarity search retrieves passages to use as context for an LLM.

## Use the model

Install the inference and training dependencies:

```bash
uv sync --extra train --extra benchmark --extra report --group dev
```

With the checkpoint at `outputs/legal/embedding/model`, run this example with `uv run --no-sync python`:

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("outputs/legal/embedding/model", device="cpu")
model.max_seq_length = 256
prefix = "Represent this sentence for searching relevant passages: "
question = "When can either party terminate the agreement?"
passages = [
    "Either party may terminate with thirty days' written notice.",
    "Payment is due within fifteen days of the invoice.",
]
query = model.encode([prefix + question], normalize_embeddings=True)
documents = model.encode(passages, normalize_embeddings=True)
print((query @ documents.T).argsort(axis=1)[:, ::-1])
```

The checkpoint and Hugging Face release assets are prepared locally. The [model card](model-cards/legal-embedding.md) documents the model and its usage.

## Train and evaluate

Training downloads the configured dataset and requires a CUDA GPU. Use a separate output directory to retain the original checkpoint:

```bash
uv run --no-sync dar train embedder --domain legal --profile t4 --output outputs/reproduction
uv run --no-sync dar evaluate embedder --domain legal --profile t4 \
  --checkpoint outputs/reproduction/legal/embedding/model \
  --output outputs/reproduction/legal/embedding/evaluation
```

See the [legal configuration](../configs/domains/legal.yaml) and [T4 profile](../configs/experiments/t4.yaml) for settings. The saved model card records the original run's settings; a new run may differ.

Evaluation uses source-document splits and searches passages within each document. It measures passage retrieval; generated answers need separate evaluation. Read the [legal evaluation](results/legal.md) for scores and protocol details.

## Prepare Hugging Face assets

Regenerate the featured report and local model cards from saved evidence:

```bash
uv run --extra report dar report suite --input outputs \
  --featured-model legal/embedding --readme README.md
```

Add `--package-output dist/huggingface` to prepare release copies with weights, cards, evaluation evidence and checksums. This command works locally; uploading uses the separate `dar publish model` command. Existing Hugging Face cards and local packages are retained.

Weights, downloaded datasets, checkpoints and release packages are excluded from Git. Code is [MIT licensed](../LICENSE); datasets and model weights retain their own terms.
