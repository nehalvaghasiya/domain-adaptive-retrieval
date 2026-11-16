# Domain Adaptive Retrieval

**Domain-adaptive embedding and cross-encoder fine-tuning, retrieval, reranking and reproducible benchmarking.**

Targets ten model variants (five BGE-small bi-encoders and five MiniLM rerankers) across Finance/FiQA, Biomedical/NFCorpus, LegalBench-RAG, Amazon ESCI and Python CodeSearchNet. Benchmark outputs must come from real held-out evaluations; **no model weights or benchmark improvements are bundled or claimed**.

## Quickstart (offline, CPU)

```bash
uv sync --group dev
uv run dar smoke --offline --config configs/experiments/smoke.yaml
uv run pytest -m 'not gpu and not network and not slow' -q
```

The offline smoke uses a redistributable synthetic corpus and needs neither a GPU nor network.
For training and full benchmark runs, install extras: `uv sync --extra train --extra benchmark --group dev`.

## Domains and protocols

| Domain | Dataset | Primary metric | Evaluation scope |
|---|---|---|---|
| Finance | BEIR FiQA-2018 | nDCG@10, Recall@100 | full corpus |
| Biomedical | BEIR NFCorpus | nDCG@10, Recall@100 | full corpus |
| Legal | LegalBench-RAG | character Recall@5 | per source document |
| E-commerce | Amazon ESCI (US) | graded nDCG@10, exact MRR@10 | only judged candidates |
| Code | CodeSearchNet Python | MRR@10, nDCG@10 | declared code pool |

**Status:** training checkpoints and full-domain benchmark scores pending independently executed T4 runs. No cross-domain aggregate leaderboard.

## Experiment matrix

S0 BM25; S1 base embedding; S2 tuned embedding; S3 S1 + base reranker; S4 S1 + tuned reranker; S5 S2 + base reranker; S6 S2 + tuned reranker. Rerankers must use identical first-stage candidate IDs for paired comparisons. No S2/S5/S6 claims without a tuned embedding checkpoint.

## CLI

```bash
dar config validate --config configs/domains/finance.yaml --profile configs/experiments/t4.yaml
dar hardware check --device cuda:0
dar data prepare --domain finance --output data/processed/finance
dar data audit --domain finance
dar train embedder --domain finance --profile t4 --output outputs/finance/embedder
dar train reranker --domain finance --profile t4 --output outputs/finance/reranker
dar evaluate embedder --domain finance --checkpoint outputs/finance/embedder/model
dar benchmark run --domain finance --embedder outputs/finance/embedder/model --reranker outputs/finance/reranker/model --output outputs/finance/benchmark
dar report build --run outputs/finance/benchmark
# Publication explicitly opt-in; private by default
dar publish model --run outputs/finance/embedder --repo-id USER/bge-small-fiqa-ft --private --confirm
```

Some commands require network access to download datasets/weights and NVIDIA CUDA for T4-profile training. Default CPU unit/integration tests never download either.

## Python API

```python
from domain_adaptive_retrieval.config.loader import load_config
from domain_adaptive_retrieval.workflows.benchmark import run_benchmark

config = load_config(
    "configs/domains/finance.yaml", profile="configs/experiments/t4.yaml"
)
# run_benchmark(config, embedding_checkpoint="...", reranker_checkpoint="...")
```

## Verification

Offline checks: **119 passing tests, 88.71% coverage, six synthetic domain dry runs**. GPU training, public-dataset downloads, lint/mypy, dependency resolution and model publication were **not** executed in this environment. Do not treat the synthetic results as real checkpoints or benchmark claims.

## Developer workflow

`make fmt`, `make check`, `make smoke`, and `make build`; PR checks enforce lint, mypy strict, tests, notebook hygiene and package build. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Research limitations

Sparse qrels cannot establish every unjudged document is irrelevant. ESCI scores are judged-list only; legal scores depend on original character offsets and source-grouped splits; code search removes original docstrings but may still contain semantically duplicated code. Training negatives may be false negatives. Published checkpoints require independent license review; retrieval relevance is not medical, legal or financial advice.

## Acknowledgments and license

Base models: [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5), [cross-encoder/ms-marco-MiniLM-L6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2). Original repository code: [MIT](LICENSE); datasets and model weights have their own licenses.
