---
language:
- en
library_name: sentence-transformers
pipeline_tag: text-ranking
base_model: cross-encoder/ms-marco-MiniLM-L6-v2
base_model_relation: finetune
license: apache-2.0
tags:
- retrieval
- domain-adaptation
- reranker
- finance
datasets:
- BeIR/fiqa
model-index:
- name: dar-finance-reranker-v1
  results:
  - task:
      type: text-ranking
      name: Reranking
    dataset:
      type: BeIR/fiqa
      name: BEIR/fiqa (full_corpus; 648 test queries)
      split: test
      args:
        protocol: full_corpus
        test_queries: 648
        capped_run: false
        max_sequence_length: 256
        candidate_top_k: 100
        candidate_source: BAAI/bge-small-en-v1.5
    metrics:
    - type: hit@10
      name: hit@10
      value: 0.18981481481481483
    - type: hit@5
      name: hit@5
      value: 0.12654320987654322
    - type: map@10
      name: map@10
      value: 0.041019815219183235
    - type: mrr@10
      name: MRR@10
      value: 0.09046149323927102
    - type: ndcg@10
      name: nDCG@10
      value: 0.06446637860189887
    - type: ndcg@5
      name: nDCG@5
      value: 0.05346207969841174
    - type: precision@10
      name: precision@10
      value: 0.022993827160493828
    - type: precision@5
      name: precision@5
      value: 0.029320987654320986
    - type: recall@10
      name: Recall@10
      value: 0.0873322065451695
    - type: recall@100
      name: Recall@100
      value: 0.6830050479356035
    - type: recall@50
      name: Recall@50
      value: 0.3601590924507591
---

# Finance reranker

Prepared model ID: `nehal-vaghasiya/dar-finance-reranker-v1`.

Tuned nDCG@10 **0.0645** vs **0.3754** base (Δ -0.3109). Research checkpoint for English domain retrieval.

Parent: [cross-encoder/ms-marco-MiniLM-L6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2). 22,713,601 parameters; saved weights F32. Evaluated sequence limit: **256 tokens**.

## Use

Install `sentence-transformers>=5.1,<6`; run from the downloaded model directory.

```python
from sentence_transformers import CrossEncoder

model = CrossEncoder(".", device="cpu", max_length=256)
query = "How can a company improve its cash flow?"
passages = ["A relevant passage.", "Another candidate passage."]
scores = model.predict([(query, passage) for passage in passages])
print(sorted(zip(scores, passages), reverse=True))
```

## Evaluation

[BEIR/fiqa](https://huggingface.co/datasets/BeIR/fiqa); **648 test queries**. Full corpus after empty-text filtering. Both rerankers receive the same top-100 candidates from the base BGE model. Scores are relevance logits, not calibrated probabilities.

| Metric | Base | Tuned | Δ | 95% CI of Δ |
|---|---:|---:|---:|---|
| nDCG@10 | 0.3754 | 0.0645 | -0.3109 | [-0.3409, -0.2829] |
| Recall@100 | 0.6830 | 0.6830 | +0.0000 | — |
| hit@10 | 0.6559 | 0.1898 | -0.4660 | — |
| hit@5 | 0.5725 | 0.1265 | -0.4460 | — |
| map@10 | 0.2989 | 0.0410 | -0.2579 | — |
| MRR@10 | 0.4453 | 0.0905 | -0.3549 | [-0.3915, -0.3203] |
| nDCG@5 | 0.3515 | 0.0535 | -0.2980 | — |
| precision@10 | 0.1051 | 0.0230 | -0.0821 | — |
| precision@5 | 0.1701 | 0.0293 | -0.1407 | — |
| Recall@10 | 0.4510 | 0.0873 | -0.3636 | — |
| Recall@50 | 0.6164 | 0.3602 | -0.2563 | — |

Higher is better; Δ = tuned − base. CIs are saved query-level paired bootstrap intervals (1,000 resamples); — means unrecorded. These results are specific to this protocol.

## Training

Train / validation / test queries: **5,498 / 500 / 648**. 42,393 training examples; `BinaryCrossEntropyLoss`; bm25 negatives.

| Setting | Value |
|---|---|
| Epochs / learning rate | 2.0 / 2e-05 |
| Batch / accumulation | 4 / 8 |
| Seed / training precision | 42 / fp16 |
| Selected checkpoint | checkpoint-2500 |
| Selection | Validation nDCG@10 on 100 queries |

Training 729.6 s; peak allocated CUDA memory 0.41 GiB; GPU model unrecorded. Saved dtype and training precision are distinct.

## Limits and attribution

Full-corpus evaluation after recorded empty-text filtering. Sparse judgments can miss relevant documents. Mined unjudged negatives can be false negatives. Intended for retrieval experiments; downstream answer quality and deployment latency are unmeasured. Base-model license: Apache-2.0; dataset rights and attribution remain governed by the linked dataset sources.

[Evaluation details](#evaluation) · [Source and reproduction](https://github.com/nehalvaghasiya/domain-adaptive-retrieval). Recorded checkpoint hash verified. Upstream revisions were not pinned; no exact historical rerun is claimed.

<details>
<summary>Recorded environment</summary>

`python=3.11.11; platform=Linux-6.18.58.1-microsoft-standard-WSL2-x86_64-with-glibc2.35; torch=2.14.1+cu126; sentence-transformers=5.7.0; transformers=4.57.6; datasets=5.0.1; numpy=2.4.6; pydantic=2.14.3`

</details>
