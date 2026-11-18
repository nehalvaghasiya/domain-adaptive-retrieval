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
- code
datasets:
- claudios/code_search_net
model-index:
- name: dar-code-reranker-v1
  results:
  - task:
      type: text-ranking
      name: Reranking
    dataset:
      type: claudios/code_search_net
      name: claudios/code_search_net/python (held_out_corpus; 500 test queries)
      split: test
      args:
        protocol: held_out_corpus
        test_queries: 500
        capped_run: true
        max_sequence_length: 256
        candidate_top_k: 100
        test_corpus_documents: 500
        candidate_source: BAAI/bge-small-en-v1.5
      config: python
    metrics:
    - type: hit@10
      name: hit@10
      value: 0.91
    - type: hit@5
      name: hit@5
      value: 0.848
    - type: map@10
      name: map@10
      value: 0.6691952380952381
    - type: mrr@10
      name: MRR@10
      value: 0.6691952380952381
    - type: ndcg@10
      name: nDCG@10
      value: 0.7278819372025406
    - type: ndcg@5
      name: nDCG@5
      value: 0.7080881726921391
    - type: precision@10
      name: precision@10
      value: 0.091
    - type: precision@5
      name: precision@5
      value: 0.16960000000000003
    - type: recall@10
      name: Recall@10
      value: 0.91
    - type: recall@100
      name: Recall@100
      value: 0.996
    - type: recall@50
      name: Recall@50
      value: 0.986
---

# Code reranker

Prepared model ID: `nehal-vaghasiya/dar-code-reranker-v1`.

Tuned MRR@10 **0.6692** vs **0.7345** base (Δ -0.0653). Research checkpoint for English domain retrieval.

Parent: [cross-encoder/ms-marco-MiniLM-L6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2). 22,713,601 parameters; saved weights F32. Evaluated sequence limit: **256 tokens**.

## Use

Install `sentence-transformers>=5.1,<6`; run from the downloaded model directory.

```python
from sentence_transformers import CrossEncoder

model = CrossEncoder(".", device="cpu", max_length=256)
query = "Sort a list of records by date."
passages = ["A relevant passage.", "Another candidate passage."]
scores = model.predict([(query, passage) for passage in passages])
print(sorted(zip(scores, passages), reverse=True))
```

## Evaluation

[claudios/code_search_net/python](https://huggingface.co/datasets/claudios/code_search_net); **500 test queries**. CodeSearchNet Python subset; 500 test queries search 500 test documents. Both rerankers receive the same top-100 candidates from the base BGE model. Scores are relevance logits, not calibrated probabilities.

| Metric | Base | Tuned | Δ | 95% CI of Δ |
|---|---:|---:|---:|---|
| MRR@10 | 0.7345 | 0.6692 | -0.0653 | [-0.1048, -0.0272] |
| nDCG@10 | 0.7818 | 0.7279 | -0.0539 | [-0.0874, -0.0212] |
| hit@10 | 0.9280 | 0.9100 | -0.0180 | — |
| hit@5 | 0.8840 | 0.8480 | -0.0360 | — |
| map@10 | 0.7345 | 0.6692 | -0.0653 | — |
| nDCG@5 | 0.7675 | 0.7081 | -0.0595 | — |
| precision@10 | 0.0928 | 0.0910 | -0.0018 | — |
| precision@5 | 0.1768 | 0.1696 | -0.0072 | — |
| Recall@10 | 0.9280 | 0.9100 | -0.0180 | — |
| Recall@100 | 0.9960 | 0.9960 | +0.0000 | — |
| Recall@50 | 0.9800 | 0.9860 | +0.0060 | — |

Higher is better; Δ = tuned − base. CIs are saved query-level paired bootstrap intervals (1,000 resamples); — means unrecorded. These results are specific to this protocol.

## Training

Train / validation / test queries: **15,000 / 500 / 500**. 45,000 training examples; `BinaryCrossEntropyLoss`; bm25 negatives.

| Setting | Value |
|---|---|
| Epochs / learning rate | 2.0 / 2e-05 |
| Batch / accumulation | 4 / 8 |
| Seed / training precision | 42 / fp16 |
| Selected checkpoint | last_checkpoint |
| Selection | Validation nDCG@10 on 100 queries |

Training 696.7 s; peak allocated CUDA memory 0.41 GiB; GPU model unrecorded. Saved dtype and training precision are distinct.

## Limits and attribution

Python only; 15,000/500/500 split caps. Docstrings are removed from code and code is deduplicated across splits. The 500-document test pool is not the full CodeSearchNet benchmark. Mined unjudged negatives can be false negatives. Intended for retrieval experiments; downstream answer quality and deployment latency are unmeasured. Base-model license: Apache-2.0; dataset rights and attribution remain governed by the linked dataset sources.

[Evaluation details](#evaluation) · [Source and reproduction](https://github.com/nehalvaghasiya/domain-adaptive-retrieval). Recorded checkpoint hash verified. Upstream revisions were not pinned; no exact historical rerun is claimed.

<details>
<summary>Recorded environment</summary>

`python=3.11.11; platform=Linux-6.18.58.1-microsoft-standard-WSL2-x86_64-with-glibc2.35; torch=2.14.1+cu126; sentence-transformers=5.7.0; transformers=4.57.6; datasets=5.0.1; numpy=2.4.6; pydantic=2.14.3`

</details>
