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
- biomedical
datasets:
- BeIR/nfcorpus
model-index:
- name: dar-biomedical-reranker-v1
  results:
  - task:
      type: text-ranking
      name: Reranking
    dataset:
      type: BeIR/nfcorpus
      name: BEIR/nfcorpus (full_corpus; 323 test queries)
      split: test
      args:
        protocol: full_corpus
        test_queries: 323
        capped_run: false
        max_sequence_length: 256
        candidate_top_k: 100
        candidate_source: BAAI/bge-small-en-v1.5
    metrics:
    - type: hit@10
      name: hit@10
      value: 0.6191950464396285
    - type: hit@5
      name: hit@5
      value: 0.544891640866873
    - type: map@10
      name: map@10
      value: 0.19659465979374205
    - type: mrr@10
      name: MRR@10
      value: 0.4443830163644405
    - type: ndcg@10
      name: nDCG@10
      value: 0.26475001490244016
    - type: ndcg@5
      name: nDCG@5
      value: 0.28458115787513727
    - type: precision@10
      name: precision@10
      value: 0.22786377708978325
    - type: precision@5
      name: precision@5
      value: 0.27925696594427246
    - type: recall@10
      name: Recall@10
      value: 0.10993803581195764
    - type: recall@100
      name: Recall@100
      value: 0.2967035723557199
    - type: recall@50
      name: Recall@50
      value: 0.21862582514586726
---

# Biomedical reranker

Prepared model ID: `nehal-vaghasiya/dar-biomedical-reranker-v1`.

Tuned nDCG@10 **0.2648** vs **0.3492** base (Δ -0.0845). Research checkpoint for English domain retrieval.

Parent: [cross-encoder/ms-marco-MiniLM-L6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2). 22,713,601 parameters; saved weights F32. Evaluated sequence limit: **256 tokens**.

## Use

Install `sentence-transformers>=5.1,<6`; run from the downloaded model directory.

```python
from sentence_transformers import CrossEncoder

model = CrossEncoder(".", device="cpu", max_length=256)
query = "What are the effects of dietary fiber?"
passages = ["A relevant passage.", "Another candidate passage."]
scores = model.predict([(query, passage) for passage in passages])
print(sorted(zip(scores, passages), reverse=True))
```

## Evaluation

[BEIR/nfcorpus](https://huggingface.co/datasets/BeIR/nfcorpus); **323 test queries**. Full corpus. Both rerankers receive the same top-100 candidates from the base BGE model. Scores are relevance logits, not calibrated probabilities.

| Metric | Base | Tuned | Δ | 95% CI of Δ |
|---|---:|---:|---:|---|
| nDCG@10 | 0.3492 | 0.2648 | -0.0845 | [-0.1118, -0.0569] |
| Recall@100 | 0.2967 | 0.2967 | +0.0000 | — |
| hit@10 | 0.6997 | 0.6192 | -0.0805 | — |
| hit@5 | 0.6594 | 0.5449 | -0.1146 | — |
| map@10 | 0.2540 | 0.1966 | -0.0574 | — |
| MRR@10 | 0.5741 | 0.4444 | -0.1297 | [-0.1766, -0.0849] |
| nDCG@5 | 0.3830 | 0.2846 | -0.0984 | — |
| precision@10 | 0.2480 | 0.2279 | -0.0201 | — |
| precision@5 | 0.3214 | 0.2793 | -0.0421 | — |
| Recall@10 | 0.1618 | 0.1099 | -0.0518 | — |
| Recall@50 | 0.2478 | 0.2186 | -0.0292 | — |

Higher is better; Δ = tuned − base. CIs are saved query-level paired bootstrap intervals (1,000 resamples); — means unrecorded. These results are specific to this protocol.

## Training

Train / validation / test queries: **2,590 / 324 / 323**. 331,725 training examples; `BinaryCrossEntropyLoss`; bm25 negatives.

| Setting | Value |
|---|---|
| Epochs / learning rate | 2.0 / 2e-05 |
| Batch / accumulation | 4 / 8 |
| Seed / training precision | 42 / fp16 |
| Selected checkpoint | checkpoint-20600 |
| Selection | Validation nDCG@10 on 100 queries |

Training 5612.1 s; peak allocated CUDA memory 0.41 GiB; GPU model unrecorded. Saved dtype and training precision are distinct.

## Limits and attribution

Full-corpus evaluation after recorded empty-text filtering. Sparse judgments can miss relevant documents. Mined unjudged negatives can be false negatives. Intended for retrieval experiments; downstream answer quality and deployment latency are unmeasured. Base-model license: Apache-2.0; dataset rights and attribution remain governed by the linked dataset sources.

[Evaluation details](#evaluation) · [Source and reproduction](https://github.com/nehalvaghasiya/domain-adaptive-retrieval). Recorded checkpoint hash verified. Upstream revisions were not pinned; no exact historical rerun is claimed.

<details>
<summary>Recorded environment</summary>

`python=3.11.11; platform=Linux-6.18.58.1-microsoft-standard-WSL2-x86_64-with-glibc2.35; torch=2.14.1+cu126; sentence-transformers=5.7.0; transformers=4.57.6; datasets=5.0.1; numpy=2.4.6; pydantic=2.14.3`

</details>
