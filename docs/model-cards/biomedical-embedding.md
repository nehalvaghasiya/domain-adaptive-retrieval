---
language:
- en
library_name: sentence-transformers
pipeline_tag: feature-extraction
base_model: BAAI/bge-small-en-v1.5
base_model_relation: finetune
license: mit
tags:
- retrieval
- domain-adaptation
- embedding
- biomedical
datasets:
- BeIR/nfcorpus
model-index:
- name: dar-biomedical-embedding-v1
  results:
  - task:
      type: feature-extraction
      name: Retrieval
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
    metrics:
    - type: hit@10
      name: hit@10
      value: 0.6934984520123839
    - type: hit@5
      name: hit@5
      value: 0.628482972136223
    - type: map@10
      name: map@10
      value: 0.25270264977335966
    - type: mrr@10
      name: MRR@10
      value: 0.5200759251068848
    - type: ndcg@10
      name: nDCG@10
      value: 0.34386370893339047
    - type: ndcg@5
      name: nDCG@5
      value: 0.3639241969049194
    - type: recall@10
      name: Recall@10
      value: 0.1673393146024562
    - type: recall@100
      name: Recall@100
      value: 0.33067265151540864
    - type: recall@50
      name: Recall@50
      value: 0.2727581915422186
---

# Biomedical embedding

Prepared model ID: `nehal-vaghasiya/dar-biomedical-embedding-v1`.

Tuned nDCG@10 **0.3439** vs **0.3461** base (Δ -0.0022). Research checkpoint for English domain retrieval.

Parent: [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5). 33,360,000 parameters; saved weights F32; 384-dimensional vectors. Evaluated sequence limit: **256 tokens**.

## Use

Install `sentence-transformers>=5.1,<6`; run from the downloaded model directory.

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer(".", device="cpu")
model.max_seq_length = 256
query_prefix = "Represent this sentence for searching relevant passages: "
query = query_prefix + "What are the effects of dietary fiber?"
passages = ["A relevant passage.", "Another candidate passage."]
q = model.encode([query], normalize_embeddings=True)
docs = model.encode(passages, normalize_embeddings=True)
print(q @ docs.T)
```

## Evaluation

[BEIR/nfcorpus](https://huggingface.co/datasets/BeIR/nfcorpus); **323 test queries**. Full corpus. Base and tuned embeddings retrieve independently from the same permitted scope.

| Metric | Base | Tuned | Δ | 95% CI of Δ |
|---|---:|---:|---:|---|
| nDCG@10 | 0.3461 | 0.3439 | -0.0022 | [-0.0155, +0.0111] |
| Recall@100 | 0.2967 | 0.3307 | +0.0340 | [+0.0182, +0.0483] |
| hit@10 | 0.7059 | 0.6935 | -0.0124 | — |
| hit@5 | 0.6440 | 0.6285 | -0.0155 | — |
| map@10 | 0.2518 | 0.2527 | +0.0009 | — |
| MRR@10 | 0.5364 | 0.5201 | -0.0163 | [-0.0405, +0.0079] |
| nDCG@5 | 0.3744 | 0.3639 | -0.0105 | — |
| Recall@10 | 0.1628 | 0.1673 | +0.0045 | — |
| Recall@50 | 0.2489 | 0.2728 | +0.0239 | — |

Higher is better; Δ = tuned − base. CIs are saved query-level paired bootstrap intervals (1,000 resamples); — means unrecorded. These results are specific to this protocol.

## Training

Train / validation / test queries: **2,590 / 324 / 323**. 6,000 training examples; `MultipleNegativesRankingLoss`; bm25 negatives.

| Setting | Value |
|---|---|
| Epochs / learning rate | 2 / 2e-05 |
| Batch / accumulation | 8 / 4 |
| Seed / training precision | 42 / fp16 |
| Selected checkpoint | checkpoint-188 |
| Selection | Validation nDCG@10 on 324 queries |

Training 258.3 s; peak allocated CUDA memory 1.53 GiB; GPU Tesla T4. Saved dtype and training precision are distinct.

## Limits and attribution

Full-corpus evaluation after recorded empty-text filtering. Sparse judgments can miss relevant documents. Mined unjudged negatives can be false negatives. Intended for retrieval experiments; downstream answer quality and deployment latency are unmeasured. Recorded latency is component-only and excludes embedding encoding. Base-model license: MIT; dataset rights and attribution remain governed by the linked dataset sources.

[Evaluation details](#evaluation) · [Source and reproduction](https://github.com/nehalvaghasiya/domain-adaptive-retrieval). Historical checkpoint hash unrecorded; current source digests provided. Upstream revisions were not pinned; no exact historical rerun is claimed.

<details>
<summary>Recorded environment</summary>

`sentence-transformers=5.7.0; transformers=4.57.6; datasets=4.8.5; torch=2.14.1+cu130; huggingface-hub=0.36.2; rank-bm25=0.2.2; numpy=2.4.6`

</details>
