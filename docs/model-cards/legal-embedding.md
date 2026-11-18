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
- legal
datasets:
- awinml/legalbench-rag
model-index:
- name: dar-legal-embedding-v1
  results:
  - task:
      type: feature-extraction
      name: Retrieval
    dataset:
      type: awinml/legalbench-rag
      name: awinml/legalbench-rag (document_scoped; 1015 test queries)
      split: test
      args:
        protocol: document_scoped
        test_queries: 1015
        capped_run: false
        max_sequence_length: 256
        candidate_top_k: 100
    metrics:
    - type: hit@10
      name: hit@10
      value: 0.9389162561576355
    - type: hit@5
      name: hit@5
      value: 0.8600985221674877
    - type: legal_character_precision@5
      name: Character Precision@5
      value: 0.06968197014395879
    - type: legal_character_recall@5
      name: Character Recall@5
      value: 0.7680431558590474
    - type: map@10
      name: map@10
      value: 0.5595272690630714
    - type: mrr@10
      name: MRR@10
      value: 0.7141684259910861
    - type: ndcg@10
      name: nDCG@10
      value: 0.6610951598847006
    - type: ndcg@5
      name: nDCG@5
      value: 0.6087429292880692
    - type: recall@10
      name: Recall@10
      value: 0.787687465790914
    - type: recall@100
      name: Recall@100
      value: 0.9848674642270702
    - type: recall@50
      name: Recall@50
      value: 0.9635491437954492
    - type: legal_source_balanced_char_recall@5
      name: Source-balanced Character Recall@5
      value: 0.7460068635497573
---

# Legal embedding

Prepared model ID: `nehal-vaghasiya/dar-legal-embedding-v1`.

Tuned Character Recall@5 **0.7680** vs **0.3984** base (Δ +0.3696). Research checkpoint for English domain retrieval.

Parent: [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5). 33,360,000 parameters; saved weights F32; 384-dimensional vectors. Evaluated sequence limit: **256 tokens**.

## Use

Install `sentence-transformers>=5.1,<6`; run from the downloaded model directory.

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer(".", device="cpu")
model.max_seq_length = 256
query_prefix = "Represent this sentence for searching relevant passages: "
query = query_prefix + "When can either party terminate the agreement?"
passages = ["A relevant passage.", "Another candidate passage."]
q = model.encode([query], normalize_embeddings=True)
docs = model.encode(passages, normalize_embeddings=True)
print(q @ docs.T)
```

## Evaluation

[awinml/legalbench-rag](https://huggingface.co/datasets/awinml/legalbench-rag); **1,015 test queries**. Document-scoped retrieval; custom split grouped by source document. Base and tuned embeddings retrieve independently from the same permitted scope.

| Metric | Base | Tuned | Δ | 95% CI of Δ |
|---|---:|---:|---:|---|
| Character Recall@5 | 0.3984 | 0.7680 | +0.3696 | [+0.3412, +0.3979] |
| nDCG@10 | 0.3159 | 0.6611 | +0.3452 | [+0.3249, +0.3643] |
| hit@10 | 0.6611 | 0.9389 | +0.2778 | — |
| hit@5 | 0.4867 | 0.8601 | +0.3734 | — |
| Character Precision@5 | 0.0312 | 0.0697 | +0.0385 | — |
| Source-balanced Character Recall@5 | 0.4884 | 0.7460 | +0.2576 | — |
| map@10 | 0.2234 | 0.5595 | +0.3362 | — |
| MRR@10 | 0.2887 | 0.7142 | +0.4255 | [+0.4009, +0.4520] |
| nDCG@5 | 0.2511 | 0.6087 | +0.3576 | — |
| Recall@10 | 0.5040 | 0.7877 | +0.2837 | — |
| Recall@100 | 0.9162 | 0.9849 | +0.0687 | [+0.0566, +0.0807] |
| Recall@50 | 0.8489 | 0.9635 | +0.1146 | — |

Higher is better; Δ = tuned − base. CIs are saved query-level paired bootstrap intervals (1,000 resamples); — means unrecorded. These results are specific to this protocol.

## Training

Train / validation / test queries: **4,848 / 1,026 / 1,015**. 5,996 training examples; `MultipleNegativesRankingLoss`; bm25 negatives.

| Setting | Value |
|---|---|
| Epochs / learning rate | 2 / 2e-05 |
| Batch / accumulation | 8 / 4 |
| Seed / training precision | 42 / fp16 |
| Selected checkpoint | checkpoint-376 |
| Selection | Validation nDCG@10 on 1,026 queries |

Training 225.5 s; peak allocated CUDA memory 1.58 GiB; GPU Tesla T4. Saved dtype and training precision are distinct.

## Limits and attribution

Custom source-document split; searches within one document, not a legal library. Character spans are unioned to avoid double counting. Mined unjudged negatives can be false negatives. Intended for retrieval experiments; downstream answer quality and deployment latency are unmeasured. Recorded latency is component-only and excludes embedding encoding. Base-model license: MIT; dataset rights and attribution remain governed by the linked dataset sources.

[Evaluation, settings and evidence](../results/legal.md) · [Source and reproduction](https://github.com/nehalvaghasiya/domain-adaptive-retrieval). Historical checkpoint hash unrecorded; current source digests provided. Upstream revisions were not pinned; no exact historical rerun is claimed.

<details>
<summary>Recorded environment</summary>

`sentence-transformers=5.7.0; transformers=4.57.6; datasets=4.8.5; torch=2.14.1+cu130; huggingface-hub=0.36.2; rank-bm25=0.2.2; numpy=2.4.6`

</details>
