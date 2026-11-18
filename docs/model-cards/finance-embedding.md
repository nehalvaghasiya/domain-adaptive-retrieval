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
- finance
datasets:
- BeIR/fiqa
model-index:
- name: dar-finance-embedding-v1
  results:
  - task:
      type: feature-extraction
      name: Retrieval
    dataset:
      type: BeIR/fiqa
      name: BEIR/fiqa (full_corpus; 648 test queries)
      split: test
      args:
        protocol: full_corpus
        test_queries: 648
        capped_run: false
        max_sequence_length: 384
        candidate_top_k: 100
    metrics:
    - type: hit@10
      name: hit@10
      value: 0.6558641975308642
    - type: hit@5
      name: hit@5
      value: 0.5817901234567902
    - type: map@10
      name: map@10
      value: 0.3111766690987094
    - type: mrr@10
      name: MRR@10
      value: 0.46891166960611397
    - type: ndcg@10
      name: nDCG@10
      value: 0.3857078482729074
    - type: ndcg@5
      name: nDCG@5
      value: 0.36275986965250184
    - type: recall@10
      name: Recall@10
      value: 0.4455701148988186
    - type: recall@100
      name: Recall@100
      value: 0.7131166244592171
    - type: recall@50
      name: Recall@50
      value: 0.6289242093640242
---

# Finance embedding

Prepared model ID: `nehal-vaghasiya/dar-finance-embedding-v1`.

Tuned nDCG@10 **0.3857** vs **0.3995** base (Δ -0.0138). Research checkpoint for English domain retrieval.

Parent: [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5). 33,360,000 parameters; saved weights F32; 384-dimensional vectors. Evaluated sequence limit: **384 tokens**.

## Use

Install `sentence-transformers>=5.1,<6`; run from the downloaded model directory.

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer(".", device="cpu")
model.max_seq_length = 384
query_prefix = "Represent this sentence for searching relevant passages: "
query = query_prefix + "How can a company improve its cash flow?"
passages = ["A relevant passage.", "Another candidate passage."]
q = model.encode([query], normalize_embeddings=True)
docs = model.encode(passages, normalize_embeddings=True)
print(q @ docs.T)
```

## Evaluation

[BEIR/fiqa](https://huggingface.co/datasets/BeIR/fiqa); **648 test queries**. Full corpus after empty-text filtering. Base and tuned embeddings retrieve independently from the same permitted scope.

| Metric | Base | Tuned | Δ | 95% CI of Δ |
|---|---:|---:|---:|---|
| nDCG@10 | 0.3995 | 0.3857 | -0.0138 | [-0.0269, -0.0012] |
| Recall@100 | 0.6898 | 0.7131 | +0.0234 | [+0.0065, +0.0425] |
| hit@10 | 0.6667 | 0.6559 | -0.0108 | — |
| hit@5 | 0.5926 | 0.5818 | -0.0108 | — |
| map@10 | 0.3252 | 0.3112 | -0.0140 | — |
| MRR@10 | 0.4838 | 0.4689 | -0.0149 | [-0.0336, +0.0039] |
| nDCG@5 | 0.3754 | 0.3628 | -0.0126 | — |
| Recall@10 | 0.4595 | 0.4456 | -0.0140 | — |
| Recall@50 | 0.6171 | 0.6289 | +0.0118 | — |

Higher is better; Δ = tuned − base. CIs are saved query-level paired bootstrap intervals (1,000 resamples); — means unrecorded. These results are specific to this protocol.

## Training

Train / validation / test queries: **5,498 / 500 / 648**. 6,000 training examples; `MultipleNegativesRankingLoss`; bm25 negatives.

| Setting | Value |
|---|---|
| Epochs / learning rate | 2 / 2e-05 |
| Batch / accumulation | 16 / 2 |
| Seed / training precision | 42 / unrecorded |
| Selected checkpoint | checkpoint-188 |
| Selection | Validation nDCG@10 on 500 queries |

Training 239.2 s; peak allocated CUDA memory 3.23 GiB; GPU model unrecorded. Saved dtype and training precision are distinct.

## Limits and attribution

Full-corpus evaluation after recorded empty-text filtering. Sparse judgments can miss relevant documents. Mined unjudged negatives can be false negatives. Intended for retrieval experiments; downstream answer quality and deployment latency are unmeasured. Recorded latency is component-only and excludes embedding encoding. Base-model license: MIT; dataset rights and attribution remain governed by the linked dataset sources.

[Evaluation details](#evaluation) · [Source and reproduction](https://github.com/nehalvaghasiya/domain-adaptive-retrieval). Historical checkpoint hash unrecorded; current source digests provided. Upstream revisions were not pinned; no exact historical rerun is claimed.

<details>
<summary>Recorded environment</summary>

`sentence-transformers=5.7.0; transformers=4.57.6; datasets=4.8.5; torch=2.14.1+cu130; huggingface-hub=0.36.2; rank-bm25=0.2.2; numpy=2.4.6`

</details>
