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
- code
datasets:
- claudios/code_search_net
model-index:
- name: dar-code-embedding-v1
  results:
  - task:
      type: feature-extraction
      name: Retrieval
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
      config: python
    metrics:
    - type: hit@10
      name: hit@10
      value: 0.962
    - type: hit@5
      name: hit@5
      value: 0.948
    - type: map@10
      name: map@10
      value: 0.8633579365079366
    - type: mrr@10
      name: MRR@10
      value: 0.8633579365079366
    - type: ndcg@10
      name: nDCG@10
      value: 0.8879163313637065
    - type: ndcg@5
      name: nDCG@5
      value: 0.8833013021768564
    - type: precision@10
      name: precision@10
      value: 0.09620000000000001
    - type: precision@5
      name: precision@5
      value: 0.18960000000000002
    - type: recall@10
      name: Recall@10
      value: 0.962
    - type: recall@100
      name: Recall@100
      value: 0.986
    - type: recall@50
      name: Recall@50
      value: 0.984
---

# Code embedding

Prepared model ID: `nehal-vaghasiya/dar-code-embedding-v1`.

Tuned MRR@10 **0.8634** vs **0.8743** base (Δ -0.0109). Research checkpoint for English domain retrieval.

Parent: [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5). 33,360,000 parameters; saved weights F32; 384-dimensional vectors. Evaluated sequence limit: **256 tokens**.

## Use

Install `sentence-transformers>=5.1,<6`; run from the downloaded model directory.

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer(".", device="cpu")
model.max_seq_length = 256
query_prefix = "Represent this sentence for searching relevant passages: "
query = query_prefix + "Sort a list of records by date."
passages = ["A relevant passage.", "Another candidate passage."]
q = model.encode([query], normalize_embeddings=True)
docs = model.encode(passages, normalize_embeddings=True)
print(q @ docs.T)
```

## Evaluation

[claudios/code_search_net/python](https://huggingface.co/datasets/claudios/code_search_net); **500 test queries**. CodeSearchNet Python subset; 500 test queries search 500 test documents. Base and tuned embeddings retrieve independently from the same permitted scope.

| Metric | Base | Tuned | Δ | 95% CI of Δ |
|---|---:|---:|---:|---|
| MRR@10 | 0.8743 | 0.8634 | -0.0109 | [-0.0293, +0.0053] |
| nDCG@10 | 0.8976 | 0.8879 | -0.0097 | [-0.0253, +0.0041] |
| hit@10 | 0.9680 | 0.9620 | -0.0060 | — |
| hit@5 | 0.9480 | 0.9480 | +0.0000 | — |
| map@10 | 0.8743 | 0.8634 | -0.0109 | — |
| nDCG@5 | 0.8908 | 0.8833 | -0.0075 | — |
| precision@10 | 0.0968 | 0.0962 | -0.0006 | — |
| precision@5 | 0.1896 | 0.1896 | +0.0000 | — |
| Recall@10 | 0.9680 | 0.9620 | -0.0060 | — |
| Recall@100 | 0.9960 | 0.9860 | -0.0100 | — |
| Recall@50 | 0.9920 | 0.9840 | -0.0080 | — |

Higher is better; Δ = tuned − base. CIs are saved query-level paired bootstrap intervals (1,000 resamples); — means unrecorded. These results are specific to this protocol.

## Training

Train / validation / test queries: **15,000 / 500 / 500**. 15,000 training examples; `MultipleNegativesRankingLoss`; bm25 negatives.

| Setting | Value |
|---|---|
| Epochs / learning rate | 2.0 / 2e-05 |
| Batch / accumulation | 4 / 8 |
| Seed / training precision | 42 / fp16 |
| Selected checkpoint | last_checkpoint |
| Selection | Validation nDCG@10 on 100 queries |

Training 1123.6 s; peak allocated CUDA memory 0.65 GiB; GPU model unrecorded. Saved dtype and training precision are distinct.

## Limits and attribution

Python only; 15,000/500/500 split caps. Docstrings are removed from code and code is deduplicated across splits. The 500-document test pool is not the full CodeSearchNet benchmark. Mined unjudged negatives can be false negatives. Intended for retrieval experiments; downstream answer quality and deployment latency are unmeasured. Base-model license: MIT; dataset rights and attribution remain governed by the linked dataset sources.

[Evaluation details](#evaluation) · [Source and reproduction](https://github.com/nehalvaghasiya/domain-adaptive-retrieval). Recorded checkpoint hash verified. Upstream revisions were not pinned; no exact historical rerun is claimed.

<details>
<summary>Recorded environment</summary>

`python=3.11.11; platform=Linux-6.18.58.1-microsoft-standard-WSL2-x86_64-with-glibc2.35; torch=2.14.1+cu126; sentence-transformers=5.7.0; transformers=4.57.6; datasets=5.0.1; numpy=2.4.6; pydantic=2.14.3`

</details>
