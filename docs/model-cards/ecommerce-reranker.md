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
- ecommerce
---

# E-commerce reranker

Prepared model ID: `nehal-vaghasiya/dar-ecommerce-reranker-v1`.

Tuned nDCG@10 **0.7828** vs **0.7858** base (Δ -0.0031). Research checkpoint for English domain retrieval.

Parent: [cross-encoder/ms-marco-MiniLM-L6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2). 22,713,601 parameters; saved weights F32. Evaluated sequence limit: **256 tokens**.

## Use

Install `sentence-transformers>=5.1,<6`; run from the downloaded model directory.

```python
from sentence_transformers import CrossEncoder

model = CrossEncoder(".", device="cpu", max_length=256)
query = "waterproof hiking shoes"
passages = ["A relevant passage.", "Another candidate passage."]
scores = model.predict([(query, passage) for passage in passages])
print(sorted(zip(scores, passages), reverse=True))
```

## Evaluation

[Amazon ESCI Task 1 us](https://github.com/amazon-science/esci-data); **8,956 test queries**. Amazon ESCI US Task 1; only products judged for each query. Both rerankers receive the same top-100 candidates from the base BGE model. Scores are relevance logits, not calibrated probabilities.

| Metric | Base | Tuned | Δ | 95% CI of Δ |
|---|---:|---:|---:|---|
| Exact MRR@10 | 0.7868 | 0.7730 | -0.0138 | — |
| nDCG@10 | 0.7858 | 0.7828 | -0.0031 | [-0.0069, +0.0008] |
| Exact Hit@10 | 0.9791 | 0.9707 | -0.0084 | — |
| MRR@10 | 0.7868 | 0.7730 | -0.0138 | [-0.0214, -0.0063] |
| nDCG@5 | 0.7500 | 0.7477 | -0.0023 | — |

Higher is better; Δ = tuned − base. CIs are saved query-level paired bootstrap intervals (1,000 resamples); — means unrecorded. These results are specific to this protocol.

## Training

Train / validation / test queries: **17,755 / 3,133 / 8,956**. 458,298 training examples; `BinaryCrossEntropyLoss`; bm25 negatives.

| Setting | Value |
|---|---|
| Epochs / learning rate | 2.0 / 2e-05 |
| Batch / accumulation | 4 / 8 |
| Seed / training precision | 42 / fp16 |
| Selected checkpoint | last_checkpoint |
| Selection | Validation nDCG@10 on 100 queries |

Training 7232.8 s; peak allocated CUDA memory 0.40 GiB; GPU model unrecorded. Saved dtype and training precision are distinct.

## Limits and attribution

Grades E/S/C/I = 3/2/0/0; Exact MRR counts E only. Results cover judged product lists, not whole-catalog retrieval. Mined unjudged negatives can be false negatives. Intended for retrieval experiments; downstream answer quality and deployment latency are unmeasured. Base-model license: Apache-2.0; dataset rights and attribution remain governed by the linked dataset sources.

[Evaluation details](#evaluation) · [Source and reproduction](https://github.com/nehalvaghasiya/domain-adaptive-retrieval). Recorded checkpoint hash verified. Upstream revisions were not pinned; no exact historical rerun is claimed.

<details>
<summary>Recorded environment</summary>

`python=3.11.11; platform=Linux-6.18.58.1-microsoft-standard-WSL2-x86_64-with-glibc2.35; torch=2.14.1+cu126; sentence-transformers=5.7.0; transformers=4.57.6; datasets=5.0.1; numpy=2.4.6; pydantic=2.14.3`

</details>
