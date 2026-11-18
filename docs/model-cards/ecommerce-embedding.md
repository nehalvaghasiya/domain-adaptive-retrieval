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
- ecommerce
---

# E-commerce embedding

Prepared model ID: `nehal-vaghasiya/dar-ecommerce-embedding-v1`.

Tuned nDCG@10 **0.7836** vs **0.7951** base (Δ -0.0115). Research checkpoint for English domain retrieval.

Parent: [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5). 33,360,000 parameters; saved weights F32; 384-dimensional vectors. Evaluated sequence limit: **256 tokens**.

## Use

Install `sentence-transformers>=5.1,<6`; run from the downloaded model directory.

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer(".", device="cpu")
model.max_seq_length = 256
query_prefix = "Represent this sentence for searching relevant passages: "
query = query_prefix + "waterproof hiking shoes"
passages = ["A relevant passage.", "Another candidate passage."]
q = model.encode([query], normalize_embeddings=True)
docs = model.encode(passages, normalize_embeddings=True)
print(q @ docs.T)
```

## Evaluation

[Amazon ESCI Task 1 us](https://github.com/amazon-science/esci-data); **8,956 test queries**. Amazon ESCI US Task 1; only products judged for each query. Base and tuned embeddings retrieve independently from the same permitted scope.

| Metric | Base | Tuned | Δ | 95% CI of Δ |
|---|---:|---:|---:|---|
| Exact MRR@10 | 0.7909 | 0.7453 | -0.0456 | — |
| nDCG@10 | 0.7951 | 0.7836 | -0.0115 | [-0.0147, -0.0084] |
| Exact Hit@10 | 0.9814 | 0.9755 | -0.0058 | — |
| MRR@10 | 0.7909 | 0.7453 | -0.0456 | [-0.0525, -0.0381] |
| nDCG@5 | 0.7595 | 0.7392 | -0.0203 | — |

Higher is better; Δ = tuned − base. CIs are saved query-level paired bootstrap intervals (1,000 resamples); — means unrecorded. These results are specific to this protocol.

## Training

Train / validation / test queries: **17,755 / 3,133 / 8,956**. 154,564 training examples; `MultipleNegativesRankingLoss`; bm25 negatives.

| Setting | Value |
|---|---|
| Epochs / learning rate | 2.0 / 2e-05 |
| Batch / accumulation | 4 / 8 |
| Seed / training precision | 42 / fp16 |
| Selected checkpoint | checkpoint-9600 |
| Selection | Validation nDCG@10 on 100 queries |

Training 13032.4 s; peak allocated CUDA memory 0.60 GiB; GPU model unrecorded. Saved dtype and training precision are distinct.

## Limits and attribution

Grades E/S/C/I = 3/2/0/0; Exact MRR counts E only. Results cover judged product lists, not whole-catalog retrieval. Mined unjudged negatives can be false negatives. Intended for retrieval experiments; downstream answer quality and deployment latency are unmeasured. Base-model license: MIT; dataset rights and attribution remain governed by the linked dataset sources.

[Evaluation details](#evaluation) · [Source and reproduction](https://github.com/nehalvaghasiya/domain-adaptive-retrieval). Recorded checkpoint hash verified. Upstream revisions were not pinned; no exact historical rerun is claimed.

<details>
<summary>Recorded environment</summary>

`python=3.11.11; platform=Linux-6.18.58.1-microsoft-standard-WSL2-x86_64-with-glibc2.35; torch=2.14.1+cu126; sentence-transformers=5.7.0; transformers=4.57.6; datasets=5.0.1; numpy=2.4.6; pydantic=2.14.3`

</details>
