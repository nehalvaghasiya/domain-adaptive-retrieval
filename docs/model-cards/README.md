# Hugging Face model cards
Prepared locally for `nehal-vaghasiya`. Model IDs are planned; repositories and collection have not been published.
| Model | Prepared model ID |
|---|---|
| [Finance embedding](finance-embedding.md) | `nehal-vaghasiya/dar-finance-embedding-v1` |
| [Finance reranker](finance-reranker.md) | `nehal-vaghasiya/dar-finance-reranker-v1` |
| [Biomedical embedding](biomedical-embedding.md) | `nehal-vaghasiya/dar-biomedical-embedding-v1` |
| [Biomedical reranker](biomedical-reranker.md) | `nehal-vaghasiya/dar-biomedical-reranker-v1` |
| [Legal embedding](legal-embedding.md) | `nehal-vaghasiya/dar-legal-embedding-v1` |
| [E-commerce embedding](ecommerce-embedding.md) | `nehal-vaghasiya/dar-ecommerce-embedding-v1` |
| [E-commerce reranker](ecommerce-reranker.md) | `nehal-vaghasiya/dar-ecommerce-reranker-v1` |
| [Code embedding](code-embedding.md) | `nehal-vaghasiya/dar-code-embedding-v1` |
| [Code reranker](code-reranker.md) | `nehal-vaghasiya/dar-code-reranker-v1` |

[Collection draft](collection.json). Each local package contains inference files, a model card, evaluation evidence and checksums.

Reproduce a local package with the report command's `--package-output dist/huggingface` option. Run the card's inference example from inside its package directory. Source dataset/model revisions were not pinned.
