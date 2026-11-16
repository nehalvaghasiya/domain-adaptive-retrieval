# Security

Report suspected vulnerabilities privately through the GitHub security advisory workflow if the repository is published. Do **not** open a public issue with tokens, credentials or private customer data.

- Never commit `.env`, `HF_TOKEN`, model keys, caches, downloadable copyrighted corpora, checkpoint weights or user documents.
- Model downloads and dataset archives should come from their published upstream locations. BEIR archives are checked for unsafe traversal paths before extraction.
- Model cards and publishing are opt-in; the Hub repository is private by default.
- Treat datasets, query text and model artifacts as untrusted input. Public repos do not guarantee third-party data redistribution rights.
