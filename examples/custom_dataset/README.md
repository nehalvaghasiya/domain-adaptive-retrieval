# Custom dataset (small offline example)

Three JSONL tables are required: `corpus.jsonl` (`doc_id`, `text`),
`queries.jsonl` (`query_id`, `text`) and `qrels.jsonl` (`query_id`, `doc_id`,
`split`, `score`). Split is `train`, `val` or `test`; grades are nonnegative.
Data may alternatively be TSV, CSV or Parquet (install the `benchmark` extra).

Optional tables: `candidates` for query-specific **judged** products, `spans`
(`query_id`, `source_id`, `start`, `end`) for legal evidence and `esci_labels`
for the original `E`/`S`/`C`/`I` grade labels. To search only a source document,
add `source_id`, `start`, `end` in `corpus`, and `scope` in `queries`.

To validate or train your own data, set `domain: custom` and
`data.custom_dir: /your/path` in YAML. Never use test qrels to generate training
negatives or select checkpoints. For a quick sample run, `dar smoke --offline`.
