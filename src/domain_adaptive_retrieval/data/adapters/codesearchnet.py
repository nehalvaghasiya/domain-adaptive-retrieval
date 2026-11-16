"""Python CodeSearchNet docstring-to-source-code retrieval."""

from typing import Any

from domain_adaptive_retrieval.config.schema import ExperimentConfig
from domain_adaptive_retrieval.data.bundle import RetrievalBundle
from domain_adaptive_retrieval.data.cleaning import clean_text, normalized_sha
from domain_adaptive_retrieval.data.dedupe import remove_python_docstring


def load_codesearchnet(config: ExperimentConfig) -> RetrievalBundle:
    """Stream official splits of a maintained Parquet mirror with cross-split dedup."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("CodeSearchNet requires: uv sync --extra train") from exc
    bundle = RetrievalBundle(meta={"dataset": "claudios/code_search_net/python", "protocol": "held_out_corpus", "preprocessing": "Python AST docstrings stripped; cross-split code SHA dedup"})
    seen_code: set[str] = set()
    training_ids: set[str] = set()
    split_map = {"train": "train", "val": "validation", "test": "test"}
    counts: dict[str, int] = {}
    split_documents: dict[str, list[str]] = {}
    for split, hf_split in split_map.items():
        cap = {"train": config.code_max_train, "val": config.code_max_val, "test": config.code_max_test}[split]
        stream: Any = load_dataset("claudios/code_search_net", "python", split=hf_split, streaming=True)
        if split == "train":
            stream = stream.shuffle(seed=config.seed, buffer_size=3000)
        added = 0
        candidate_documents: list[str] = []
        for row in stream:
            query = clean_text(row.get("func_documentation_string"))
            code = remove_python_docstring(clean_text(row.get("func_code_string")))
            if len(query) < 12 or len(code) < 30 or query.casefold() in code.casefold():
                continue
            sha = normalized_sha(code)
            if sha in seen_code:
                continue
            seen_code.add(sha)
            did, qid = f"code:{sha[:24]}", f"csn:{sha[:24]}"
            bundle.corpus[did] = code[: config.code_max_chars]
            bundle.queries[qid] = query[: config.code_max_query_chars]
            bundle.qrels[split][qid] = {did: 1.0}
            candidate_documents.append(did)
            if split == "train":
                training_ids.add(did)
            added += 1
            if cap and added >= cap:
                break
        counts[split] = added
        split_documents[split] = candidate_documents
    bundle.train_docs = training_ids
    bundle.split_docs = split_documents
    bundle.meta["capped_run"] = any((config.code_max_train, config.code_max_val, config.code_max_test))
    bundle.meta["data_quality"] = {"loaded_queries_by_split": counts}
    return bundle
