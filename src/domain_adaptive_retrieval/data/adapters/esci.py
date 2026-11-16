"""Amazon ESCI: candidate-scoped product relevance judgments."""

import random
from pathlib import Path
from typing import Any

from domain_adaptive_retrieval.data.bundle import RetrievalBundle
from domain_adaptive_retrieval.data.cleaning import clean_text

ESCI_GRADE = {"E": 3.0, "S": 2.0, "C": 0.0, "I": 0.0}
ESCI_DATA_ROOT = "https://media.githubusercontent.com/media/amazon-science/esci-data/main/shopping_queries_dataset/"


def _download_esci(path: Path) -> None:
    """Download the official (large) files to a local cache only on demand."""
    import shutil
    import urllib.request

    if path.is_file():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(ESCI_DATA_ROOT + path.name, timeout=300) as response:
        with path.with_suffix(".parquet.part").open("wb") as sink:
            shutil.copyfileobj(response, sink)
    path.with_suffix(".parquet.part").replace(path)


def load_esci(root: str | Path, *, locale: str = "us", seed: int = 42) -> RetrievalBundle:
    """Rank only judged products from ESCI small-version US Task 1."""
    try:
        import pandas as pd
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("ESCI requires: uv sync --extra benchmark") from exc
    path = Path(root)
    examples_path = path / "shopping_queries_dataset_examples.parquet"
    products_path = path / "shopping_queries_dataset_products.parquet"
    _download_esci(examples_path)
    _download_esci(products_path)
    rows = pd.read_parquet(examples_path, columns=["query", "query_id", "product_id", "product_locale", "esci_label", "small_version", "split"])
    rows = rows[(rows.small_version == 1) & (rows.product_locale == locale)].copy()
    if rows.empty:
        raise ValueError(f"No ESCI Task 1 rows for locale={locale}")
    train_ids = sorted(set(rows.loc[rows.split == "train", "query_id"].astype(str)))
    random.Random(seed).shuffle(train_ids)
    val_ids = set(train_ids[:max(1, int(0.15 * len(train_ids)))])
    product_ids = set(rows.product_id.astype(str))
    titles: dict[str, str] = {}
    for batch in pq.ParquetFile(products_path).iter_batches(batch_size=30000, columns=["product_id", "product_title", "product_locale"]):
        for pid, title, plocale in zip(*(col.to_pylist() for col in batch.columns), strict=True):
            if plocale == locale and str(pid) in product_ids:
                titles[str(pid)] = clean_text(title)
    bundle = RetrievalBundle(meta={"dataset": f"Amazon ESCI Task 1 {locale}", "protocol": "judged_candidates", "esci_grade_map": ESCI_GRADE, "exact_threshold": 3.0})
    bundle.corpus = {f"{locale}:{pid}": title for pid, title in titles.items() if title}
    skipped = 0
    for row in rows.itertuples(index=False):
        did = f"{locale}:{row.product_id}"
        if did not in bundle.corpus:
            skipped += 1
            continue
        qid = f"esci:{row.query_id}"
        bundle.queries[qid] = clean_text(row.query)
        split = "test" if row.split == "test" else "val" if str(row.query_id) in val_ids else "train"
        grade = ESCI_GRADE[str(row.esci_label)]
        labels = bundle.qrels[split].setdefault(qid, {})
        labels[did] = max(labels.get(did, -1.0), grade)
        bundle.esci_labels[(qid, did)] = str(row.esci_label)
        bundle.pools.setdefault(qid, []).append(did)
    bundle.pools = {qid: list(dict.fromkeys(pool)) for qid, pool in bundle.pools.items()}
    bundle.train_docs = {did for judgments in bundle.qrels["train"].values() for did in judgments}
    bundle.meta["data_quality"] = {"excluded_unusable_product_titles": skipped}
    return bundle
