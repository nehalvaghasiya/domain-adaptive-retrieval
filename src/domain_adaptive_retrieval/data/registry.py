"""Explicit dataset registry; avoid hidden or magical adapter discovery."""

from domain_adaptive_retrieval.config.schema import ExperimentConfig
from domain_adaptive_retrieval.data.adapters.beir import load_beir
from domain_adaptive_retrieval.data.adapters.codesearchnet import load_codesearchnet
from domain_adaptive_retrieval.data.adapters.custom import load_custom
from domain_adaptive_retrieval.data.adapters.esci import load_esci
from domain_adaptive_retrieval.data.adapters.legalbench import download_legalbench, load_legalbench
from domain_adaptive_retrieval.data.bundle import RetrievalBundle
from domain_adaptive_retrieval.data.validation import validate_bundle


def load_domain(config: ExperimentConfig) -> RetrievalBundle:
    """Load and validate a domain dataset, downloading public data when necessary."""
    if config.domain in {"finance", "biomedical"}:
        bundle = load_beir(config.domain, config.cache_dir)
    elif config.domain == "legal":
        root = config.legal_dir or download_legalbench(config.cache_dir)
        bundle = load_legalbench(root, seed=config.seed, width=config.legal_chunk_words, overlap=config.legal_chunk_overlap)
    elif config.domain == "ecommerce":
        bundle = load_esci(config.esci_dir or config.cache_dir / "esci", locale=config.esci_locale, seed=config.seed)
    elif config.domain == "code":
        bundle = load_codesearchnet(config)
    elif config.domain == "custom":
        if config.custom_dir is None:
            raise ValueError("Custom dataset requires data.custom_dir path")
        bundle = load_custom(config.custom_dir)
    else:
        raise ValueError(f"Unsupported domain: {config.domain}")
    validate_bundle(bundle)
    return bundle
