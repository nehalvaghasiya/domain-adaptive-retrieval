"""Train/resume only LegalBench-RAG reranking and collect base/tuned test results.

Run from the repository root with an NVIDIA GPU and a CUDA 12.6-compatible driver:

    uv sync --extra train --extra benchmark --extra cuda126 --group dev
    uv run --no-sync python scripts/run_legal_reranker.py --dry-run
    uv run --no-sync python scripts/run_legal_reranker.py

Rerun the last command after interruption to resume completed checkpoints/stages.
Default results: outputs/legal/reranker/evaluation/{metrics.json,comparison.md,
paired_bootstrap.json,per_query.json,predictions_base.jsonl,predictions_tuned.jsonl,
frozen_candidates.json}. Selected weights: outputs/legal/reranker/model/.
Logs and progress: outputs/legal/reranker/{run.log,job_state.json}.
Project documentation is not updated; nothing is uploaded.
"""

import argparse
from pathlib import Path

from domain_adaptive_retrieval.config.presets import REPO_ROOT
from domain_adaptive_retrieval.workflows import finetune_suite


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs",
        help="Output parent; the job writes legal/reranker/ below it",
    )
    parser.add_argument(
        "--profile",
        type=Path,
        default=REPO_ROOT / "configs/experiments/local_gpu.yaml",
        help="Use unchanged settings when resuming; choose new outputs for new settings",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Check settings and resume status without downloads, training or writes",
    )
    args = parser.parse_args(argv)
    command = [
        "--jobs",
        "legal/reranker",
        "--output-dir",
        str(args.output_dir.resolve()),
        "--profile",
        str(args.profile.resolve()),
    ]
    if args.dry_run:
        command.append("--dry-run")
    return finetune_suite.main(command)


if __name__ == "__main__":
    raise SystemExit(main())
