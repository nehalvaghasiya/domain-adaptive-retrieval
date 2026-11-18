"""Run or resume the seven domain fine-tuning jobs from the repository root.

uv run --extra train --extra benchmark --extra cuda126 python scripts/run_finetuning.py
"""

from domain_adaptive_retrieval.workflows.finetune_suite import main

if __name__ == "__main__":
    raise SystemExit(main())
