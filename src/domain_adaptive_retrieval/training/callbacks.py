"""Record complete checkpoint saves after Hugging Face has saved all state."""

from pathlib import Path
from typing import Any

from transformers import TrainerCallback

from domain_adaptive_retrieval.training.resume import mark_checkpoint_complete


# Transformers is optional during checks of a core/dev-only installation.
class CompleteCheckpointCallback(TrainerCallback):  # type: ignore[misc, unused-ignore]
    def on_save(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        if args.should_save:
            mark_checkpoint_complete(
                Path(args.output_dir) / f"checkpoint-{state.global_step}"
            )
        return control
