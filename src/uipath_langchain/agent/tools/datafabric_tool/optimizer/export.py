"""Export and load optimized prompt artefacts."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Default location next to the datafabric_tool package
_DEFAULT_PATH = Path(__file__).resolve().parent.parent / "optimized_prompts.json"


def export_optimized_prompts(
    optimized_instruction: str,
    few_shot_examples: list[dict[str, str]],
    use_reasoning: bool,
    optimizer_name: str,
    val_accuracy: float,
    output_path: Path | None = None,
) -> Path:
    """Write optimized prompts to a JSON file.

    Args:
        optimized_instruction: The DSPy-optimized system instruction.
        few_shot_examples: List of {question, schema_context, sql} dicts.
        use_reasoning: Whether the winning signature uses CoT reasoning.
        optimizer_name: Name of the DSPy optimizer used (e.g. "miprov2").
        val_accuracy: Validation set accuracy achieved.
        output_path: Where to write the JSON.  Defaults to
            ``datafabric_tool/optimized_prompts.json``.

    Returns:
        The resolved path where the file was written.
    """
    path = output_path or _DEFAULT_PATH
    payload: dict[str, Any] = {
        "optimized_instruction": optimized_instruction,
        "few_shot_examples": few_shot_examples,
        "use_reasoning": use_reasoning,
        "optimization_metadata": {
            "optimizer": optimizer_name,
            "val_accuracy": val_accuracy,
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    logger.info("Optimized prompts exported to %s", path)
    return path


def load_optimized_prompts(
    path: Path | None = None,
) -> dict[str, Any] | None:
    """Load optimized prompts from disk if available.

    Returns:
        Parsed JSON dict or None if the file does not exist.
    """
    path = path or _DEFAULT_PATH
    if not path.is_file():
        return None
    try:
        data: dict[str, Any] = json.loads(path.read_text())
        return data
    except (json.JSONDecodeError, OSError):
        logger.warning("Failed to load optimized prompts from %s", path)
        return None
