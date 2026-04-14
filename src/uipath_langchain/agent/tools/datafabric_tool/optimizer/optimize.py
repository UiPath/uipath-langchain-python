"""CLI entry point for DSPy-based Text2SQL prompt optimization.

Usage::

    uv run python -m uipath_langchain.agent.tools.datafabric_tool.optimizer.optimize \\
        --eval-dataset path/to/eval.json \\
        --output src/.../optimized_prompts.json \\
        --model openai/gpt-4o \\
        --optimizer miprov2
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def _load_dataset(path: Path) -> list[dict[str, str]]:
    """Load eval dataset from JSON file.

    Expected format::

        [
            {
                "question": "How many customers are there?",
                "schema": "| Field | Type | ... ",
                "sql": "SELECT COUNT(id) FROM Customer LIMIT 1"
            },
            ...
        ]
    """
    data: list[dict[str, str]] = json.loads(path.read_text())
    required_keys = {"question", "schema", "sql"}
    for i, item in enumerate(data):
        missing = required_keys - item.keys()
        if missing:
            raise ValueError(f"Dataset item {i} is missing keys: {missing}")
    return data


def main(argv: list[str] | None = None) -> None:
    """Run DSPy optimization pipeline."""
    parser = argparse.ArgumentParser(description="Optimize Text2SQL prompts using DSPy")
    parser.add_argument(
        "--eval-dataset",
        type=Path,
        required=True,
        help="Path to eval JSON dataset",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for optimized prompts JSON",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openai/gpt-4o",
        help="DSPy LM model identifier (default: openai/gpt-4o)",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        choices=["bootstrap", "miprov2"],
        default="miprov2",
        help="DSPy optimizer to use (default: miprov2)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Fraction of data for training (default: 0.7)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Lazy import — dspy is an optional dependency
    try:
        import dspy
    except ImportError:
        logger.error("dspy is not installed.  Run: uv sync --extra optimize")
        sys.exit(1)

    from .export import export_optimized_prompts
    from .metrics import sql_match_metric
    from .signatures import get_signatures

    Text2SQL, Text2SQLWithReasoning = get_signatures()

    # Load dataset
    logger.info("Loading dataset from %s", args.eval_dataset)
    raw = _load_dataset(args.eval_dataset)
    examples = [
        dspy.Example(
            question=item["question"],
            schema=item["schema"],
            sql=item["sql"],
        ).with_inputs("question", "schema")
        for item in raw
    ]

    # Train/val split
    split = int(len(examples) * args.train_ratio)
    train_set = examples[:split]
    val_set = examples[split:]
    logger.info("Train: %d, Val: %d", len(train_set), len(val_set))

    # Configure DSPy LM
    lm = dspy.LM(args.model)
    dspy.configure(lm=lm)

    # Build modules for both signatures
    direct_module = dspy.ChainOfThought(Text2SQL)
    cot_module = dspy.ChainOfThought(Text2SQLWithReasoning)

    # Select optimizer
    if args.optimizer == "bootstrap":
        optimizer = dspy.BootstrapFewShot(
            metric=sql_match_metric,
            max_bootstrapped_demos=4,
            max_labeled_demos=8,
        )
    else:
        optimizer = dspy.MIPROv2(
            metric=sql_match_metric,
            auto="medium",
        )

    # Optimize both variants
    logger.info("Optimizing direct (Text2SQL)...")
    optimized_direct = optimizer.compile(
        direct_module,
        trainset=train_set,
    )

    logger.info("Optimizing CoT (Text2SQLWithReasoning)...")
    optimized_cot = optimizer.compile(
        cot_module,
        trainset=train_set,
    )

    # Evaluate both on validation set
    evaluator = dspy.Evaluate(
        devset=val_set,
        metric=sql_match_metric,
        display_progress=True,
    )
    direct_score = evaluator(optimized_direct)
    cot_score = evaluator(optimized_cot)
    logger.info("Direct score: %.3f, CoT score: %.3f", direct_score, cot_score)

    # Pick the winner
    use_reasoning = cot_score >= direct_score
    winner = optimized_cot if use_reasoning else optimized_direct
    val_accuracy = max(direct_score, cot_score)

    # Extract optimized instruction and few-shot demos
    # DSPy stores these in the module's predict attribute
    instruction = ""
    few_shots: list[dict[str, str]] = []
    for _name, param in winner.named_parameters():
        if hasattr(param, "signature") and hasattr(param.signature, "instructions"):
            instruction = param.signature.instructions
        if hasattr(param, "demos"):
            for demo in param.demos:
                shot: dict[str, str] = {}
                if hasattr(demo, "question"):
                    shot["question"] = demo.question
                if hasattr(demo, "schema"):
                    shot["schema_context"] = demo.schema
                if hasattr(demo, "sql"):
                    shot["sql"] = demo.sql
                if shot:
                    few_shots.append(shot)

    # Export
    output_path = export_optimized_prompts(
        optimized_instruction=instruction,
        few_shot_examples=few_shots,
        use_reasoning=use_reasoning,
        optimizer_name=args.optimizer,
        val_accuracy=val_accuracy,
        output_path=args.output,
    )
    logger.info("Done!  Optimized prompts written to %s", output_path)
    logger.info(
        "Winner: %s (accuracy: %.1f%%)",
        "CoT" if use_reasoning else "Direct",
        val_accuracy * 100,
    )


if __name__ == "__main__":
    main()
