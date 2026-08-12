"""Complete workflow: HuggingFace dataset + HuggingFace model -> full report.

Two steps, each via subprocess so the example mirrors what you would run
by hand:

1. Convert a HF dataset to psdn-sonar TSV (``huggingface_dataset_loader.py``).
2. Evaluate a HF model on it with ``psdn-sonar single --report``, which also
   generates the comparison plots and markdown report.
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parent.parent


def run_complete_workflow(
    dataset_name: str,
    hf_model: str,
    language: str,
    output_dir: Path,
    config: str = None,
    split: str = "test",
    max_samples: int = 0,
) -> dict:
    """Run dataset conversion then evaluation + report generation.

    Returns a dict with the dataset TSV, results CSV, and analysis directory.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "HuggingFace workflow: dataset=%s (%s) model=%s language=%s output=%s",
        dataset_name,
        config or "default",
        hf_model,
        language,
        output_dir,
    )

    logger.info("Step 1/2: loading dataset from HuggingFace Hub")
    dataset_tsv = output_dir / "dataset.tsv"
    cmd = [
        sys.executable,
        "examples/huggingface_dataset_loader.py",
        "--dataset",
        dataset_name,
        "--split",
        split,
        "--output",
        str(dataset_tsv),
    ]
    if config:
        cmd.extend(["--config", config])
    if max_samples > 0:
        cmd.extend(["--max-samples", str(max_samples)])

    result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("Dataset loading failed:\n%s", result.stderr)
        sys.exit(1)
    logger.info("Dataset loaded")

    logger.info("Step 2/2: running ASR evaluation with report generation")
    eval_output_dir = output_dir / "evaluation"
    cmd = [
        "psdn-sonar",
        "single",
        "--input",
        str(dataset_tsv),
        "--hf-model",
        hf_model,
        "--language",
        language,
        "--output",
        str(eval_output_dir),
        "--report",
    ]
    if max_samples > 0:
        cmd.extend(["--max-samples", str(max_samples)])

    result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("Evaluation failed:\n%s", result.stderr)
        sys.exit(1)
    logger.info("Evaluation completed")

    results_csv = next(
        (f for f in eval_output_dir.glob("*.csv") if f.stem.startswith("asr_detailed_")),
        None,
    )
    if not results_csv:
        logger.error("Could not find results CSV under %s", eval_output_dir)
        sys.exit(1)

    analysis_dir = eval_output_dir / "analysis"

    logger.info("Workflow finished")
    logger.info("Dataset: %s", dataset_tsv)
    logger.info("Results: %s", results_csv)
    logger.info("Analysis: %s/", analysis_dir)

    return {"dataset_tsv": dataset_tsv, "results_csv": results_csv, "analysis_dir": analysis_dir}


def main():
    parser = argparse.ArgumentParser(
        description="Complete workflow: HuggingFace dataset + model -> full psdn-sonar report",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Korean FLEURS with Whisper
  python examples/huggingface_complete_workflow.py \\
      --dataset "google/fleurs" \\
      --config ko_kr \\
      --hf-model "openai/whisper-small" \\
      --language ko \\
      --max-samples 50 \\
      --output-dir results/korean-eval

  # Bengali FLEURS with IndicWav2Vec
  python examples/huggingface_complete_workflow.py \\
      --dataset "google/fleurs" \\
      --config bn_in \\
      --hf-model "ai4bharat/indicwav2vec-bengali" \\
      --language bn \\
      --output-dir results/bengali-eval
        """,
    )

    parser.add_argument("--dataset", required=True, help='HuggingFace dataset ID (e.g., "google/fleurs")')
    parser.add_argument("--hf-model", required=True, help='HuggingFace model ID (e.g., "openai/whisper-small")')
    parser.add_argument("--language", required=True, choices=["bn", "hi", "ko", "en"], help="Language code")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for all results")
    parser.add_argument("--config", type=str, help="Dataset config (e.g., bn_in, ko_kr)")
    parser.add_argument("--split", default="test", help="Dataset split (default: test)")
    parser.add_argument("--max-samples", type=int, default=0, help="Maximum samples to evaluate (0 = all)")

    args = parser.parse_args()

    run_complete_workflow(
        dataset_name=args.dataset,
        hf_model=args.hf_model,
        language=args.language,
        output_dir=args.output_dir,
        config=args.config,
        split=args.split,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
