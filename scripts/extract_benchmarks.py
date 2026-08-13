#!/usr/bin/env python3
"""Aggregate raw evaluation CSVs into the public benchmarks summary CSV.

Usage:
    python scripts/extract_benchmarks.py --language bengali
    python scripts/extract_benchmarks.py --input-dir path/to/raw-evaluations --output out.csv
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from psdn_sonar.utils.metrics import ensure_poseidon_score

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
BENCHMARKS_ROOT = REPO_ROOT / "psdn_sonar" / "benchmarks"

METRICS = ["cer_conv", "wer_conv", "semantic_similarity_conv", "poseidon_score"]


def summarize_csv(csv_path: Path, model: str, dataset: str) -> dict:
    """Aggregate one model x dataset results CSV into a mean/std summary row.

    Returns an empty dict when the file is unreadable or has no known metrics.
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        logger.warning("Skipping unreadable CSV %s: %s", csv_path, e)
        return {}

    df = ensure_poseidon_score(df)
    row: dict = {"model": model, "dataset": dataset, "n_samples": len(df)}

    found = False
    for metric in METRICS:
        if metric not in df.columns:
            continue
        values = pd.to_numeric(df[metric], errors="coerce").dropna()
        if values.empty:
            continue
        row[f"{metric}_mean"] = round(float(values.mean()), 4)
        row[f"{metric}_std"] = round(float(values.std()), 4) if len(values) > 1 else 0.0
        found = True

    return row if found else {}


def extract_benchmarks(input_dir: Path) -> pd.DataFrame:
    """Scan ``<input_dir>/<model>/<dataset>.csv`` and build the summary table."""
    rows = []
    for csv_path in sorted(input_dir.glob("*/*.csv")):
        row = summarize_csv(csv_path, model=csv_path.parent.name, dataset=csv_path.stem)
        if row:
            rows.append(row)
            logger.info("%s / %s: %d samples", row["model"], row["dataset"], row["n_samples"])
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Aggregate raw evaluations into a public benchmarks CSV")
    parser.add_argument("--language", help="Use psdn_sonar/benchmarks/<language>/raw-evaluations as input")
    parser.add_argument("--input-dir", type=Path, help="Explicit raw-evaluations directory (overrides --language)")
    parser.add_argument(
        "--output",
        type=Path,
        default=BENCHMARKS_ROOT / "public_benchmarks.csv",
        help="Output CSV path (default: psdn_sonar/benchmarks/public_benchmarks.csv)",
    )
    args = parser.parse_args()

    if args.input_dir:
        input_dir = args.input_dir
    elif args.language:
        input_dir = BENCHMARKS_ROOT / args.language.lower() / "raw-evaluations"
    else:
        parser.error("provide --language or --input-dir")

    if not input_dir.is_dir():
        logger.error("Input directory not found: %s", input_dir)
        sys.exit(1)

    summary = extract_benchmarks(input_dir)
    if summary.empty:
        logger.error("No benchmark data extracted from %s", input_dir)
        sys.exit(1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output, index=False)
    logger.info("Saved %d model-dataset rows to %s", len(summary), args.output)


if __name__ == "__main__":
    main()
