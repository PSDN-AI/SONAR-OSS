#!/usr/bin/env python3
"""Compute the unweighted macro-mean leaderboard across locales.

Averages per-locale aggregate metrics with one vote per locale, so large
datasets do not drown out small ones. Input is either the CSV produced by
``scripts/extract_benchmarks.py`` or a ``{locale: {model: {metric: value}}}``
JSON file; output matches :func:`psdn_sonar.aggregators.macro_mean_per_model`.

Usage:
    python scripts/build_macro_summary.py
    python scripts/build_macro_summary.py --input benchmarks.csv --output macro.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from psdn_sonar.aggregators import macro_mean_per_model

logger = logging.getLogger(__name__)

DEFAULT_INPUT = Path("psdn_sonar/benchmarks/public_benchmarks.csv")
DEFAULT_METRICS = (
    "cer_conv_mean",
    "wer_conv_mean",
    "semantic_similarity_conv_mean",
    "poseidon_score_mean",
)


def _load_csv(
    path: Path,
    metric_cols: list[str],
    allow_duplicates: bool = False,
) -> dict[str, dict[str, dict[str, float]]]:
    """Read a benchmarks CSV into ``{locale: {model: {metric: value}}}``.

    Duplicate ``(dataset, model)`` rows abort with ``SystemExit`` unless
    *allow_duplicates* opts into last-row-wins with a warning, since the
    canonical extract is unique per pair and a duplicate signals corruption.
    """
    df = pd.read_csv(path)
    missing = {"model", "dataset", *metric_cols} - set(df.columns)
    if missing:
        raise SystemExit(
            f"input {path} is missing required columns: {sorted(missing)}; available: {sorted(df.columns)}"
        )

    payload: dict[str, dict[str, dict[str, float]]] = {}
    overwritten: list[tuple[str, str]] = []
    for row in df.to_dict(orient="records"):
        locale = str(row["dataset"])
        model = str(row["model"])
        bucket = payload.setdefault(locale, {})
        if model in bucket:
            overwritten.append((locale, model))
        bucket[model] = {col: row[col] for col in metric_cols}

    if overwritten:
        duplicates = sorted(set(overwritten))
        if not allow_duplicates:
            raise SystemExit(
                f"input {path} has duplicate (dataset, model) rows: {duplicates}; "
                "deduplicate the source CSV or pass --allow-duplicates for last-row-wins."
            )
        logger.warning("duplicate (dataset, model) rows in %s; last-row-wins applied for: %s", path, duplicates)

    return payload


def _load_json(path: Path) -> dict[str, dict[str, dict[str, Any]]]:
    """Read a pre-aggregated ``{locale: {model: {metric: value}}}`` JSON file."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise SystemExit(f"{path}: expected top-level JSON object {{locale: {{model: ...}}}}, got {type(raw).__name__}")
    return raw


def _load_input(
    path: Path,
    metric_cols: list[str],
    *,
    allow_duplicates: bool = False,
) -> dict[str, dict[str, dict[str, Any]]]:
    if not path.exists():
        raise SystemExit(f"input file not found: {path}")
    if path.suffix.lower() == ".json":
        return _load_json(path)
    return _load_csv(path, metric_cols, allow_duplicates=allow_duplicates)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--input", type=Path, default=DEFAULT_INPUT, help=f"CSV or JSON input. Default: {DEFAULT_INPUT}"
    )
    parser.add_argument("--output", type=Path, default=None, help="Write result JSON here instead of stdout.")
    parser.add_argument(
        "--metric",
        action="append",
        default=None,
        help=f"Metric column to include (CSV mode). May be repeated. Default: {list(DEFAULT_METRICS)}.",
    )
    parser.add_argument(
        "--expected-locale",
        action="append",
        default=None,
        help="Locale that should be present for every model (warn-only). May be repeated.",
    )
    parser.add_argument("--indent", type=int, default=2, help="JSON indent for output.")
    parser.add_argument(
        "--allow-duplicates",
        action="store_true",
        help="Tolerate duplicate (dataset, model) CSV rows; last-row-wins with a warning instead of erroring.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = _parse_args(argv)
    metric_cols = list(args.metric) if args.metric else list(DEFAULT_METRICS)

    payload = _load_input(args.input, metric_cols, allow_duplicates=args.allow_duplicates)
    summary = macro_mean_per_model(payload, expected_locales=args.expected_locale)

    text = json.dumps(summary, indent=args.indent, ensure_ascii=False) + "\n"
    if args.output is None:
        sys.stdout.write(text)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        logger.info("wrote macro summary for %d models to %s", len(summary), args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
