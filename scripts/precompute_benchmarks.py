#!/usr/bin/env python3
"""Precompute per-language benchmark data: raw evaluations + lexical statistics.

Usage:
    python scripts/precompute_benchmarks.py --language korean --prepare fleurs zeroth
    python scripts/precompute_benchmarks.py --language english --tsv commonvoice=path/to/test.tsv
"""

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
BENCHMARKS_ROOT = REPO_ROOT / "psdn_sonar" / "benchmarks"

_API_ENV_KEYS = {
    "whisper_api": "OPENAI_API_KEY",
    "assemblyai_api": "ASSEMBLYAI_API_KEY",
    "elevenlabs_api": "ELEVENLABS_API_KEY",
}

_TRANSCRIPT_COLUMNS = ["transcription", "transcript", "text", "sentence"]

MAX_TRANSCRIPTS_FOR_STATS = 5000


def parse_tsv_specs(specs: list) -> dict:
    """Parse repeated ``name=path`` dataset specs into ``{name: Path}``."""
    datasets = {}
    for spec in specs:
        name, sep, path = spec.partition("=")
        if not sep or not name or not path:
            raise ValueError(f"--tsv expects NAME=PATH, got {spec!r}")
        datasets[name] = Path(path)
    return datasets


def prepare_hf_datasets(names: list, language_code: str, datasets_dir: Path, max_samples: int) -> dict:
    """Prepare registered HuggingFace datasets; returns ``{name: tsv_path}``."""
    from psdn_sonar.data import prepare_dataset

    datasets = {}
    for name in names:
        try:
            tsv_path = prepare_dataset(name, language_code, "test", datasets_dir / name, max_samples=max_samples)
            datasets[name] = tsv_path
            logger.info("Prepared %s -> %s", name, tsv_path)
        except Exception as e:
            logger.error("Failed to prepare %s: %s", name, e)
    return datasets


def evaluate_dataset(tsv_path: Path, dataset_name: str, model_name: str, raw_eval_dir: Path, **eval_kwargs) -> bool:
    """Evaluate one dataset with one model; write ``raw-evaluations/<model>/<dataset>.csv``."""
    from psdn_sonar.evaluators.single_speaker import SingleSpeakerEvaluator

    model_dir = raw_eval_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    results_csv = model_dir / f"{dataset_name}.csv"

    if results_csv.exists():
        logger.info("Already exists, skipping: %s", results_csv)
        return True

    run_dir = model_dir / f"_run_{dataset_name}"
    run_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Evaluating %s with %s ...", dataset_name, model_name)
    try:
        SingleSpeakerEvaluator.run_evaluation(
            tsv_path=str(tsv_path), output_dir=str(run_dir), models=[model_name], compute_sem=True, **eval_kwargs
        )
    except Exception as e:
        logger.error("Evaluation failed for %s on %s: %s", model_name, dataset_name, e)
        return False

    generated = run_dir / f"asr_detailed_{model_name}.csv"
    if not generated.exists():
        logger.warning("No CSV generated for %s on %s", model_name, dataset_name)
        return False

    generated.rename(results_csv)
    logger.info("Saved: %s", results_csv)
    return True


def load_transcripts(tsv_path: Path) -> list:
    """Read transcripts from a TSV, sampling large corpora deterministically."""
    import pandas as pd

    df = pd.read_csv(tsv_path, sep="\t")
    col = next((c for c in _TRANSCRIPT_COLUMNS if c in df.columns), None)
    if col is None:
        logger.warning("No transcript column found in %s", tsv_path)
        return []

    transcripts = [t for t in df[col].dropna().astype(str).tolist() if t.strip()]
    if len(transcripts) > MAX_TRANSCRIPTS_FOR_STATS:
        random.seed(42)
        transcripts = random.sample(transcripts, MAX_TRANSCRIPTS_FOR_STATS)
    return transcripts


def compute_lexical_stats(datasets: dict, language: str) -> None:
    """Write vocabulary-growth/Zipf, n-gram diversity, and utterance-length JSONs."""
    from psdn_sonar.reporting.metrics import (
        calculate_lexical_diversity_metrics,
        compute_utterance_length_stats,
        compute_vocabulary_growth,
        compute_zipf_law,
    )

    lexical = {}
    diversity = {}
    length_stats = {}

    for name, tsv_path in datasets.items():
        transcripts = load_transcripts(tsv_path)
        if not transcripts:
            continue

        display = name.replace("_", " ").title()
        lexical[display] = {
            "total_transcripts": len(transcripts),
            "vocabulary_growth": compute_vocabulary_growth(transcripts),
            "zipf_law": compute_zipf_law(transcripts),
        }

        metrics = calculate_lexical_diversity_metrics(transcripts)
        diversity[display] = {
            "unigram_diversity": round(metrics["unigram_diversity"], 4),
            "bigram_diversity": round(metrics["bigram_diversity"], 4),
            "trigram_diversity": round(metrics["trigram_diversity"], 4),
            "ttr": round(metrics["unigram_diversity"], 4),
            "total_samples": len(transcripts),
        }
        length_stats[display] = compute_utterance_length_stats(transcripts)
        logger.info("Lexical stats for %s: %d transcripts", display, len(transcripts))

    if length_stats:
        # Utterance length drives score comparability (issue #119): WER on a
        # 3-word utterance is quasi-binary, so datasets with very different
        # length profiles must not be read against each other.
        length_stats["_note"] = (
            "Scores are comparable within a dataset only. Utterance-length "
            "profiles differ across datasets; see docs/SCORE_INTERPRETATION.md."
        )

    BENCHMARKS_ROOT.mkdir(parents=True, exist_ok=True)
    payloads = [
        ("public_lexical_data", lexical),
        ("public_diversity_stats", diversity),
        ("public_length_stats", length_stats),
    ]
    for prefix, payload in payloads:
        out_path = BENCHMARKS_ROOT / f"{prefix}_{language}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        logger.info("Saved: %s", out_path)


def write_domain_markers(models: list, datasets: dict, raw_eval_dir: Path) -> Path:
    """Write per-cell training-overlap markers for the leaderboard to render.

    ``domain_markers.json`` maps model -> dataset -> one of "in-domain" /
    "not-declared" / "unknown", from HuggingFace model-card declarations
    recorded in psdn_sonar.models.provenance (issue #119).
    """
    from psdn_sonar.models.provenance import evaluation_domain

    markers = {model: {dataset: evaluation_domain(model, dataset) for dataset in datasets} for model in models}
    payload = {
        "note": (
            "in-domain: the model card declares this dataset as training data; "
            "not-declared: the audited card declares other corpora; "
            "unknown: no audited declaration (unaudited card or hosted API). "
            "Split hygiene cannot be verified from public metadata; see "
            "docs/SCORE_INTERPRETATION.md."
        ),
        "markers": markers,
    }
    raw_eval_dir.mkdir(parents=True, exist_ok=True)
    out_path = raw_eval_dir / "domain_markers.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    logger.info("Saved: %s", out_path)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Precompute per-language benchmark data")
    parser.add_argument("--language", required=True, help="Language name or code (e.g. english, hi, korean)")
    parser.add_argument(
        "--tsv",
        nargs="+",
        default=[],
        metavar="NAME=PATH",
        help="Local dataset TSVs (e.g. commonvoice=path/to/test.tsv)",
    )
    parser.add_argument(
        "--prepare",
        nargs="+",
        default=[],
        metavar="NAME",
        help="Registered HuggingFace datasets to download and convert (e.g. fleurs zeroth)",
    )
    parser.add_argument("--models", nargs="+", help="Models to evaluate (default: the language's registry roster)")
    parser.add_argument("--output-dir", type=Path, help="Benchmark dir (default: psdn_sonar/benchmarks/<language>)")
    parser.add_argument("--max-samples", type=int, default=0, help="Limit samples per dataset (0 = all)")
    parser.add_argument("--skip-eval", action="store_true", help="Skip model evaluation (only compute lexical stats)")
    args = parser.parse_args()

    from psdn_sonar.recipe import get_recipe

    recipe = get_recipe(args.language)
    language = recipe.language
    language_code = recipe.language_code

    models = args.models or [m["name"] for m in recipe.models]
    output_dir = args.output_dir or BENCHMARKS_ROOT / language
    datasets_dir = output_dir / "datasets"
    raw_eval_dir = output_dir / "raw-evaluations"

    logger.info("Benchmark precomputation: language=%s models=%s", language, ", ".join(models))

    datasets = parse_tsv_specs(args.tsv)
    for name, path in datasets.items():
        if not path.exists():
            logger.error("TSV not found: %s", path)
            sys.exit(1)
        logger.info("Using local TSV for %s: %s", name, path)

    datasets.update(prepare_hf_datasets(args.prepare, language_code, datasets_dir, args.max_samples))

    if not datasets:
        logger.error("No datasets. Provide --tsv NAME=PATH and/or --prepare NAME.")
        sys.exit(1)

    if not args.skip_eval:
        from psdn_sonar.models.provenance import IN_DOMAIN, evaluation_domain

        for model_name in models:
            required_key = _API_ENV_KEYS.get(model_name)
            if required_key and not os.getenv(required_key):
                logger.warning("Skipping %s — %s not set", model_name, required_key)
                continue
            for dataset_name, tsv_path in datasets.items():
                if evaluation_domain(model_name, dataset_name) == IN_DOMAIN:
                    logger.warning(
                        "%s declares %s as training data in its model card — this cell is "
                        "in-domain and will be marked as such in domain_markers.json",
                        model_name,
                        dataset_name,
                    )
                evaluate_dataset(
                    tsv_path,
                    dataset_name,
                    model_name,
                    raw_eval_dir,
                    max_samples=args.max_samples,
                    language=language_code,
                )

    write_domain_markers(models, datasets, raw_eval_dir)
    compute_lexical_stats(datasets, language)

    logger.info("Done. Raw evaluations: %s", raw_eval_dir)


if __name__ == "__main__":
    main()
