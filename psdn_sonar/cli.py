"""SONAR command-line interface.

Single entrypoint for all evaluation workflows: single-speaker evaluation
(``single``), multi-speaker evaluation (``multi``), public dataset discovery
(``discover``), and bring-your-own-model evaluation (``custom``).
"""

import argparse
import logging
import os
import sys
import warnings
from pathlib import Path

from psdn_sonar import __version__

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)

# Third-party libraries whose INFO/DEBUG output would drown the CLI's own
# progress logs; real warnings and errors still surface.
_QUIET_LOGGERS = [
    "httpx",
    "httpcore",
    "urllib3",
    "requests",
    "transformers",
    "transformers.modeling_utils",
    "torch",
    "torchaudio",
    "accelerate",
    "safetensors",
    "sentence_transformers",
    "sentence_transformers.SentenceTransformer",
    "speechmos",
    "absl",
    "absl-py",
    "matplotlib",
    "PIL",
    "numba",
    "librosa",
    "psdn_sonar.config_loader",
    "psdn_sonar.quality_models",
]
for _lib in _QUIET_LOGGERS:
    logging.getLogger(_lib).setLevel(logging.WARNING)

warnings.filterwarnings("ignore", category=UserWarning, module=r"torch|transformers|torchaudio|librosa|safetensors")
warnings.filterwarnings("ignore", category=FutureWarning, module=r"torch|transformers|huggingface_hub|librosa")
warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"torch|transformers|pkg_resources|speechmos")

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

_DEFAULT_LANGUAGE = "bn"
# Languages with a dedicated normalizer branch in normalize_text_unified().
_NORMALIZER_LANGUAGES = ("bn", "en", "hi", "ko")
_LANGUAGE_LONG_NAMES = {"bengali": "bn", "english": "en", "hindi": "hi", "korean": "ko"}
_API_MODEL_ENV_VARS = {
    "whisper_api": ("OPENAI_API_KEY",),
    "elevenlabs_api": ("ELEVENLABS_API_KEY", "XI_API_KEY"),
    "assemblyai_api": ("ASSEMBLYAI_API_KEY",),
}


def _custom_model_name(hf_model: str) -> str:
    """Sanitize a HuggingFace repo id into a results-file stem."""
    return f"custom_{hf_model.replace('/', '_').replace('-', '_')}"


def _resolve_language(args) -> str:
    """Return the validated run language, exiting on codes the toolkit cannot score.

    The language selects the normalization branch, so a typo silently changes
    every WER/CER in the run. Unknown codes are rejected before any model is
    loaded or any score is written; recognized ISO codes without a dedicated
    normalizer proceed with an explicit fallback warning.
    """
    language = getattr(args, "language", None)
    if not language:
        logger.warning(
            "No --language specified; defaulting to 'bn' (Bengali). "
            "Pass --language en|hi|bn|ko so the correct normalizer is used."
        )
        return _DEFAULT_LANGUAGE

    from psdn_sonar.language_codes import LANG_CODE_TO_NAME

    language = language.lower()
    language = _LANGUAGE_LONG_NAMES.get(language, language)

    if language not in LANG_CODE_TO_NAME:
        logger.error(
            "Unknown --language '%s': not a recognized ISO 639-1 code, so no scores were written. "
            "Languages with dedicated normalizers: %s. "
            "Run `psdn-sonar discover --language <code> --dry-run` to check dataset support for a code.",
            language,
            ", ".join(_NORMALIZER_LANGUAGES),
        )
        sys.exit(1)

    if language not in _NORMALIZER_LANGUAGES:
        logger.warning(
            "Language '%s' (%s) has no dedicated normalizer; WER/CER will use the generic "
            "fallback normalization, which can shift metrics. Dedicated normalizers exist for: %s. "
            "For other languages, prefer `psdn-sonar custom` with a YAML config.",
            language,
            LANG_CODE_TO_NAME[language],
            ", ".join(_NORMALIZER_LANGUAGES),
        )

    return language


def _filter_unavailable_api_defaults(models: list) -> list:
    """Drop hosted API defaults when the matching credential is unset.

    Explicit ``--models elevenlabs_api`` is not filtered; only the
    language-default list is, so a first-time ``--language en`` run does
    not fail (or bill) three hosted backends.
    """
    from psdn_sonar.config import load_env

    load_env()
    kept = []
    for name in models:
        env_vars = _API_MODEL_ENV_VARS.get(name)
        if env_vars and not any(os.getenv(var) for var in env_vars):
            logger.warning(
                "Skipping default model %s: %s not set. "
                "Pass --models %s after setting the key, or use a local --models / --hf-model.",
                name,
                " or ".join(env_vars),
                name,
            )
            continue
        kept.append(name)
    return kept


def run_single_speaker(args):
    """Run single-speaker evaluation."""
    from psdn_sonar.evaluators.single_speaker import NoSamplesEvaluatedError, SingleSpeakerEvaluator
    from psdn_sonar.models.base import MissingFfmpegError

    args.language = _resolve_language(args)

    if args.hf_model:
        custom_model_name = _custom_model_name(args.hf_model)
        models = [custom_model_name]
        custom_hf_model = args.hf_model
        logger.info(f"Using custom HuggingFace model: {args.hf_model}")
        logger.info(f"Results will be saved as: {custom_model_name}")
    elif args.models:
        models = args.models
        custom_hf_model = None
    else:
        language_key = args.language.lower()
        models = SingleSpeakerEvaluator.LANGUAGE_DEFAULT_MODELS.get(language_key)
        if not models:
            supported = list(SingleSpeakerEvaluator.LANGUAGE_DEFAULT_MODELS.keys())
            logger.error(
                f"No --models specified and no default model list found for language "
                f"'{args.language}'. Either pass --models explicitly or use a supported "
                f"language: {supported}"
            )
            sys.exit(1)
        models = _filter_unavailable_api_defaults(models)
        if not models:
            logger.error(
                "No runnable default models for language '%s'. Set API keys or pass --models / --hf-model explicitly.",
                args.language,
            )
            sys.exit(1)
        custom_hf_model = None
        logger.info(
            f"No --models specified. Running {len(models)} default local/available model(s) "
            f"for language '{args.language}': {', '.join(models)}"
        )

    logger.info("Single-speaker evaluation: dataset=%s models=%s output=%s", args.input, ", ".join(models), args.output)

    try:
        from psdn_sonar.utils.metrics import DEFAULT_SIGNIFICANT_WER_THRESHOLD

        significant_wer_threshold = (
            args.significant_wer_threshold
            if getattr(args, "significant_wer_threshold", None) is not None
            else DEFAULT_SIGNIFICANT_WER_THRESHOLD
        )

        SingleSpeakerEvaluator.run_evaluation(
            tsv_path=args.input,
            output_dir=args.output,
            models=models,
            max_samples=args.max_samples,
            compute_sem=True,
            custom_hf_model=custom_hf_model,
            language=args.language,
            allow_absolute_audio_paths=not args.strict_audio_paths,
            significant_wer_threshold=significant_wer_threshold,
        )
        logger.info("Evaluation complete. Results: %s/", args.output)

        for model in models:
            results_csv = Path(args.output) / f"asr_detailed_{model}.csv"
            if results_csv.exists():
                display_aggregate_stats(str(results_csv), model)

        if args.report:
            all_csvs = [
                (m, str(Path(args.output) / f"asr_detailed_{m}.csv"))
                for m in models
                if (Path(args.output) / f"asr_detailed_{m}.csv").exists()
            ]

            for model in models:
                results_csv = Path(args.output) / f"asr_detailed_{model}.csv"
                if results_csv.exists():
                    logger.info("Report pipeline for: %s", model)
                    run_comprehensive_report(
                        input_path=args.input,
                        results_csv=str(results_csv),
                        model_name=model,
                        output_dir=str(Path(args.output) / "analysis"),
                        language=args.language,
                        all_results_csvs=all_csvs,
                    )
                else:
                    logger.warning(f"No results CSV found for {model}, skipping report")

    except (ValueError, FileNotFoundError, NoSamplesEvaluatedError, MissingFfmpegError) as e:
        # Expected input/configuration/environment failures: clean actionable
        # message, non-zero exit, no traceback noise.
        logger.error(f"Evaluation failed: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        sys.exit(1)


def run_multi_speaker(args):
    """Run multi-speaker evaluation."""
    from psdn_sonar.multispeaker_pipeline import run_multispeaker_evaluation

    args.language = _resolve_language(args)

    if args.hf_model:
        custom_model_name = _custom_model_name(args.hf_model)
        models = [custom_model_name]
        custom_hf_model = args.hf_model
        logger.info("Using custom HuggingFace model: %s", args.hf_model)
        logger.info("Results will be saved as: %s", custom_model_name)
    elif args.models:
        models = args.models if isinstance(args.models, list) else [args.models]
        custom_hf_model = None
    else:
        logger.error(
            "Either --models or --hf-model must be specified for multi-speaker mode. "
            "Language-based auto-selection is only supported for single-speaker mode."
        )
        sys.exit(1)

    try:
        logger.info(
            "Multi-speaker evaluation: manifest=%s models=%s output=%s", args.input, ", ".join(models), args.output
        )

        output_csvs = []

        for model_name in models:
            logger.info("Evaluating model: %s", model_name)
            output_csv = run_multispeaker_evaluation(
                manifest_path=args.input,
                model_name=model_name,
                output_dir=args.output,
                max_samples=args.max_samples,
                sweep=getattr(args, "sweep", False),
                method=getattr(args, "method", None),
                language=args.language,
                custom_hf_model=custom_hf_model,
            )
            output_csvs.append((model_name, output_csv))
            logger.info("Completed: %s", model_name)

        if args.demographics and args.dataset_dir:
            logger.info("Generating demographic analysis for all models")
            for model_name, output_csv in output_csvs:
                logger.info("Processing demographics for: %s", model_name)
                run_demographic_analysis(output_csv, args.dataset_dir, args.output)

        if args.report:
            logger.info("Generating comprehensive reports for all models")
            all_csvs = list(output_csvs)
            for model_name, output_csv in output_csvs:
                logger.info("Report pipeline for: %s", model_name)
                run_comprehensive_report(
                    input_path=args.input,
                    results_csv=output_csv,
                    model_name=model_name,
                    output_dir=str(Path(args.output) / "analysis"),
                    language=args.language,
                    all_results_csvs=all_csvs,
                )

        for model_name, output_csv in output_csvs:
            display_aggregate_stats(output_csv, model_name)

        logger.info("All evaluations complete.")

    except (ValueError, FileNotFoundError, RuntimeError) as e:
        # Expected input/configuration failures (unknown model, missing
        # manifest, zero clips processed, missing optional dependency):
        # clean actionable message, non-zero exit, no traceback noise.
        logger.error(f"Evaluation failed: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        sys.exit(1)


def run_demographic_analysis(results_csv, dataset_dir, output_dir):
    """Generate demographic analysis plots."""
    from psdn_sonar.analysis.demographic_analyzer import DemographicAnalyzer

    results_csv = Path(results_csv)
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir) / "demographic-analysis"

    model_name = results_csv.stem.replace("asr_eval_results_", "").replace("_manifest", "")

    try:
        DemographicAnalyzer.run_full_analysis(
            results_csv=results_csv, dataset_dir=dataset_dir, output_dir=output_dir, model_name=model_name
        )
        logger.info("Demographic analysis complete: %s/demographic_plots/%s/", output_dir, model_name)
    except Exception as e:
        logger.error(f"Demographic analysis failed: {e}", exc_info=True)
        raise


def display_aggregate_stats(results_csv, model_name):
    """Log aggregate metric statistics from a results CSV."""
    import pandas as pd

    try:
        df = pd.read_csv(results_csv)
        if df.empty:
            logger.info("No evaluated rows in %s for %s — nothing to summarize.", results_csv, model_name)
            return

        if "cer_conv" in df.columns:
            cer_col, wer_col = "cer_conv", "wer_conv"
            sem_col = "semantic_similarity_conv"
            poseidon_col = "poseidon_score_conv"
        else:
            cer_col, wer_col = "cer", "wer"
            sem_col = "semantic_similarity"
            poseidon_col = "poseidon_score"

        stats = {}
        for col, name in [
            (cer_col, "CER"),
            (wer_col, "WER"),
            (sem_col, "Semantic Similarity"),
            (poseidon_col, "POSEIDON"),
        ]:
            if col in df.columns:
                values = df[col].dropna()
                stats[name] = {
                    "mean": values.mean(),
                    "std": values.std(),
                    "min": values.min(),
                    "max": values.max(),
                    "count": len(values),
                }

        logger.info(f"\nModel: {model_name}")
        logger.info("-" * 70)
        logger.info(f"{'Metric':<20} {'Mean':<12} {'Std Dev':<12} {'Min':<12} {'Max':<12} {'Samples':<10}")
        logger.info("-" * 70)

        for metric_name, metric_stats in stats.items():
            logger.info(
                f"{metric_name:<20} "
                f"{metric_stats['mean']:<12.4f} "
                f"{metric_stats['std']:<12.4f} "
                f"{metric_stats['min']:<12.4f} "
                f"{metric_stats['max']:<12.4f} "
                f"{metric_stats['count']:<10}"
            )
        logger.info("-" * 70)

    except Exception as e:
        logger.warning(f"Could not display stats for {model_name}: {e}")


def run_comprehensive_report(input_path, results_csv, model_name, output_dir, language="bn", all_results_csvs=None):
    """Run the full analysis pipeline for one model and generate a report.

    ``all_results_csvs`` is an optional list of ``(model_name, csv_path)``
    tuples; when provided, the audio-quality and latency plots include every
    model for side-by-side comparison. Plot steps degrade gracefully on
    failure; only report generation itself is fatal.
    """
    import pandas as pd

    from psdn_sonar.language_codes import to_long_name
    from psdn_sonar.reporting.generators.report_generator import generate_report

    dataset_name = Path(input_path).stem
    analysis_dir = Path(output_dir) / model_name

    diversity_dir = analysis_dir / "diversity-analysis"
    cross_dir = analysis_dir / "cross-dataset-analysis"
    hard_neg_dir = analysis_dir / "hard-negatives-analysis"
    audio_qual_dir = analysis_dir / "audio-quality-analysis"
    for d in (diversity_dir, cross_dir, hard_neg_dir, audio_qual_dir):
        d.mkdir(parents=True, exist_ok=True)

    lang_long = to_long_name(language)

    logger.info("Comprehensive report pipeline for model: %s", model_name)

    logger.info("Step 1/6: lexical diversity analysis")
    try:
        from psdn_sonar.reporting.metrics.lexical import calculate_lexical_diversity_metrics
        from psdn_sonar.reporting.plots.lexical_diversity import (
            plot_ngram_diversity_comparison,
            plot_vocabulary_growth,
            plot_zipf_law,
        )

        if input_path.endswith(".jsonl"):
            from psdn_sonar.reporting.loaders.transcript_loader import load_transcripts_from_jsonl

            transcripts = load_transcripts_from_jsonl(Path(input_path))
        else:
            df = pd.read_csv(input_path, sep="\t")
            col = "transcription" if "transcription" in df.columns else "transcript"
            transcripts = df[col].dropna().astype(str).tolist()

        diversity_results = {"User Dataset": calculate_lexical_diversity_metrics(transcripts)}
        plot_ngram_diversity_comparison(
            diversity_results,
            str(diversity_dir / "diversity_gt_comparative_diversity.png"),
            include_benchmarks=True,
            language=lang_long,
        )
        plot_vocabulary_growth(
            {"User Dataset": transcripts},
            str(diversity_dir / "diversity_gt_vocabulary_growth_curve.png"),
            include_public_benchmarks=True,
            language=lang_long,
        )
        plot_zipf_law(
            {"User Dataset": transcripts},
            str(diversity_dir / "diversity_gt_zipf_law.png"),
            include_public_benchmarks=True,
            language=lang_long,
        )
    except Exception as e:
        logger.warning(f"Lexical diversity analysis failed (continuing): {e}")

    logger.info("Step 2/6: cross-dataset comparison")
    try:
        from psdn_sonar.reporting.plots.cross_dataset import generate_cross_dataset_plots

        generate_cross_dataset_plots(
            results_csv=results_csv,
            model_name=model_name,
            output_dir=str(cross_dir),
            language=lang_long,
        )
    except Exception as e:
        logger.warning(f"Cross-dataset comparison failed (continuing): {e}")

    logger.info("Step 3/6: hard negatives analysis")
    try:
        from psdn_sonar.reporting.plots.hard_negatives import generate_hard_negatives_comparison

        generate_hard_negatives_comparison(
            results_csv=results_csv,
            output_dir=str(hard_neg_dir),
            language=lang_long,
        )
    except Exception as e:
        logger.warning(f"Hard negatives analysis failed (continuing): {e}")

    logger.info("Step 4/6: audio quality analysis")
    try:
        from psdn_sonar.reporting.plots.audio_quality import generate_audio_quality_plots

        aq_csvs = all_results_csvs if all_results_csvs else [(model_name, results_csv)]
        generate_audio_quality_plots(
            results_csvs=aq_csvs,
            output_dir=str(audio_qual_dir),
            language=lang_long,
        )
    except Exception as e:
        logger.warning(f"Audio quality analysis failed (continuing): {e}")

    logger.info("Step 5/6: inference latency analysis")
    try:
        from psdn_sonar.reporting.plots.latency import generate_latency_plots

        latency_csvs = all_results_csvs if all_results_csvs else [(model_name, results_csv)]
        generate_latency_plots(
            results_csvs=latency_csvs,
            output_dir=str(analysis_dir / "latency-analysis"),
        )
    except Exception as e:
        logger.warning(f"Latency analysis failed (continuing): {e}")

    logger.info("Step 6/6: generating final report")
    try:
        stats_path = results_csv if input_path.endswith(".jsonl") else input_path
        report_path = generate_report(
            dataset_name=dataset_name,
            dataset_path=stats_path,
            output_path=str(analysis_dir / "EVAL_REPORT.md"),
            language=lang_long,
        )
        logger.info("Report saved: %s", report_path)
    except Exception as e:
        logger.error(f"Report generation failed: {e}", exc_info=True)
        raise


def run_discover(args):
    """Discover and prepare public datasets for a language."""
    from psdn_sonar.data.discovery import DatasetDiscovery
    from psdn_sonar.data.preparer import DatasetPreparer

    dataset_filter = None
    if args.datasets:
        dataset_filter = [d.strip() for d in args.datasets.split(",") if d.strip()]

    split_ratio = (80, 10, 10)
    if args.split_ratio:
        parts = [int(x.strip()) for x in args.split_ratio.split(",")]
        if len(parts) != 3:
            logger.error("--split-ratio must have exactly 3 comma-separated integers (e.g. 80,10,10)")
            sys.exit(1)
        split_ratio = (parts[0], parts[1], parts[2])

    logger.info("Dataset discovery for language: %s", args.language)

    try:
        available = DatasetDiscovery.discover(
            language=args.language,
            dataset_filter=dataset_filter,
            validate_remote=args.validate,
        )
    except ValueError as e:
        logger.error("%s", e)
        sys.exit(1)

    DatasetDiscovery.print_summary(available, args.language)

    if not available:
        if dataset_filter:
            from psdn_sonar.data.discovery import dataset_language_support

            hints = "; ".join(dataset_language_support(name) for name in dataset_filter)
            logger.error(
                "No datasets matched --datasets '%s' for language '%s'. "
                "The --datasets filter, not the language, excluded everything (%s). "
                "Run `psdn-sonar discover --language %s --dry-run` without --datasets to see what is available.",
                ",".join(dataset_filter),
                args.language,
                hints,
                args.language,
            )
            sys.exit(1)
        logger.warning("No datasets found for language '%s'", args.language)
        sys.exit(0)

    if args.dry_run:
        logger.info("Dry run — skipping download.")
        sys.exit(0)

    output_base = Path(args.output) if args.output else Path("data") / args.language
    prepared, failed = [], []
    for ds in available:
        logger.info("Preparing: %s (%s, config=%s)", ds.name, ds.hf_id, ds.config)
        try:
            preparer = DatasetPreparer(
                dataset=ds,
                language=args.language,
                output_dir=output_base,
                max_samples=args.max_samples,
                split_ratio=split_ratio,
                skip_audio_validation=args.skip_audio_validation,
            )
            preparer.prepare()
            prepared.append(ds.name)
        except OSError as e:
            # Environment problem (unwritable --output, disk full, network) —
            # actionable as a single clean ERROR line, like the other user-error
            # paths. No traceback: the chained mkdir(parents=True) traceback
            # names a FileNotFoundError before the real PermissionError, leading
            # the reader to the wrong diagnosis (issue #149). str(e) is the
            # exception that actually propagated, i.e. the real cause.
            logger.error(f"Failed to prepare {ds.name}: {e}")
            failed.append(ds.name)
        except Exception as e:
            # Unexpected bugs stay loud with their traceback.
            logger.error(f"Failed to prepare {ds.name}: {e}", exc_info=True)
            failed.append(ds.name)

    if not prepared:
        logger.error("All %d dataset(s) failed to prepare: %s", len(failed), ", ".join(failed))
        sys.exit(1)
    if failed:
        logger.warning("Prepared %d dataset(s); %d failed: %s", len(prepared), len(failed), ", ".join(failed))
    else:
        logger.info("All %d dataset(s) prepared. Output: %s/", len(prepared), output_base)


def run_leaderboard(args):
    """Render a leaderboard table from ``scores_*.json`` run artifacts.

    Shows only measured numbers from completed runs — nothing is derived or
    back-solved from other metrics (issue #117).
    """
    from psdn_sonar.benchmark.leaderboard import (
        build_leaderboard,
        collect_scores,
        render_leaderboard,
        rows_as_json,
        run_language,
    )

    roots = [Path(r) for r in args.runs]
    missing_roots = [str(r) for r in roots if not r.is_dir()]
    if missing_roots:
        logger.error("Not a directory: %s", ", ".join(missing_roots))
        sys.exit(1)

    loaded, skipped = collect_scores(roots)
    for message in skipped:
        logger.warning(message)

    if not loaded:
        logger.error(
            "No scores_*.json artifacts found under: %s. Every `psdn-sonar single` run "
            "writes scores_<model>.json into its --output directory; point --runs there.",
            ", ".join(str(r) for r in roots),
        )
        sys.exit(1)

    rows = build_leaderboard(loaded, language=args.language, sort=args.sort)
    if not rows:
        languages = sorted({lang for run in loaded if (lang := run_language(run.artifact)) is not None})
        logger.error(
            "Found %d run artifact(s), but none for --language '%s'. Languages present: %s.",
            len(loaded),
            args.language,
            ", ".join(languages) or "none recorded",
        )
        sys.exit(1)

    if args.json:
        print(rows_as_json(rows))
    else:
        print(render_leaderboard(rows, sort=args.sort))


def run_custom(args):
    """Run custom language evaluation from YAML config."""
    from psdn_sonar.custom_eval import CustomEvalConfig, run_custom_evaluation

    try:
        config = CustomEvalConfig(args.config)
        logger.info(f"Loaded config: {config}")

        evaluated_models = run_custom_evaluation(
            config=config,
            output_dir=args.output,
            max_samples=args.max_samples,
            generate_report=args.report,
        )

        for model_name, csv_path in evaluated_models:
            if Path(csv_path).exists():
                display_aggregate_stats(csv_path, model_name)

    except Exception as e:
        logger.error(f"Custom evaluation failed: {e}", exc_info=True)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="SONAR: Multi-Language ASR Evaluation Toolkit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single-speaker evaluation with one local model
  psdn-sonar single --input dataset.tsv --models wav2vec2_bengali --language bn --max-samples 10

  # Single-speaker — run default *local* models for a language (hosted APIs
  # are skipped unless their keys are set)
  psdn-sonar single --input dataset.tsv --language ko

  # Single-speaker with a custom HuggingFace model (pair generic multilingual
  # checkpoints with English; for bn/hi/ko prefer the registered per-language models)
  psdn-sonar single --input dataset.tsv --hf-model openai/whisper-small --language en

  # Single-speaker with report generation
  psdn-sonar single --input dataset.tsv --models wav2vec2_bengali --language bn --report

  # Custom language evaluation with YAML config
  psdn-sonar custom --config my_eval.yaml --output results/custom-eval --report

  # Multi-speaker evaluation (local model)
  psdn-sonar multi --input manifest.jsonl --models whisper_base_en --language en

  # Multi-speaker with a custom HuggingFace model
  psdn-sonar multi --input manifest.jsonl --hf-model openai/whisper-tiny --language en

  # Multi-speaker with demographic analysis
  psdn-sonar multi --input manifest.jsonl --models whisper_base_en --language en \\
      --demographics --dataset-dir /path/to/dataset

  # Discover available public datasets for a language
  psdn-sonar discover --language en --dry-run

  # Discover and download a small FLEURS subset
  psdn-sonar discover --language en --datasets fleurs --max-samples 10 --output data/en

  # Leaderboard table from completed runs' scores_*.json artifacts
  psdn-sonar leaderboard --runs results/ --language bn
        """.strip(),
    )

    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument("--verbose", "-v", action="store_true", help="Show detailed debug output")
    verbosity.add_argument("--quiet", "-q", action="store_true", help="Only show warnings and errors")

    subparsers = parser.add_subparsers(dest="mode", help="Evaluation mode")
    subparsers.required = True

    single_parser = subparsers.add_parser("single", help="Single-speaker evaluation")
    single_parser.add_argument(
        "--input", required=True, help="Path to TSV/CSV file with audio_path and transcription columns"
    )
    single_parser.add_argument(
        "--models",
        nargs="+",
        help="ASR models to evaluate (e.g., wav2vec2_bengali, elevenlabs_api). If omitted, all default models for --language are run.",
    )
    single_parser.add_argument(
        "--hf-model", type=str, help='Custom HuggingFace model ID (e.g., "openai/whisper-small"). Use this OR --models.'
    )
    single_parser.add_argument(
        "--language",
        type=str,
        default=None,
        help=(
            "Language code (bn=Bengali, ko=Korean, hi=Hindi, en=English). "
            "Defaults to bn if omitted — always pass this so the correct normalizer is used."
        ),
    )
    single_parser.add_argument(
        "--output",
        default="results/single-speaker-eval",
        help="Output directory (default: results/single-speaker-eval)",
    )
    single_parser.add_argument("--max-samples", type=int, default=0, help="Maximum samples to process (0=all)")
    single_parser.add_argument(
        "--significant-wer-threshold",
        type=float,
        default=None,
        help=(
            "WER value at and above which an utterance is flagged as a significant error "
            "(per-row significant_wer column + run-level significant_wer_rate in scores.json). "
            "Default: 0.30. The threshold actually used is recorded in scores.json."
        ),
    )
    single_parser.add_argument(
        "--strict-audio-paths",
        action="store_true",
        help=(
            "Reject absolute audio_path values in the TSV and require every path to be an "
            "existing regular file inside the TSV's directory. Relative paths that escape "
            "the TSV's directory (e.g. via ../) are always rejected, with or without this flag."
        ),
    )
    single_parser.add_argument(
        "--report", action="store_true", help="Generate comprehensive report with benchmark comparisons"
    )
    single_parser.set_defaults(func=run_single_speaker)

    custom_parser = subparsers.add_parser("custom", help="Custom language evaluation via YAML config")
    custom_parser.add_argument("--config", required=True, help="Path to YAML config file")
    custom_parser.add_argument(
        "--output", default="results/custom-eval", help="Output directory (default: results/custom-eval)"
    )
    custom_parser.add_argument("--max-samples", type=int, default=0, help="Maximum samples to process (0=all)")
    custom_parser.add_argument("--report", action="store_true", help="Generate comprehensive evaluation report")
    custom_parser.set_defaults(func=run_custom)

    multi_parser = subparsers.add_parser("multi", help="Multi-speaker evaluation")
    multi_parser.add_argument("--input", required=True, help="Path to manifest.jsonl file")
    multi_parser.add_argument(
        "--models", nargs="+", help="ASR model name(s) to evaluate (e.g., wav2vec2_bengali, elevenlabs_api)"
    )
    multi_parser.add_argument(
        "--hf-model", type=str, help='Custom HuggingFace model ID (e.g., "openai/whisper-small"). Use this OR --models.'
    )
    multi_parser.add_argument(
        "--output", default="results/multispeaker-eval", help="Output directory (default: results/multispeaker-eval)"
    )
    multi_parser.add_argument("--max-samples", type=int, default=0, help="Maximum samples to process (0=all)")
    multi_parser.add_argument(
        "--language",
        type=str,
        default=None,
        help=(
            "Language code (bn=Bengali, ko=Korean, hi=Hindi, en=English). "
            "Defaults to bn if omitted — always pass this so the correct normalizer is used."
        ),
    )
    multi_parser.add_argument(
        "--method",
        type=str,
        default=None,
        help=(
            "Preprocessing method to use for all clips "
            "(energy_trim, timestamp_trim, no_trim, pyannote_vad). "
            "If omitted, method is auto-selected per clip based on available data."
        ),
    )
    multi_parser.add_argument(
        "--sweep",
        action="store_true",
        help=(
            "Run all methods and pick the best per clip using ground truth (oracle selection). "
            "WARNING: inflates reported metrics. Use only for ablation studies."
        ),
    )
    multi_parser.add_argument("--demographics", action="store_true", help="Generate demographic analysis plots")
    multi_parser.add_argument(
        "--dataset-dir", help="Dataset directory (required if --demographics or --report is used)"
    )
    multi_parser.add_argument(
        "--report", action="store_true", help="Generate comprehensive report with benchmark comparisons"
    )
    multi_parser.set_defaults(func=run_multi_speaker)

    discover_parser = subparsers.add_parser("discover", help="Discover and prepare public datasets for a language")
    discover_parser.add_argument("--language", required=True, help="ISO 639-1 language code (e.g. ur, bn, ko, en)")
    discover_parser.add_argument(
        "--output",
        default=None,
        help="Output directory (default: data/<language>)",
    )
    discover_parser.add_argument(
        "--datasets",
        type=str,
        default=None,
        help="Comma-separated dataset filter (e.g. fleurs,voxpopuli). Default: all available.",
    )
    discover_parser.add_argument("--max-samples", type=int, default=0, help="Limit samples per split (0=all)")
    discover_parser.add_argument(
        "--split-ratio",
        type=str,
        default=None,
        help="Train,val,test ratio for datasets without predefined splits (default: 80,10,10)",
    )
    discover_parser.add_argument(
        "--skip-audio-validation",
        action="store_true",
        help="Skip SNR/clipping computation for faster processing",
    )
    discover_parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate dataset availability on HuggingFace Hub before downloading",
    )
    discover_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only show available datasets, do not download",
    )
    discover_parser.set_defaults(func=run_discover)

    leaderboard_parser = subparsers.add_parser(
        "leaderboard",
        help="Render a leaderboard table from scores_*.json run artifacts (measured numbers only)",
    )
    leaderboard_parser.add_argument(
        "--runs",
        nargs="+",
        default=["."],
        metavar="DIR",
        help="Director(y/ies) scanned recursively for scores_*.json (default: current directory)",
    )
    leaderboard_parser.add_argument(
        "--language",
        default=None,
        help="Only include runs recorded with this language code (e.g. bn)",
    )
    leaderboard_parser.add_argument(
        "--sort",
        default="poseidon",
        choices=("poseidon", "semantic", "wer", "cer"),
        help="Metric to sort by (default: poseidon; rows missing it sort last)",
    )
    leaderboard_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of a table",
    )
    leaderboard_parser.set_defaults(func=run_leaderboard)

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        for _lib in _QUIET_LOGGERS:
            logging.getLogger(_lib).setLevel(logging.DEBUG)
    elif args.quiet:
        logging.getLogger().setLevel(logging.WARNING)

    if args.mode in ("single", "multi"):
        if getattr(args, "models", None) and getattr(args, "hf_model", None):
            parser.error("Cannot use both --models and --hf-model. Choose one.")

    if args.mode == "multi":
        if args.demographics and not args.dataset_dir:
            parser.error("--demographics requires --dataset-dir")
        if args.report and not args.dataset_dir:
            parser.error("--report requires --dataset-dir for multi-speaker mode")

    if getattr(args, "input", None) and not Path(args.input).exists():
        parser.error(f"Input file not found: {args.input}")

    if getattr(args, "config", None) and not Path(args.config).exists():
        parser.error(f"Config file not found: {args.config}")

    if args.mode == "discover" and args.output is None:
        args.output = f"data/{args.language}"

    args.func(args)


def entrypoint():
    """Console-script entry: run :func:`main`, then leave via ``os._exit``.

    A torch-family native extension intermittently aborts during interpreter
    teardown (SIGABRT from an uncaught C++ ``std::system_error``,
    ``recursive_mutex lock failed``) after the exit code has already been
    decided, so the same failing command returned exit 1 on some runs and
    134 on others (issue #139). All run artifacts are written and closed
    before the run functions return, so nothing is lost by skipping
    finalization: flush logging and the std streams, then ``os._exit`` with
    the code Python had already chosen.

    Kept separate from :func:`main` so in-process callers (tests) still get
    normal ``SystemExit`` propagation.
    """
    try:
        main()
        code = 0
    except SystemExit as exc:
        if exc.code is None:
            code = 0
        elif isinstance(exc.code, int):
            code = exc.code
        else:
            # sys.exit("message") semantics: message to stderr, exit 1.
            print(exc.code, file=sys.stderr)
            code = 1
    except KeyboardInterrupt:
        code = 130
    except Exception:
        # Genuine bugs must stay loud: same traceback the interpreter would
        # have printed, same exit code — just without the fragile teardown.
        import traceback

        traceback.print_exc()
        code = 1
    logging.shutdown()
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except (OSError, ValueError):
        pass  # closed/broken streams must not turn the exit into a crash
    os._exit(code)


if __name__ == "__main__":
    entrypoint()
