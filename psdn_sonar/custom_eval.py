"""Custom language evaluation: bring your own HuggingFace model + dataset.

Evaluates the user's HF model(s) plus any configured hosted API models
(Whisper, ElevenLabs, AssemblyAI) on a local TSV or HuggingFace dataset,
then generates a combined report with comparison plots.
"""

import csv
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, cast

if TYPE_CHECKING:
    from datasets import Dataset

import yaml

logger = logging.getLogger(__name__)

_API_KEY_MAP = {
    "whisper_api": "OPENAI_API_KEY",
    "assemblyai_api": "ASSEMBLYAI_API_KEY",
    "elevenlabs_api": "ELEVENLABS_API_KEY",
}


class CustomEvalConfig:
    """Parsed representation of a custom evaluation YAML config."""

    def __init__(self, config_path: str):
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        lang = raw.get("language", {})
        self.language_code = lang.get("code", "custom")
        self.language_name = lang.get("name", self.language_code.title())

        models_raw = raw.get("models", [])
        if not models_raw:
            raise ValueError("Config must specify at least one model under 'models'")
        self.models: List[Dict] = [{"hf_model_id": m} if isinstance(m, str) else m for m in models_raw]

        ds = raw.get("dataset", {})
        self.tsv_path = ds.get("tsv_path")
        self.hf_dataset_id = ds.get("hf_dataset_id")
        self.hf_subset = ds.get("hf_subset")
        self.hf_split = ds.get("hf_split", "test")
        self.text_column = ds.get("text_column", "sentence")
        self.audio_column = ds.get("audio_column", "audio")
        if not self.tsv_path and not self.hf_dataset_id:
            raise ValueError("Config must specify either dataset.tsv_path or dataset.hf_dataset_id")

        api = raw.get("api_models", {})
        self.include_api_models = api.get("enabled", True)
        self.api_models_list = api.get("include", ["whisper_api", "elevenlabs_api", "assemblyai_api"])

    def __repr__(self):
        return (
            f"CustomEvalConfig(lang={self.language_code}/{self.language_name}, "
            f"models={len(self.models)}, api={self.include_api_models}, "
            f"dataset={'HF' if self.hf_dataset_id else 'TSV'})"
        )


def prepare_dataset(config: CustomEvalConfig, output_dir: str, max_samples: int = 0) -> str:
    """Materialize the evaluation dataset as a TSV with audio_path + transcription columns.

    Local TSVs are used as-is; HuggingFace datasets are downloaded and audio
    arrays written out as WAV files. Returns the path to the ready TSV.
    """
    if config.tsv_path:
        tsv = Path(config.tsv_path)
        if not tsv.exists():
            raise FileNotFoundError(f"Dataset TSV not found: {config.tsv_path}")
        logger.info("Using local TSV: %s (%d samples)", tsv, _count_lines(tsv))
        return str(tsv)

    logger.info(
        "Loading HuggingFace dataset: %s (subset=%s, split=%s)",
        config.hf_dataset_id,
        config.hf_subset or "default",
        config.hf_split,
    )
    from datasets import load_dataset

    load_args = [config.hf_dataset_id]
    if config.hf_subset:
        load_args.append(config.hf_subset)

    try:
        # A concrete split= always yields a Dataset, but the stubs declare
        # the full DatasetDict/IterableDataset union.
        ds = cast("Dataset", load_dataset(*load_args, split=config.hf_split))
    except Exception as first_err:
        raise RuntimeError(
            f"Could not load HuggingFace dataset '{config.hf_dataset_id}' "
            f"(subset={config.hf_subset}). Original error: {first_err}\n"
            f"Options:\n"
            f"  1. If this dataset requires trust_remote_code, add it explicitly in your config\n"
            f"  2. Provide a local TSV via dataset.tsv_path in your config\n"
            f"  3. Download the dataset separately and point to the TSV"
        ) from first_err

    logger.info("Loaded %d samples", len(ds))

    out_dir = Path(output_dir) / "dataset"
    audio_dir = out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    tsv_out = out_dir / "evaluation_dataset.tsv"

    import soundfile as sf

    records = []
    limit = min(len(ds), max_samples) if max_samples > 0 else len(ds)

    for i in range(limit):
        item = ds[i]
        text = item.get(config.text_column, "")
        if not text or not str(text).strip():
            continue

        audio_data = item.get(config.audio_column)
        if isinstance(audio_data, dict) and "array" in audio_data:
            wav_path = audio_dir / f"sample_{i:05d}.wav"
            sf.write(str(wav_path), audio_data["array"], audio_data["sampling_rate"])
            records.append({"audio_path": str(wav_path.resolve()), "transcription": str(text).strip()})
        elif isinstance(audio_data, dict) and "path" in audio_data:
            records.append({"audio_path": audio_data["path"], "transcription": str(text).strip()})
        else:
            records.append({"audio_path": "", "transcription": str(text).strip()})

        if (i + 1) % 500 == 0:
            logger.info("Processed %d/%d", i + 1, limit)

    with open(tsv_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["audio_path", "transcription"], delimiter="\t")
        writer.writeheader()
        writer.writerows(records)

    logger.info("Saved %d samples to %s", len(records), tsv_out)
    return str(tsv_out)


def _create_api_model(api_name: str, language_code: str):
    """Instantiate a hosted API model with the correct language parameters."""
    from psdn_sonar.models.apis import AssemblyAIAPIModel, ElevenLabsAPIModel, WhisperAPIModel

    if api_name == "whisper_api":
        return WhisperAPIModel(language=language_code)
    if api_name == "elevenlabs_api":
        # The adapter owns the ISO 639-1 -> vendor-code conversion (#186).
        return ElevenLabsAPIModel(language=language_code)
    if api_name == "assemblyai_api":
        return AssemblyAIAPIModel(language=language_code)
    return None


def _evaluate_model_on_dataset(
    model,
    tsv_path: str,
    model_name: str,
    output_dir: str,
    max_samples: int,
    language_code: str,
) -> Optional[str]:
    """Evaluate one model on the dataset; return the results CSV path or None on failure."""
    from psdn_sonar.evaluators.single_speaker import SingleSpeakerEvaluator

    try:
        data = SingleSpeakerEvaluator.load_data(tsv_path)
        result = SingleSpeakerEvaluator.evaluate_one(
            model=model,
            data=data,
            model_name=model_name,
            max_samples=max_samples,
            compute_sem=True,
            language=language_code,
        )

        results_csv = Path(output_dir) / f"asr_detailed_{model_name}.csv"
        with open(results_csv, "w", encoding="utf-8", newline="") as f:
            fieldnames = result["results"][0].keys() if result["results"] else SingleSpeakerEvaluator._csv_fieldnames()
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(result["results"])

        logger.info("Results saved: %s", results_csv)
        return str(results_csv)
    except Exception as e:
        logger.error("Evaluation failed for %s: %s", model_name, e, exc_info=True)
        return None


def run_custom_evaluation(
    config: CustomEvalConfig,
    output_dir: str,
    max_samples: int = 0,
    generate_report: bool = True,
) -> List[Tuple[str, str]]:
    """Run the full custom evaluation pipeline.

    Prepares the dataset, evaluates the user's HF model(s) and any available
    API models, and optionally generates a combined report.

    Returns:
        List of ``(model_name, results_csv_path)`` for models that succeeded.
    """
    from psdn_sonar.config import load_env
    from psdn_sonar.models.huggingface import CustomHuggingFaceModel

    load_env()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    available_apis = []
    if config.include_api_models:
        for api_name in config.api_models_list:
            env_var = _API_KEY_MAP.get(api_name)
            if env_var and os.getenv(env_var):
                available_apis.append(api_name)
            elif env_var:
                logger.warning("Skipping %s: %s not set", api_name, env_var)

    total_models = len(config.models) + len(available_apis)

    logger.info(
        "Custom evaluation: language=%s (%s) hf_models=%d api_models=%s dataset=%s",
        config.language_name,
        config.language_code,
        len(config.models),
        ", ".join(available_apis) or "none",
        config.hf_dataset_id or config.tsv_path,
    )

    tsv_path = prepare_dataset(config, output_dir, max_samples=max_samples)

    evaluated_models: List[Tuple[str, str]] = []
    model_idx = 0

    for model_cfg in config.models:
        hf_model_id = model_cfg["hf_model_id"]
        model_name = f"custom_{hf_model_id.replace('/', '_').replace('-', '_')}"
        model_idx += 1
        logger.info("[%d/%d] Evaluating HF model: %s", model_idx, total_models, hf_model_id)

        model = CustomHuggingFaceModel(model_id=hf_model_id, language=config.language_code)
        csv_path = _evaluate_model_on_dataset(
            model=model,
            tsv_path=tsv_path,
            model_name=model_name,
            output_dir=output_dir,
            max_samples=max_samples,
            language_code=config.language_code,
        )
        if csv_path:
            evaluated_models.append((model_name, csv_path))

    for api_name in available_apis:
        model_idx += 1
        logger.info("[%d/%d] Evaluating API model: %s", model_idx, total_models, api_name)

        try:
            api_model = _create_api_model(api_name, config.language_code)
            if api_model is None:
                logger.warning("Could not instantiate %s", api_name)
                continue

            csv_path = _evaluate_model_on_dataset(
                model=api_model,
                tsv_path=tsv_path,
                model_name=api_name,
                output_dir=output_dir,
                max_samples=max_samples,
                language_code=config.language_code,
            )
            if csv_path:
                evaluated_models.append((api_name, csv_path))
        except Exception as e:
            logger.error("API model %s failed: %s", api_name, e, exc_info=True)

    if generate_report and evaluated_models:
        _generate_custom_report(
            config=config,
            input_path=tsv_path,
            evaluated_models=evaluated_models,
            output_dir=str(output_path / "analysis"),
        )

    logger.info(
        "Custom evaluation complete: %d/%d models, results in %s", len(evaluated_models), total_models, output_dir
    )
    return evaluated_models


def _generate_custom_report(
    config: CustomEvalConfig,
    input_path: str,
    evaluated_models: List[Tuple[str, str]],
    output_dir: str,
) -> None:
    """Generate comparison plots and the combined markdown report.

    Plot steps degrade gracefully on failure; only report generation itself
    is fatal.
    """
    import pandas as pd

    from psdn_sonar.reporting.generators.report_generator import generate_report

    analysis_dir = Path(output_dir) / "custom_eval"
    diversity_dir = analysis_dir / "diversity-analysis"
    comparison_dir = analysis_dir / "model-comparison"
    audio_quality_dir = analysis_dir / "audio-quality-analysis"
    for d in (diversity_dir, comparison_dir, audio_quality_dir):
        d.mkdir(parents=True, exist_ok=True)

    lang_long = config.language_name.lower()
    dataset_name = Path(input_path).stem

    logger.info("Report pipeline: %s (%d models)", config.language_name, len(evaluated_models))

    logger.info("Step 1/4: lexical diversity analysis")
    try:
        from psdn_sonar.reporting.metrics.lexical import calculate_lexical_diversity_metrics
        from psdn_sonar.reporting.plots.lexical_diversity import (
            plot_ngram_diversity_comparison,
            plot_vocabulary_growth,
            plot_zipf_law,
        )

        df = pd.read_csv(input_path, sep="\t")
        col = "transcription" if "transcription" in df.columns else "transcript"
        transcripts = df[col].dropna().astype(str).tolist()

        diversity_results = {"User Dataset": calculate_lexical_diversity_metrics(transcripts)}
        plot_ngram_diversity_comparison(
            diversity_results,
            str(diversity_dir / "diversity_gt_comparative_diversity.png"),
            include_benchmarks=False,
            language=lang_long,
        )
        plot_vocabulary_growth(
            {"User Dataset": transcripts},
            str(diversity_dir / "diversity_gt_vocabulary_growth_curve.png"),
            include_public_benchmarks=False,
            language=lang_long,
        )
        plot_zipf_law(
            {"User Dataset": transcripts},
            str(diversity_dir / "diversity_gt_zipf_law.png"),
            include_public_benchmarks=False,
            language=lang_long,
        )
    except Exception as e:
        logger.warning("Lexical diversity analysis failed (continuing): %s", e)

    logger.info("Step 2/4: model comparison boxplots")
    try:
        from psdn_sonar.reporting.plots.cross_dataset import generate_model_comparison_plots

        generate_model_comparison_plots(
            evaluated_models=evaluated_models,
            output_dir=str(comparison_dir),
            language=lang_long,
        )
    except Exception as e:
        logger.warning("Model comparison plots failed (continuing): %s", e)

    logger.info("Step 3/4: audio quality plots")
    try:
        from psdn_sonar.reporting.plots.audio_quality import generate_audio_quality_plots

        generate_audio_quality_plots(
            results_csvs=evaluated_models,
            output_dir=str(audio_quality_dir),
            language=lang_long,
        )
    except Exception as e:
        logger.warning("Audio quality plots failed (continuing): %s", e)

    logger.info("Step 4/4: generating report")
    report_path = generate_report(
        dataset_name=dataset_name,
        dataset_path=input_path,
        output_path=str(analysis_dir / "EVAL_REPORT.md"),
        language=lang_long,
    )
    logger.info("Report saved: %s", report_path)


def _count_lines(path: Path) -> int:
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f) - 1
