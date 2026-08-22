"""Markdown evaluation report generator.

Assembles ``EVAL_REPORT.md`` from the dataset statistics and whatever plot
directories exist next to the output path (``cross-dataset-analysis/``,
``model-comparison/``, ``hard-negatives-analysis/``, ``audio-quality-analysis/``,
``latency-analysis/``, ``demographic-analysis/``, ``diversity-analysis/``).
Sections whose plots are absent are skipped.

Every claim of a public-benchmark comparison is gated on benchmark data
actually being present (issue #113). No benchmark data ships in the
repository — it is generated locally by the precompute scripts — so a stock
install must produce a report that says the plots show the user's dataset
only, never that it was "compared against public benchmarks". The
availability probes are the same loaders the plots read from, so the wording
and the pictures cannot disagree.
"""

import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

from ..loaders.benchmark_loader import (
    available_benchmark_datasets,
    load_public_benchmark_diversity,
    load_public_lexical_data,
)

logger = logging.getLogger(__name__)

_TRANSCRIPT_COLUMNS = ["transcription", "transcript", "transcription_norm"]


def _token_stats(transcripts: list, n_rows: int) -> dict:
    all_tokens = []
    for text in transcripts:
        all_tokens.extend(str(text).split())
    total_tokens = len(all_tokens)
    unique_tokens = len(set(all_tokens))
    return {
        "total_transcripts": n_rows,
        "total_tokens": total_tokens,
        "unique_tokens": unique_tokens,
        "ttr": (unique_tokens / total_tokens * 100) if total_tokens > 0 else 0,
        "avg_length": total_tokens / n_rows if n_rows > 0 else 0,
    }


def load_dataset_stats(dataset_path) -> dict:
    """Token statistics (counts, TTR, average length) for a TSV/CSV/JSONL dataset file."""
    dataset_path = Path(dataset_path)
    if not dataset_path.is_file():
        raise ValueError(f"Expected a file, got: {dataset_path}")

    suffix = dataset_path.suffix.lower()

    if suffix == ".jsonl":
        transcripts = []
        with open(dataset_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                for key in ("transcription", "transcript"):
                    if key in entry:
                        transcripts.append(str(entry[key]))
                        break
        if transcripts:
            n_rows = len(transcripts)
        else:
            with open(dataset_path, encoding="utf-8") as fh:
                n_rows = max(1, sum(1 for _ in fh))
        return _token_stats(transcripts, n_rows)

    df = pd.read_csv(str(dataset_path), sep="," if suffix == ".csv" else "\t")

    transcript_col = next((col for col in _TRANSCRIPT_COLUMNS if col in df.columns), None)
    if transcript_col is None:
        raise ValueError(f"Could not find transcript column in {dataset_path}. Columns: {df.columns.tolist()}")

    transcripts = [text for text in df[transcript_col] if pd.notna(text)]
    return _token_stats(transcripts, len(df))


def _relative_to(report_dir: Path, plot_path: Path):
    try:
        return plot_path.relative_to(report_dir)
    except ValueError:
        return plot_path


def _plot_entry(report_dir: Path, plot_path: Path, title: str, caption: str = "", level: str = "###") -> list:
    """Markdown lines embedding one plot, or [] when the file is absent."""
    if not plot_path.exists():
        return []
    lines = [f"{level} {title}", "", f"![{title}]({_relative_to(report_dir, plot_path)})", ""]
    if caption:
        lines.extend([f"*{caption}*", ""])
    return lines


def _header_section(dataset_name: str, stats: dict) -> list:
    return [
        f"# SONAR - {dataset_name} Results",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Overview",
        "",
        f"This report shows a comprehensive ASR analysis and linguistic diversity analysis of the {dataset_name} "
        f"dataset, containing {stats['total_transcripts']} audio files with transcripts totaling "
        f"{stats['total_tokens']:,} tokens. The analysis evaluates transcript diversity using multiple metrics "
        "including n-gram diversity, Type-Token Ratio, Gini coefficient, and more.",
        "",
        "## Dataset Statistics",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Total Transcripts | {stats['total_transcripts']:,} |",
        f"| Total Tokens | {stats['total_tokens']:,} |",
        f"| Unique Tokens | {stats['unique_tokens']:,} |",
        f"| Type-Token Ratio | {stats['ttr']:.2f}% |",
        f"| Average Text Length | {stats['avg_length']:.2f} tokens |",
        "",
    ]


def _setup_section(stats: dict, language: str, benchmark_datasets: list) -> list:
    """Evaluation-setup table. The coverage rows are derived from the
    benchmark data actually present, never from a hardcoded per-language
    claim (issue #113)."""
    if benchmark_datasets:
        coverage_rows = [
            f"| **Dataset Coverage** | Your dataset + public {language.title()} "
            f"benchmark evaluations ({', '.join(benchmark_datasets)}) |",
            f"| **Sample Size** | Your dataset: {stats['total_transcripts']:,} samples evaluated"
            "<br>Public benchmarks: precomputed reference evaluations shown for comparison |",
        ]
    else:
        coverage_rows = [
            f"| **Language** | {language.title()} |",
            f"| **Dataset Coverage** | Your dataset ({stats['total_transcripts']:,} samples) |",
            "| **Public Benchmarks** | None included — this installation ships no public-benchmark "
            "evaluations, so every number and plot in this report describes your dataset only |",
        ]

    return [
        "## Evaluation Setup",
        "",
        "| **Component** | **Specification** |",
        "| --- | --- |",
        *coverage_rows,
        "| **Error Metrics** | CER and WER (shared normalization pipeline) |",
        "| **Semantic Similarity** | Sentence embeddings: paraphrase-multilingual-MiniLM-L12-v2 |",
        "| **Performance Indicators** | 1. Lower CER/WER = Better performance"
        "<br>2. Higher semantic similarity = Better performance |",
        "",
    ]


def _performance_section(report_dir: Path, benchmarks_available: bool) -> list:
    """Model-comparison gallery when present, otherwise the cross-dataset gallery.

    The cross-dataset wording depends on ``benchmarks_available``: the plots
    only contain public-benchmark boxplots when benchmark evaluations were
    present on disk, so the prose must not claim a comparison the plot does
    not make (issue #113).
    """
    lines = ["## ASR Performance Analysis", ""]

    model_comparison_dir = report_dir / "model-comparison"
    cross_dataset_dir = report_dir / "cross-dataset-analysis"
    poseidon_caption = (
        "POSEIDON is a composite metric (35% WER + 20% CER + 45% Semantic). "
        "Higher scores indicate better overall ASR quality."
    )

    if model_comparison_dir.exists() and any(model_comparison_dir.glob("*.png")):
        lines.extend(
            [
                "### Model Comparison",
                "",
                "The following boxplots compare ASR model performance on your dataset.",
                "",
            ]
        )
        for filename, title, caption in [
            (
                "cer_model_comparison.png",
                "Character Error Rate (CER) by Model",
                "Lower CER indicates better character-level transcription accuracy.",
            ),
            (
                "wer_model_comparison.png",
                "Word Error Rate (WER) by Model",
                "Lower WER indicates better word-level transcription accuracy.",
            ),
            (
                "sem_model_comparison.png",
                "Semantic Similarity by Model",
                "Higher semantic similarity indicates better preservation of meaning.",
            ),
            ("poseidon_model_comparison.png", "POSEIDON Score by Model", poseidon_caption),
        ]:
            lines.extend(_plot_entry(report_dir, model_comparison_dir / filename, title, caption, level="####"))
    elif cross_dataset_dir.exists():
        if benchmarks_available:
            heading = "### Cross-Dataset Comparison"
            intro = "The following plots compare your dataset's ASR performance against public benchmarks."
            cer_extra = " Compare your dataset against public benchmarks."
            wer_extra = " See how your dataset performs relative to the public benchmarks shown."
            sem_scope = " across all datasets"
        else:
            heading = "### Performance Distributions"
            intro = (
                "The following plots show the distribution of your dataset's ASR performance. "
                "No public-benchmark evaluations are present in this installation, so the plots "
                "contain your dataset only."
            )
            cer_extra = ""
            wer_extra = ""
            sem_scope = ""
        lines.extend([heading, "", intro, ""])
        for filename, title, caption in [
            (
                "cer_by_dataset_model.png",
                "Character Error Rate (CER) by Dataset and Model",
                f"Lower CER indicates better character-level transcription accuracy.{cer_extra}",
            ),
            (
                "wer_by_dataset_model.png",
                "Word Error Rate (WER) by Dataset and Model",
                f"Lower WER indicates better word-level transcription accuracy.{wer_extra}",
            ),
            (
                "sem_by_dataset_model.png",
                "Semantic Similarity by Dataset and Model",
                f"Higher semantic similarity indicates better preservation of meaning{sem_scope}.",
            ),
            ("poseidon_by_dataset_model.png", "POSEIDON Score by Dataset and Model", poseidon_caption),
        ]:
            lines.extend(_plot_entry(report_dir, cross_dataset_dir / filename, title, caption, level="####"))

    return lines


def _hard_negatives_section(report_dir: Path) -> list:
    hard_negatives_dir = report_dir / "hard-negatives-analysis"
    if not (hard_negatives_dir.exists() and any(hard_negatives_dir.glob("*.png"))):
        return []

    lines = [
        "### Hard Negatives Analysis",
        "",
        "Hard negatives are samples with high error rates (top 25% WER/CER) within your dataset. "
        "Comparing them against the overall distribution helps identify problematic audio and "
        "transcripts.",
        "",
    ]
    for filename, title in [
        ("wer_overall_vs_hard_negatives.png", "WER: Overall vs Hard Negatives"),
        ("cer_overall_vs_hard_negatives.png", "CER: Overall vs Hard Negatives"),
    ]:
        lines.extend(_plot_entry(report_dir, hard_negatives_dir / filename, title, level="####"))
    return lines


def _quality_summary_table(summary_json_path: Path) -> list:
    if not summary_json_path.exists():
        return []
    try:
        with open(summary_json_path, encoding="utf-8") as f:
            qs = json.load(f)
        return [
            "### Data Quality Summary",
            "",
            "| Metric | Value |",
            "| --- | --- |",
            f"| **Total Utterances** | {qs.get('total_utterances', 'N/A'):,} |",
            f"| **Mean SNR** | {qs.get('mean_snr_db', 'N/A')} dB |",
            f"| **Files Clipped (>1% ratio)** | {qs.get('pct_clipped', 'N/A')}% |",
            f"| **Excessive Silence (>80%)** | {qs.get('pct_excessive_silence', 'N/A')}% |",
            f"| **Passing All Quality Checks** | {qs.get('pct_passing_all', 'N/A')}% |",
            "",
            "*At-a-glance summary of recording quality across all evaluated utterances.*",
            "",
        ]
    except Exception as exc:
        logger.debug("Could not generate audio quality summary: %s", exc)
        return []


def _audio_quality_section(report_dir: Path) -> list:
    audio_quality_dir = report_dir / "audio-quality-analysis"
    if not (audio_quality_dir.exists() and any(audio_quality_dir.glob("*.png"))):
        return []

    lines = [
        "## Audio Quality Analysis",
        "",
        "Audio quality metrics (SNR, clipping, silence) are computed per utterance and "
        "correlated with ASR error rates. Files are grouped into SNR quality tiers: "
        "**Low** (<10 dB), **Medium** (10–20 dB), **High** (>20 dB).",
        "",
    ]
    for filename, title, caption in [
        (
            "wer_by_snr_tier.png",
            "WER by SNR Quality Tier",
            "Shows how word error rate varies across audio quality tiers for each model. "
            "Higher SNR tiers generally yield lower WER.",
        ),
        (
            "quality_tier_composition.png",
            "Audio Quality Tier Composition",
            "Percentage of audio files falling into each SNR quality tier. A dataset dominated "
            "by the Low tier may indicate noisy recording conditions.",
        ),
        (
            "model_tier_heatmap.png",
            "Model × Quality Tier Heatmap",
            "Median WER for each model within each SNR tier. Reveals which models are most "
            "resilient to low-quality audio.",
        ),
        (
            "snr_vs_wer_scatter.png",
            "SNR vs. WER Scatter",
            "Per-utterance SNR plotted against WER with a regression line. A negative slope "
            "confirms that higher SNR leads to lower error rates.",
        ),
        (
            "snr_distribution.png",
            "SNR Distribution",
            "Histogram of SNR values with tier boundaries. Vertical dashed lines mark the "
            "Low/Medium (10 dB) and Medium/High (20 dB) thresholds.",
        ),
    ]:
        lines.extend(_plot_entry(report_dir, audio_quality_dir / filename, title, caption))

    lines.extend(_quality_summary_table(audio_quality_dir / "quality_summary.json"))
    return lines


def _latency_section(report_dir: Path) -> list:
    latency_dir = report_dir / "latency-analysis"
    if not (latency_dir.exists() and any(latency_dir.glob("*.png"))):
        return []

    lines = ["## Inference Latency", "", "Per-utterance inference latency measured during evaluation.", ""]
    lines.extend(
        _plot_entry(
            report_dir,
            latency_dir / "latency_boxplot.png",
            "Latency Distribution",
            "Boxplot comparing per-utterance inference latency across models. API models typically show "
            "higher but more consistent latency, while local models vary with hardware.",
        )
    )
    return lines


def _demographic_section(report_dir: Path) -> list:
    demographic_dir = report_dir / "demographic-analysis" / "demographic_plots" / "model"
    if not demographic_dir.exists():
        return []

    lines = [
        "## Demographic Analysis",
        "",
        "Performance analysis across demographic groups (Age, Gender, Region) for multi-speaker datasets.",
        "",
    ]
    for metric, metric_name in [
        ("cer_conv", "Character Error Rate (CER)"),
        ("wer_conv", "Word Error Rate (WER)"),
        ("semantic_similarity_conv", "Semantic Similarity"),
        ("poseidon_score", "Poseidon Score"),
    ]:
        lines.extend([f"### {metric_name} by Demographics", ""])
        for demo, demo_name in [("gender", "Gender"), ("age", "Age Group"), ("region", "Region")]:
            lines.extend(
                _plot_entry(
                    report_dir,
                    demographic_dir / f"{demo}_{metric}.png",
                    demo_name,
                    level="####",
                )
            )
    return lines


def _diversity_section(report_dir: Path, language: str, diversity_benchmarks_available: bool) -> list:
    """Lexical diversity gallery. The captions mention public benchmarks only
    when the diversity plots actually overlaid benchmark data — the plots and
    this text read the same loaders (issue #113)."""
    diversity_dir = report_dir / "diversity-analysis"
    lines = ["## Lexical Diversity Analysis", ""]

    if diversity_benchmarks_available:
        ngram_caption = (
            "Inference: Compare your dataset's n-gram diversity against public benchmarks. "
            "Higher diversity indicates richer linguistic variety."
        )
        growth_caption = (
            "Inference: Shows how vocabulary size increases with token count for your dataset. "
            "A healthy curve indicates consistent introduction of new vocabulary. "
            "The overlaid public-benchmark curves give a reference for the growth shape."
        )
        zipf_caption = (
            "Inference: Word frequency distribution on log-log axes. A near-linear relationship "
            "(Zipf's law) indicates natural language patterns; compare the balance between common "
            f"and rare words against the public {language.title()} benchmark curves."
        )
    else:
        ngram_caption = (
            "Inference: Your dataset's n-gram diversity by n-gram size. "
            "Higher diversity indicates richer linguistic variety."
        )
        growth_caption = (
            "Inference: Shows how vocabulary size increases with token count for your dataset. "
            "A healthy curve indicates consistent introduction of new vocabulary."
        )
        zipf_caption = (
            "Inference: Word frequency distribution on log-log axes. A near-linear relationship "
            "(Zipf's law) indicates natural language patterns and a balance between common and "
            "rare words."
        )

    for filename, title, caption in [
        ("diversity_gt_comparative_diversity.png", "N-gram Diversity", ngram_caption),
        ("diversity_gt_vocabulary_growth_curve.png", "Vocabulary Growth Curve", growth_caption),
        ("diversity_gt_zipf_law.png", "Zipf's Law Curve", zipf_caption),
    ]:
        lines.extend(_plot_entry(report_dir, diversity_dir / filename, title, caption))
    return lines


def _insights_section(language: str, diversity_benchmarks_available: bool) -> list:
    """Closing pointers for reading the report.

    These used to be canned verdicts — "your dataset shows strong diversity
    metrics compared to public benchmarks", "the dataset follows Zipf's law" —
    emitted for every run regardless of what was measured or whether any
    benchmark data existed (issue #113). No conclusion is asserted here that
    the report does not compute.
    """
    if diversity_benchmarks_available:
        diversity_insight = (
            f"   Use the diversity plots above to compare repetitive patterns and vocabulary "
            f"variety in your dataset against the public {language.title()} benchmark curves."
        )
    else:
        diversity_insight = (
            "   The diversity plots above characterize repetitive patterns and vocabulary variety in your dataset."
        )
    return [
        "## Key Insights",
        "",
        "1. **Lexical diversity:**",
        "",
        diversity_insight,
        "",
        "2. **Natural language distribution:**",
        "",
        "   Check the Zipf's Law curve: a near-linear log-log fit indicates natural language "
        "distribution patterns and balanced topic coverage.",
        "",
        "3. **ASR model performance:**",
        "",
        "   Performance can vary significantly across models; use the performance plots above "
        "to compare the models evaluated on your data.",
        "",
        "---",
        "",
        "*This report was automatically generated by SONAR.*",
        "",
    ]


def generate_report(dataset_name: str, dataset_path, output_path, language: str = "bengali") -> Path:
    """Write a markdown evaluation report and return its path.

    Parameters
    ----------
    dataset_name : display name used in the report title
    dataset_path : dataset TSV/CSV/JSONL used for the statistics section
    output_path  : where to write the report; plot sections are discovered in
                   the analysis subdirectories next to this file
    language     : language label; all benchmark wording is gated on the
                   benchmark data actually present for this language
                   (issue #113 — no benchmark data ships by default)
    """
    stats = load_dataset_stats(Path(dataset_path).resolve())

    output_file = Path(output_path).resolve()
    report_dir = output_file.parent

    # Same sources the plots read from: cross-dataset boxplots use the
    # raw-evaluations CSVs, the diversity plots use the precomputed JSONs.
    benchmark_datasets = available_benchmark_datasets(language)
    diversity_benchmarks_available = bool(load_public_benchmark_diversity(language)) or bool(
        load_public_lexical_data(language)
    )

    report = [
        *_header_section(dataset_name, stats),
        *_setup_section(stats, language, benchmark_datasets),
        *_performance_section(report_dir, benchmarks_available=bool(benchmark_datasets)),
        *_hard_negatives_section(report_dir),
        *_audio_quality_section(report_dir),
        *_latency_section(report_dir),
        *_demographic_section(report_dir),
        *_diversity_section(report_dir, language, diversity_benchmarks_available),
        *_insights_section(language, diversity_benchmarks_available),
    ]

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    return output_file
