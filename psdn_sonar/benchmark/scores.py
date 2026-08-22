"""Build and write per-run ``scores.json`` artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field

from psdn_sonar.utils.metrics import (
    DEFAULT_SIGNIFICANT_WER_THRESHOLD,
    significant_wer_rate,
)

from .submission import SubmissionConfig

_UTTERANCE_EXPORT_FIELDS = (
    "audio_path",
    "wer",
    "cer",
    "semantic_similarity",
    "poseidon_score",
    "significant_wer",
    "inference_latency_s",
    "ttft_s",
    "complete_s",
    "error",
)


class RunLineage(BaseModel):
    """Provenance facts needed to compare two runs like-for-like.

    Added for issue #120: a published number could not be checked against a
    reproduction because neither the checkpoint revision nor the
    normalization rule set in force were recorded anywhere. All fields are
    best-effort and ``None`` when unresolvable (e.g. hosted API models have
    no local HF checkpoint).
    """

    # HuggingFace repo id actually loaded (from the resolved transformers
    # config), which may differ from the registry name in ``model_name``.
    hf_model_id: Optional[str] = None
    # Commit SHA of the checkpoint revision actually loaded. The model
    # registry pins no revisions, so this is the only record of which
    # weights produced the numbers.
    hf_revision: Optional[str] = None
    # WER normalization contract in force, e.g. ``"bn:v1+bnlp"`` — see
    # ``psdn_sonar.utils.text_processing.wer_normalization_contract``.
    normalization: Optional[str] = None


class RunAggregate(BaseModel):
    """Headline metrics for one model run."""

    # ``None`` when the run had zero successful samples: a run that scored
    # nothing must not report the best possible (0.0) error rates.
    cer_mean: Optional[float] = None
    wer_mean: Optional[float] = None
    semantic_similarity_mean: Optional[float] = None
    poseidon_score_mean: Optional[float] = None
    # Fraction of utterances whose WER >= ``significant_wer_threshold``.
    # ``None`` if no utterance had a finite WER (so callers can distinguish
    # "0% significant errors" from "no measurable inputs at all").
    significant_wer_rate: Optional[float] = None
    # Threshold actually applied for this run, recorded so a future change
    # to the default does not retroactively reinterpret old ``scores.json``
    # files. Always populated when ``significant_wer_rate`` is computed.
    significant_wer_threshold: Optional[float] = None
    latency_avg_s: Optional[float] = None
    latency_median_s: Optional[float] = None
    latency_p95_s: Optional[float] = None
    # Protocol-aware latency split (additive — the scalar fields above are
    # retained for backwards compatibility). Complete-latency percentiles are
    # populated whenever any utterance had a measured latency; TTFT
    # percentiles stay ``None`` unless a streaming adapter reported TTFT.
    complete_p50_s: Optional[float] = None
    complete_p95_s: Optional[float] = None
    ttft_p50_s: Optional[float] = None
    ttft_p95_s: Optional[float] = None
    total_samples: int
    successful: int
    failed: int
    elapsed_time_s: float


class RunScoresArtifact(BaseModel):
    """Canonical machine-readable output for one model evaluation run."""

    submission: SubmissionConfig
    model_name: str
    aggregate: RunAggregate
    lineage: Optional[RunLineage] = None
    # Run-level configuration warnings that make the numbers suspect, e.g.
    # the reference script contradicting --language (issue #148: such a run
    # used to be indistinguishable from a correct one downstream). Empty for
    # a clean run; absent in pre-#148 artifacts.
    warnings: list[str] = Field(default_factory=list)
    utterances: list[dict[str, Any]] = Field(default_factory=list)


def _slim_utterance(row: dict[str, Any]) -> dict[str, Any]:
    return {key: row.get(key) for key in _UTTERANCE_EXPORT_FIELDS}


def build_run_scores(
    submission: SubmissionConfig,
    evaluate_result: dict[str, Any],
    *,
    compute_sem: bool = False,
    significant_wer_threshold: float = DEFAULT_SIGNIFICANT_WER_THRESHOLD,
    lineage: Optional[RunLineage] = None,
    run_warnings: Optional[list[str]] = None,
) -> RunScoresArtifact:
    """Build a scores artifact from ``SingleSpeakerEvaluator.evaluate_one`` output.

    ``significant_wer_threshold`` is the WER value at and above which an
    utterance is counted as a significant error. The threshold is
    recorded on the artifact so consumers do not have to assume the
    library default. ``run_warnings`` are run-level configuration warnings
    (e.g. a reference-script/--language mismatch) preserved verbatim in the
    artifact.
    """
    summary = evaluate_result.get("summary") or {}
    results = evaluate_result.get("results") or []
    model_name = evaluate_result.get("model_name") or submission.model_snapshot

    sem_values = [r["semantic_similarity"] for r in results if r.get("semantic_similarity") is not None]
    poseidon_values = [r["poseidon_score"] for r in results if r.get("poseidon_score") is not None]

    avg_sem = sum(sem_values) / len(sem_values) if sem_values else None
    avg_poseidon = sum(poseidon_values) / len(poseidon_values) if poseidon_values else None

    sig_rate = significant_wer_rate(
        (r.get("wer") for r in results),
        threshold=significant_wer_threshold,
    )

    avg_cer = summary.get("avg_cer")
    avg_wer = summary.get("avg_wer")

    aggregate = RunAggregate(
        cer_mean=float(avg_cer) if avg_cer is not None else None,
        wer_mean=float(avg_wer) if avg_wer is not None else None,
        semantic_similarity_mean=avg_sem if compute_sem else None,
        poseidon_score_mean=avg_poseidon if compute_sem else None,
        significant_wer_rate=sig_rate,
        significant_wer_threshold=significant_wer_threshold if sig_rate is not None else None,
        latency_avg_s=summary.get("avg_latency_s"),
        latency_median_s=summary.get("median_latency_s"),
        latency_p95_s=summary.get("p95_latency_s"),
        complete_p50_s=summary.get("complete_p50_s"),
        complete_p95_s=summary.get("complete_p95_s"),
        ttft_p50_s=summary.get("ttft_p50_s"),
        ttft_p95_s=summary.get("ttft_p95_s"),
        total_samples=int(summary.get("total_samples", len(results))),
        successful=int(summary.get("successful", 0)),
        failed=int(summary.get("failed", 0)),
        elapsed_time_s=float(summary.get("elapsed_time", 0.0)),
    )

    return RunScoresArtifact(
        submission=submission,
        model_name=model_name,
        aggregate=aggregate,
        lineage=lineage,
        warnings=list(run_warnings or []),
        utterances=[_slim_utterance(row) for row in results],
    )


def write_scores_json(path: Path | str, artifact: RunScoresArtifact) -> Path:
    """Serialize ``artifact`` to JSON and return the written path."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = artifact.model_dump(mode="json")
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out


def scores_json_path(output_dir: Path | str, model_name: str) -> Path:
    """Path for ``scores.json`` alongside ``asr_detailed_<model>.csv``."""
    return Path(output_dir) / f"scores_{model_name}.json"
