import logging
import math
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd

from ..config import config

logger = logging.getLogger(__name__)

_SEMANTIC_MODEL = None

# Default WER >= threshold for the "significant error" pass/fail flag.
# 0.30 matches the lenient end of the public ASR benchmark literature
# (utterances with more than 30% of words wrong are treated as material
# failures). The threshold is recorded in every ``scores.json`` it scores so
# a future default change does not retroactively change the meaning of old
# runs; callers who care can override per-call or per-CLI invocation.
DEFAULT_SIGNIFICANT_WER_THRESHOLD = 0.30


def _get_semantic_model():
    """Lazy-load paraphrase-multilingual-MiniLM-L12-v2."""
    global _SEMANTIC_MODEL
    if _SEMANTIC_MODEL is None:
        from sentence_transformers import SentenceTransformer

        logger.debug("Loading semantic similarity model")
        _SEMANTIC_MODEL = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    return _SEMANTIC_MODEL


def calculate_cer_wer(reference: str, hypothesis: str) -> Tuple[Optional[float], Optional[float]]:
    """Calculate CER and WER."""
    if not reference or not reference.strip():
        return None, None

    if not hypothesis:
        hypothesis = ""

    try:
        from jiwer import cer, wer
    except ImportError:
        logger.warning("jiwer is not installed — CER/WER metrics unavailable. Install with: pip install jiwer")
        return None, None

    try:
        return cer(reference, hypothesis), wer(reference, hypothesis)
    except Exception:
        logger.warning("CER/WER calculation failed for input pair", exc_info=True)
        return None, None


def clamp_similarity(similarity: float) -> float:
    """Clamp a cosine similarity to ``[0, 1]`` — the published metric range.

    Raw cosine similarity legitimately ranges ``[-1, 1]``, but every artifact
    (per-utterance CSV, ``scores.json`` means, POSEIDON, the public
    leaderboard) reports "semantic similarity — higher is better" on
    ``[0, 1]``. Clamping once at the source keeps
    ``semantic_similarity_mean`` and ``poseidon_score_mean`` computed over
    the same value range (issue #107: similarity used to be clamped inside
    the POSEIDON formula but stored and averaged raw, so the mean could go
    negative while POSEIDON could not). A negative cosine — reference and
    hypothesis semantically unrelated — reads as 0.0: no semantic overlap.
    """
    return min(max(similarity, 0.0), 1.0)


def compute_semantic_similarity(reference: str, hypothesis: str) -> Optional[float]:
    """Compute cosine similarity, clamped to ``[0, 1]`` (see :func:`clamp_similarity`)."""
    if not reference or not reference.strip():
        return None
    if not hypothesis:
        hypothesis = ""
    try:
        model = _get_semantic_model()
        from sentence_transformers import util

        e1, e2 = model.encode([reference, hypothesis], convert_to_tensor=False, show_progress_bar=False)
        sim = float(util.cos_sim(e1[None], e2[None])[0][0])
        return clamp_similarity(sim)
    except ImportError:
        logger.warning(
            "sentence-transformers is not installed — semantic similarity unavailable. "
            'Install with: pip install "psdn-sonar[ml]"'
        )
        return None
    except Exception:
        logger.warning("Semantic similarity calculation failed", exc_info=True)
        return None


def calculate_poseidon_score(
    cer: float,
    wer: float,
    similarity: float,
    *,
    wer_weight: Optional[float] = None,
    cer_weight: Optional[float] = None,
    semantic_weight: Optional[float] = None,
) -> float:
    """
    Calculate POSEIDON score.

    Formula: POSEIDON = wer_weight * (1 - WER) + cer_weight * (1 - CER) + semantic_weight * Similarity

    When called without explicit weights, the global defaults from config are
    used (WER=0.35, CER=0.20, Similarity=0.45, configurable via env vars).
    Supply all three weights to override; they must sum to 1.0.

    Raises:
        TypeError: If ``cer``, ``wer``, or ``similarity`` is ``None``. The
            upstream helpers (:func:`calculate_cer_wer`,
            :func:`compute_semantic_similarity`) return ``None`` when a
            metric could not be computed; the error message names the likely
            cause and fix instead of failing on an opaque comparison.
        ValueError: If explicit weights do not sum to 1.0.
    """
    if similarity is None:
        raise TypeError(
            "calculate_poseidon_score: similarity is None. "
            "compute_semantic_similarity() returns None when sentence-transformers "
            'is not installed (install with: pip install "psdn-sonar[ml]") or the '
            "reference is empty. Check the similarity for None before computing POSEIDON."
        )
    if cer is None or wer is None:
        raise TypeError(
            "calculate_poseidon_score: cer/wer is None. "
            "calculate_cer_wer() returns (None, None) when the reference is empty or "
            "jiwer is unavailable. Check CER/WER for None before computing POSEIDON."
        )

    w_wer = wer_weight if wer_weight is not None else config.wer_weight
    w_cer = cer_weight if cer_weight is not None else config.cer_weight
    w_sem = semantic_weight if semantic_weight is not None else config.semantic_weight

    if wer_weight is not None or cer_weight is not None or semantic_weight is not None:
        total = w_wer + w_cer + w_sem
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"POSEIDON weights must sum to 1.0, got {total}")

    cer_capped = min(max(cer, 0.0), 1.0)
    wer_capped = min(max(wer, 0.0), 1.0)
    similarity_capped = min(max(similarity, 0.0), 1.0)

    score = w_wer * (1 - wer_capped) + w_cer * (1 - cer_capped) + w_sem * similarity_capped

    return min(max(score, 0.0), 1.0)


# Column layouts tried in order: multi-speaker conversation metrics, their
# non-conversation variants, single-speaker plain names, legacy uppercase.
_POSEIDON_COLUMN_LAYOUTS = [
    ("cer_conv", "wer_conv", "semantic_similarity_conv"),
    ("cer_non", "wer_non", "semantic_similarity_non"),
    ("cer", "wer", "semantic_similarity"),
    ("CER", "WER", "semantic_similarity"),
]


def ensure_poseidon_score(df: pd.DataFrame) -> pd.DataFrame:
    """Return *df* with a ``poseidon_score`` column, deriving it if missing.

    The score is computed per row from the first CER/WER/similarity column
    layout present, and only for rows where all three metrics are present —
    the project-wide missing-value convention (issue #107): an uncomputable
    metric is excluded from aggregates, never substituted with a best- or
    worst-case value. Rows with any missing metric get ``NaN`` (which pandas
    aggregations skip); when the similarity column is absent entirely, the
    whole derived column is ``NaN``, matching the canonical pipeline where
    POSEIDON is undefined unless semantic similarity was computed. Returned
    unchanged when the column already exists or no known layout is present.
    """
    if "poseidon_score" in df.columns:
        return df

    for cer_col, wer_col, sem_col in _POSEIDON_COLUMN_LAYOUTS:
        if cer_col in df.columns and wer_col in df.columns:
            break
    else:
        return df

    df = df.copy()
    if sem_col not in df.columns:
        df["poseidon_score"] = float("nan")
        return df

    def _row_score(row: pd.Series) -> float:
        if pd.isna(row[wer_col]) or pd.isna(row[cer_col]) or pd.isna(row[sem_col]):
            return float("nan")
        return calculate_poseidon_score(float(row[cer_col]), float(row[wer_col]), float(row[sem_col]))

    df["poseidon_score"] = df.apply(_row_score, axis=1)
    return df


def _coerce_finite_wer(value: object) -> Optional[float]:
    """Return ``value`` as a finite ``float`` or ``None`` if unusable.

    ``None`` / ``NaN`` / ``inf`` are treated as missing rather than as 0.0
    so that an utterance with no measurable WER (e.g. empty reference,
    failed transcription) does not silently get counted as a "pass" by
    :func:`is_significant_wer`.
    """
    if value is None or not isinstance(value, (int, float, str)):
        return None
    try:
        coerced = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(coerced):
        return None
    return coerced


def is_significant_wer(
    wer: Optional[float],
    threshold: float = DEFAULT_SIGNIFICANT_WER_THRESHOLD,
) -> Optional[bool]:
    """Return whether ``wer`` is at or above ``threshold``.

    The comparison is ``>=`` (not ``>``): the literature treats
    "exactly at threshold" as the borderline failure case we want to
    surface, not hide. ``None`` / ``NaN`` / ``inf`` inputs propagate as
    ``None`` so a missing measurement is never counted as either a pass
    or a fail.

    Args:
        wer: Per-utterance WER (typically in ``[0, 1]`` but unbounded
            above is allowed; ``jiwer.wer`` can exceed 1.0).
        threshold: Fail-iff-``wer >= threshold``. Must be non-negative.

    Returns:
        ``True`` if ``wer`` is a "significant error", ``False`` if it is
        below threshold, ``None`` if ``wer`` is missing.
    """
    if threshold < 0:
        raise ValueError(f"significantWer threshold must be >= 0, got {threshold}")
    coerced = _coerce_finite_wer(wer)
    if coerced is None:
        return None
    return coerced >= threshold


def significant_wer_rate(
    wer_values: Iterable[Optional[float]],
    threshold: float = DEFAULT_SIGNIFICANT_WER_THRESHOLD,
) -> Optional[float]:
    """Fraction of utterances whose WER is at or above ``threshold``.

    Missing / non-finite WER values are excluded from *both* the
    numerator and the denominator. The rate is reported only over
    utterances that actually have a measurable WER, so a run with mixed
    successful / failed transcriptions does not artificially deflate the
    rate by counting failures as passes.

    Args:
        wer_values: Per-utterance WERs. ``None`` / ``NaN`` / ``inf``
            entries are skipped.
        threshold: Same semantics as :func:`is_significant_wer`.

    Returns:
        ``float`` in ``[0, 1]``, or ``None`` if no utterance had a
        finite WER (so callers can distinguish "0% significant errors"
        from "no measurable inputs at all").
    """
    if threshold < 0:
        raise ValueError(f"significantWer threshold must be >= 0, got {threshold}")
    finite = [coerced for raw in wer_values if (coerced := _coerce_finite_wer(raw)) is not None]
    if not finite:
        return None
    return sum(1 for value in finite if value >= threshold) / len(finite)


def compute_latency_summary(latency_values: list[float]) -> Dict[str, Optional[float]]:
    """Compute avg, median, and p95 latency from a list of per-utterance values.

    Returns dict with keys avg_latency_s, median_latency_s, p95_latency_s.
    All values are None if the input list is empty.
    """
    if not latency_values:
        return {"avg_latency_s": None, "median_latency_s": None, "p95_latency_s": None}
    arr = np.asarray(latency_values, dtype=float)
    return {
        "avg_latency_s": round(float(np.mean(arr)), 4),
        "median_latency_s": round(float(np.median(arr)), 4),
        "p95_latency_s": round(float(np.percentile(arr, 95)), 4),
    }


def _finite_floats(values: Iterable[Optional[float]]) -> list:
    """Filter an iterable down to finite floats (drop None / NaN / inf)."""
    out = []
    for v in values:
        coerced = _coerce_finite_wer(v)  # generic finite-float coercion (name is historical)
        if coerced is not None:
            out.append(coerced)
    return out


def compute_protocol_latency_summary(
    complete_values: Iterable[Optional[float]],
    ttft_values: Optional[Iterable[Optional[float]]] = None,
) -> Dict[str, Optional[float]]:
    """Protocol-aware latency summary: p50 / p95 for complete and TTFT.

    Unlike :func:`compute_latency_summary` (which treats latency as a single
    scalar), this splits the two operationally distinct numbers a streaming
    benchmark needs to compare against batch providers:

    - ``complete_p50`` / ``complete_p95``: percentiles of total wall-clock
      latency (defined for batch and streaming).
    - ``ttft_p50`` / ``ttft_p95``: percentiles of time-to-first-token. These
      are ``None`` when no streaming adapter contributed a TTFT value (e.g. a
      batch-only adapter set), so a missing TTFT reads as "not applicable"
      rather than a misleading ``0``.

    Non-finite / ``None`` entries are dropped from each series independently,
    so a partially-instrumented run still reports sensible percentiles.
    """
    complete = _finite_floats(complete_values)
    ttft = _finite_floats(ttft_values) if ttft_values is not None else []

    def _pct(arr: list, q: float) -> Optional[float]:
        if not arr:
            return None
        return round(float(np.percentile(np.asarray(arr, dtype=float), q)), 4)

    return {
        "complete_p50": _pct(complete, 50),
        "complete_p95": _pct(complete, 95),
        "ttft_p50": _pct(ttft, 50),
        "ttft_p95": _pct(ttft, 95),
    }
