"""Shared normalize-and-score helpers for single-utterance ASR evaluation.

This module is the canonical scoring primitive for core dataset/manifest flows
and for callers that already have raw reference/hypothesis strings.

Contract:
- ``score_normalized_pair`` / ``score_dual_variant`` preserve ``None`` CER/WER
  from ``calculate_cer_wer`` (no caller-specific fallbacks).
- ``score_single_variant`` normalizes text and returns raw ``Optional`` metrics
  only; callers that need legacy aggregate behavior (e.g. treating ``None`` as
  ``0.0`` for CSV rollups) must apply that policy themselves.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class UtteranceScore:
    cer: Optional[float]
    wer: Optional[float]
    semantic_similarity: Optional[float]
    poseidon_score: Optional[float]


@dataclass(frozen=True)
class DualUtteranceScore:
    """Non-conversion and Devanagari-converted metric paths (Bengali core flows)."""

    ref_norm: str
    hyp_norm_non: str
    hyp_norm_conv: str
    hyp_converted: str
    non: UtteranceScore
    conv: UtteranceScore


class UtteranceEvaluator:
    """Canonical normalize -> CER/WER/sem/POSEIDON pipeline for one utterance.

    Does not apply per-evaluator reporting policies (such as coercing missing
    metrics to zero); keep those at the call site.
    """

    @staticmethod
    def convert_hypothesis(hypothesis_raw: str, *, language: str) -> str:
        if language == "bn" and hypothesis_raw:
            from psdn_sonar.utils.text_processing import convert_devanagari_to_bengali, is_devanagari

            if is_devanagari(hypothesis_raw):
                return convert_devanagari_to_bengali(hypothesis_raw)
        return hypothesis_raw

    @staticmethod
    def normalize_reference_and_hypotheses(
        reference: str,
        hypothesis_raw: str,
        *,
        language: str,
    ) -> tuple[str, str, str, str]:
        """Return (ref_norm, hyp_norm_non, hyp_norm_conv, hyp_converted)."""
        hyp_converted = UtteranceEvaluator.convert_hypothesis(hypothesis_raw, language=language)
        from psdn_sonar.utils.text_processing import normalize_text_unified

        ref_norm = normalize_text_unified(reference, language=language)
        hyp_norm_non = normalize_text_unified(hypothesis_raw, language=language)
        hyp_norm_conv = normalize_text_unified(hyp_converted, language=language)
        return ref_norm, hyp_norm_non, hyp_norm_conv, hyp_converted

    @staticmethod
    def score_normalized_pair(
        ref_norm: str,
        hyp_norm: str,
        *,
        with_semantics: bool = True,
    ) -> UtteranceScore:
        from psdn_sonar.utils.metrics import (
            calculate_cer_wer,
            calculate_poseidon_score,
            compute_semantic_similarity,
        )

        cer, wer = calculate_cer_wer(ref_norm, hyp_norm)
        if not with_semantics:
            return UtteranceScore(cer=cer, wer=wer, semantic_similarity=None, poseidon_score=None)
        sem = compute_semantic_similarity(ref_norm, hyp_norm)
        poseidon = (
            calculate_poseidon_score(cer, wer, sem)
            if (cer is not None and wer is not None and sem is not None)
            else None
        )
        return UtteranceScore(cer=cer, wer=wer, semantic_similarity=sem, poseidon_score=poseidon)

    @classmethod
    def score_dual_variant(
        cls,
        reference: str,
        hypothesis_raw: str,
        *,
        language: str,
        with_semantics: bool = True,
    ) -> DualUtteranceScore:
        ref_norm, hyp_norm_non, hyp_norm_conv, hyp_converted = cls.normalize_reference_and_hypotheses(
            reference,
            hypothesis_raw,
            language=language,
        )
        return DualUtteranceScore(
            ref_norm=ref_norm,
            hyp_norm_non=hyp_norm_non,
            hyp_norm_conv=hyp_norm_conv,
            hyp_converted=hyp_converted,
            non=cls.score_normalized_pair(ref_norm, hyp_norm_non, with_semantics=with_semantics),
            conv=cls.score_normalized_pair(ref_norm, hyp_norm_conv, with_semantics=with_semantics),
        )

    @staticmethod
    def score_single_variant(
        reference: str,
        hypothesis_raw: str,
        *,
        language: str,
    ) -> tuple[Optional[float], Optional[float], str, str]:
        """Return (cer, wer, ref_norm, hyp_norm) with metrics unchanged from ``calculate_cer_wer``."""
        from psdn_sonar.utils.metrics import calculate_cer_wer
        from psdn_sonar.utils.text_processing import normalize_text_unified

        ref_norm = normalize_text_unified(reference, language=language)
        hyp_norm = normalize_text_unified(hypothesis_raw, language=language)
        cer, wer = calculate_cer_wer(ref_norm, hyp_norm)
        return cer, wer, ref_norm, hyp_norm
