"""Tests for the ``significantWer`` pass/fail metric.

Threshold semantics under test:

- The flag is True iff WER >= threshold (boundary value fails on purpose).
- ``None`` / ``NaN`` / ``inf`` WER inputs propagate as ``None`` for the
  per-utterance flag, and are excluded from both numerator and denominator
  for the aggregate rate, so missing measurements never silently pass.
- The aggregate returns ``None`` (not 0.0) when no utterance has a finite
  WER, so callers can distinguish "0% significant errors over 100 rows"
  from "no measurable inputs at all".
- Negative thresholds are rejected at call time.
"""

from __future__ import annotations

import math

import pytest

from psdn_sonar.utils.metrics import (
    DEFAULT_SIGNIFICANT_WER_THRESHOLD,
    is_significant_wer,
    significant_wer_rate,
)


class TestIsSignificantWer:
    """Per-utterance flag boundary behaviour."""

    def test_default_threshold_matches_published_constant(self):
        assert DEFAULT_SIGNIFICANT_WER_THRESHOLD == 0.30

    @pytest.mark.parametrize("wer", [0.0, 0.1, 0.2999])
    def test_below_threshold_is_false(self, wer):
        assert is_significant_wer(wer) is False

    def test_at_threshold_is_true(self):
        """Exact-threshold case must fail (>=, not >). 0.30 means 30% of
        words wrong; that is a borderline failure, not a borderline pass."""
        assert is_significant_wer(0.30) is True

    @pytest.mark.parametrize("wer", [0.3001, 0.5, 1.0, 1.5])
    def test_above_threshold_is_true(self, wer):
        # WER can exceed 1.0 (jiwer counts insertions); the flag must still
        # fire rather than wrap around or clamp.
        assert is_significant_wer(wer) is True

    @pytest.mark.parametrize("wer", [None, float("nan"), float("inf"), -math.inf])
    def test_missing_or_non_finite_returns_none(self, wer):
        assert is_significant_wer(wer) is None

    @pytest.mark.parametrize("wer", ["not-a-number", object()])
    def test_non_numeric_returns_none(self, wer):
        assert is_significant_wer(wer) is None

    def test_custom_threshold_is_respected(self):
        assert is_significant_wer(0.4, threshold=0.5) is False
        assert is_significant_wer(0.5, threshold=0.5) is True

    def test_threshold_zero_treats_every_finite_wer_as_significant(self):
        # A zero threshold means every utterance with any error is flagged.
        # Only WER == 0 (perfect) escapes -- and even WER == 0 is "at the
        # threshold", which is the >= boundary case, so it's also True.
        assert is_significant_wer(0.0, threshold=0.0) is True
        assert is_significant_wer(0.001, threshold=0.0) is True

    def test_negative_threshold_rejected(self):
        with pytest.raises(ValueError, match="significantWer threshold must be >= 0"):
            is_significant_wer(0.1, threshold=-0.01)


class TestSignificantWerRate:
    """Aggregate rate behaviour."""

    def test_simple_rate(self):
        # 2 of 4 utterances at/above 0.30: 0.30 (boundary, True), 0.5 (True),
        # 0.1 (False), 0.0 (False). Rate = 2/4 = 0.5.
        assert significant_wer_rate([0.30, 0.5, 0.1, 0.0]) == pytest.approx(0.5)

    def test_all_pass_returns_zero(self):
        assert significant_wer_rate([0.0, 0.05, 0.1, 0.2]) == 0.0

    def test_all_fail_returns_one(self):
        assert significant_wer_rate([0.3, 0.5, 0.7, 1.0]) == 1.0

    def test_empty_input_returns_none(self):
        """No utterances -> no rate. Distinct from "0.0 of N have errors"."""
        assert significant_wer_rate([]) is None

    def test_all_missing_returns_none(self):
        """Missing measurements must not collapse into a fake "all pass" 0.0."""
        assert significant_wer_rate([None, float("nan"), float("inf")]) is None

    def test_missing_excluded_from_numerator_and_denominator(self):
        # Two finite values (0.5 fails, 0.1 passes) -> rate over n=2 is 0.5.
        # The two None entries must not push the denominator to 4.
        assert significant_wer_rate([0.5, None, 0.1, float("nan")]) == pytest.approx(0.5)

    def test_custom_threshold_is_respected(self):
        # At threshold=0.5: 0.4 passes, 0.5 fails, 0.6 fails -> 2/3.
        assert significant_wer_rate([0.4, 0.5, 0.6], threshold=0.5) == pytest.approx(2 / 3)

    def test_negative_threshold_rejected(self):
        with pytest.raises(ValueError, match="significantWer threshold must be >= 0"):
            significant_wer_rate([0.1, 0.2], threshold=-1.0)
