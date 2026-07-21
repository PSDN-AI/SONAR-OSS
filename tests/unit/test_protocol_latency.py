"""Tests for protocol-aware latency (TTFT vs complete).

Covers:
  * ``LatencyMetrics`` + ``unpack_transcription`` (the backwards-compatible
    transcribe return contract).
  * ``compute_protocol_latency_summary`` for batch-only (ttft None), a
    streaming set (both populated), and a mixed set.

Evaluator wiring and the AssemblyAI streaming TTFT capture are covered
alongside their own modules (evaluators / API adapters).
"""

from __future__ import annotations

import pytest

from psdn_sonar.models.base import ASRModel, LatencyMetrics, unpack_transcription
from psdn_sonar.utils.metrics import compute_protocol_latency_summary


class TestLatencyMetrics:
    def test_batch_defaults_ttft_none(self):
        lm = LatencyMetrics(complete_s=0.4)
        assert lm.complete_s == 0.4
        assert lm.ttft_s is None

    def test_streaming_carries_both(self):
        lm = LatencyMetrics(complete_s=0.8, ttft_s=0.2)
        assert lm.ttft_s == 0.2

    def test_base_model_does_not_advertise_latency(self):
        assert ASRModel.supports_latency_metrics is False


class TestUnpackTranscription:
    def test_bare_string_synthesises_complete_from_fallback(self):
        text, lm = unpack_transcription("hi", fallback_complete_s=0.5)
        assert text == "hi"
        assert lm.complete_s == 0.5
        assert lm.ttft_s is None

    def test_tuple_returned_as_is(self):
        original = LatencyMetrics(complete_s=1.0, ttft_s=0.3)
        text, lm = unpack_transcription(("hi", original), fallback_complete_s=99.0)
        assert text == "hi"
        assert lm is original  # fallback ignored when adapter reports its own

    def test_none_without_fallback_has_no_metrics(self):
        text, lm = unpack_transcription(None)
        assert text is None
        assert lm is None


class TestProtocolLatencySummary:
    def test_batch_only_set_has_none_ttft(self):
        out = compute_protocol_latency_summary([0.2, 0.4, 0.6], [None, None, None])
        assert out["complete_p50"] == pytest.approx(0.4)
        assert out["ttft_p50"] is None
        assert out["ttft_p95"] is None

    def test_streaming_set_populates_both(self):
        out = compute_protocol_latency_summary([0.5, 0.7, 0.9], [0.1, 0.2, 0.3])
        assert out["complete_p50"] == pytest.approx(0.7)
        assert out["ttft_p50"] == pytest.approx(0.2)
        assert out["ttft_p95"] is not None

    def test_mixed_set_uses_only_finite_ttft(self):
        # A mixed adapter set still summarises — TTFT percentiles are
        # computed over only the streaming rows that reported a value.
        out = compute_protocol_latency_summary([0.2, 0.4, 0.6, 0.8], [None, 0.1, None, 0.3])
        assert out["complete_p50"] == pytest.approx(0.5)
        assert out["ttft_p50"] == pytest.approx(0.2)

    def test_empty_inputs_all_none(self):
        out = compute_protocol_latency_summary([], [])
        assert out == {
            "complete_p50": None,
            "complete_p95": None,
            "ttft_p50": None,
            "ttft_p95": None,
        }

    def test_non_finite_values_dropped(self):
        out = compute_protocol_latency_summary([float("nan"), float("inf"), 0.3], None)
        assert out["complete_p50"] == pytest.approx(0.3)
        assert out["ttft_p50"] is None
