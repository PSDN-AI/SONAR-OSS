"""Tests for psdn_sonar.quality_models — reference-free speech quality scorers."""

import numpy as np

from psdn_sonar.quality_models import (
    _EMPTY_MOS,
    MOS_TIER_HIGH_THRESHOLD,
    MOS_TIER_LOW_THRESHOLD,
    assign_mos_tier,
    compute_mos_metrics,
    score_dnsmos,
    score_squim,
    score_utmos,
)


class TestAssignMosTier:
    def test_high(self):
        assert assign_mos_tier(4.0) == "High"

    def test_medium(self):
        assert assign_mos_tier(3.0) == "Medium"

    def test_low(self):
        assert assign_mos_tier(2.0) == "Low"

    def test_boundary_high(self):
        assert assign_mos_tier(MOS_TIER_HIGH_THRESHOLD) == "High"

    def test_boundary_medium(self):
        assert assign_mos_tier(MOS_TIER_LOW_THRESHOLD) == "Medium"

    def test_none(self):
        assert assign_mos_tier(None) is None


class TestScoreDnsmos:
    def test_returns_dict_with_expected_keys(self):
        audio = np.random.randn(16000).astype(np.float32)
        result = score_dnsmos(audio, sr=16000)
        assert "dnsmos_ovrl" in result
        assert "dnsmos_sig" in result
        assert "dnsmos_bak" in result

    def test_scores_are_numeric_or_none(self):
        audio = np.random.randn(16000).astype(np.float32)
        result = score_dnsmos(audio, sr=16000)
        for key in ("dnsmos_ovrl", "dnsmos_sig", "dnsmos_bak"):
            assert result[key] is None or isinstance(result[key], float)

    def test_scores_in_valid_range(self):
        audio = np.random.randn(16000 * 3).astype(np.float32) * 0.1
        result = score_dnsmos(audio, sr=16000)
        if result["dnsmos_ovrl"] is not None:
            assert 1.0 <= result["dnsmos_ovrl"] <= 5.0
            assert 1.0 <= result["dnsmos_sig"] <= 5.0
            assert 1.0 <= result["dnsmos_bak"] <= 5.0


class TestScoreUtmos:
    def test_returns_dict_with_utmos_key(self):
        audio = np.random.randn(16000).astype(np.float32)
        result = score_utmos(audio, sr=16000)
        assert "utmos" in result

    def test_score_is_numeric_or_none(self):
        audio = np.random.randn(16000).astype(np.float32)
        result = score_utmos(audio, sr=16000)
        assert result["utmos"] is None or isinstance(result["utmos"], float)


class TestScoreSquim:
    def test_returns_dict_with_expected_keys(self):
        audio = np.random.randn(16000).astype(np.float32)
        result = score_squim(audio, sr=16000)
        assert "squim_pesq" in result
        assert "squim_stoi" in result
        assert "squim_si_sdr" in result

    def test_scores_are_numeric_or_none(self):
        audio = np.random.randn(16000).astype(np.float32)
        result = score_squim(audio, sr=16000)
        for key in ("squim_pesq", "squim_stoi", "squim_si_sdr"):
            assert result[key] is None or isinstance(result[key], float)


class TestComputeMosMetrics:
    def test_returns_all_expected_keys(self, tmp_path):
        import soundfile as sf

        audio = np.random.randn(16000 * 2).astype(np.float32) * 0.1
        wav_path = str(tmp_path / "test.wav")
        sf.write(wav_path, audio, 16000)

        result = compute_mos_metrics(wav_path)
        for key in _EMPTY_MOS:
            assert key in result, f"Missing key: {key}"

    def test_invalid_path_returns_empty_with_reason(self):
        result = compute_mos_metrics("/nonexistent/path.wav")
        for key, value in _EMPTY_MOS.items():
            assert result[key] == value
        # Issue #245: the reason the columns are empty must be reported.
        assert len(result["mos_warnings"]) == 1
        assert result["mos_warnings"][0].startswith("mos_metrics_unavailable:")

    def test_mos_tier_assigned(self, tmp_path):
        import soundfile as sf

        audio = np.random.randn(16000 * 2).astype(np.float32) * 0.1
        wav_path = str(tmp_path / "test.wav")
        sf.write(wav_path, audio, 16000)

        result = compute_mos_metrics(wav_path)
        if result["dnsmos_ovrl"] is not None:
            assert result["mos_tier"] in ("Low", "Medium", "High")

    def test_accepts_ndarray_directly(self):
        """Passing an np.ndarray avoids redundant librosa.load I/O."""
        audio = np.random.randn(16000 * 2).astype(np.float32) * 0.1
        result = compute_mos_metrics(audio, sr=16000)
        for key in _EMPTY_MOS:
            assert key in result, f"Missing key: {key}"


class TestComputeAudioQualityMetricsWithMos:
    """Integration test: compute_audio_quality_metrics now includes MOS."""

    def test_includes_mos_keys(self, tmp_path):
        import soundfile as sf

        from psdn_sonar.audio_quality import compute_audio_quality_metrics

        audio = np.random.randn(16000 * 2).astype(np.float32) * 0.1
        wav_path = str(tmp_path / "test.wav")
        sf.write(wav_path, audio, 16000)

        result = compute_audio_quality_metrics(wav_path, include_mos=True)
        assert "snr_db" in result
        assert "dnsmos_ovrl" in result
        assert "utmos" in result
        assert "squim_pesq" in result

    def test_without_mos(self, tmp_path):
        import soundfile as sf

        from psdn_sonar.audio_quality import compute_audio_quality_metrics

        audio = np.random.randn(16000 * 2).astype(np.float32) * 0.1
        wav_path = str(tmp_path / "test.wav")
        sf.write(wav_path, audio, 16000)

        result = compute_audio_quality_metrics(wav_path, include_mos=False)
        assert "snr_db" in result
        assert "dnsmos_ovrl" not in result


class TestMosFailureReasonsRecorded:
    """Issue #245: a family that produced no value must say why via
    ``mos_warnings`` instead of leaving an empty column whose only trace is
    one terminal WARNING line."""

    @staticmethod
    def _stub_families(monkeypatch, *, utmos_value):
        import psdn_sonar.quality_models as qm

        monkeypatch.setattr(
            qm, "score_dnsmos", lambda audio, sr=16000: {"dnsmos_ovrl": 2.6, "dnsmos_sig": 3.0, "dnsmos_bak": 3.1}
        )
        monkeypatch.setattr(
            qm, "score_squim", lambda audio, sr=16000: {"squim_pesq": 2.4, "squim_stoi": 0.9, "squim_si_sdr": 18.0}
        )
        monkeypatch.setattr(qm, "score_utmos", lambda audio, sr=16000: {"utmos": utmos_value})

    def test_utmos_fetch_failure_lands_in_mos_warnings(self, monkeypatch):
        import psdn_sonar.quality_models as qm

        self._stub_families(monkeypatch, utmos_value=None)
        monkeypatch.setattr(qm, "_utmos_error", "'Authorization'")

        result = compute_mos_metrics(np.zeros(16000, dtype=np.float32))

        assert result["utmos"] is None
        assert result["dnsmos_ovrl"] == 2.6  # siblings still populated
        assert result["mos_warnings"] == ["utmos_unavailable: 'Authorization'"]

    def test_successful_scores_produce_no_warnings(self, monkeypatch):
        import psdn_sonar.quality_models as qm

        self._stub_families(monkeypatch, utmos_value=4.1)
        # Even a stale error from an earlier clip must not fire once the
        # family produces a value again.
        monkeypatch.setattr(qm, "_utmos_error", "'Authorization'")

        result = compute_mos_metrics(np.zeros(16000, dtype=np.float32))

        assert result["utmos"] == 4.1
        assert result["mos_warnings"] == []

    def test_loader_failure_records_the_reason(self, monkeypatch):
        """The issue's deterministic repro: torch.hub.load raising during the
        UTMOS fetch."""
        import torch

        import psdn_sonar.quality_models as qm

        def boom(*args, **kwargs):
            raise KeyError("Authorization")

        monkeypatch.setattr(torch.hub, "load", boom)
        monkeypatch.setattr(qm, "_utmos_predictor", None)
        monkeypatch.setattr(qm, "_utmos_available", None)
        monkeypatch.setattr(qm, "_utmos_error", None)

        assert score_utmos(np.zeros(16000, dtype=np.float32)) == {"utmos": None}
        assert qm._utmos_error == "'Authorization'"
