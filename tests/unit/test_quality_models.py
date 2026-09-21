"""Tests for psdn_sonar.quality_models — reference-free speech quality scorers."""

import numpy as np
import pytest

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
        torch = pytest.importorskip("torch")

        import psdn_sonar.quality_models as qm

        def boom(*args, **kwargs):
            raise KeyError("Authorization")

        monkeypatch.setattr(torch.hub, "load", boom)
        monkeypatch.setattr(qm, "_utmos_predictor", None)
        monkeypatch.setattr(qm, "_utmos_available", None)
        monkeypatch.setattr(qm, "_utmos_error", None)

        assert score_utmos(np.zeros(16000, dtype=np.float32)) == {"utmos": None}
        assert qm._utmos_error == "'Authorization'"


class TestMetaDevicePredictorRefused:
    """Issue #288: a predictor constructed while a concurrent
    init_empty_weights() window was open has parameters on the meta device —
    shapes and no values. The loader used to report it healthy; every score
    then failed and was discarded, emptying the column. A meta-device model
    must be reported as unavailable, with the reason recorded."""

    @staticmethod
    def _meta_module():
        torch = pytest.importorskip("torch")
        import torch.nn as nn

        with torch.device("meta"):
            return nn.Linear(4, 4)

    def test_reject_helper_names_the_family_and_the_window(self):
        import psdn_sonar.quality_models as qm

        with pytest.raises(RuntimeError, match="UTMOS was constructed on the meta device"):
            qm._reject_meta_parameters(self._meta_module(), "UTMOS")

    def test_cpu_module_passes_the_guard(self):
        torch = pytest.importorskip("torch")
        import torch.nn as nn

        import psdn_sonar.quality_models as qm

        qm._reject_meta_parameters(nn.Linear(4, 4), "UTMOS")
        assert torch is not None  # importorskip guard

    def test_object_without_parameters_is_tolerated(self):
        import psdn_sonar.quality_models as qm

        qm._reject_meta_parameters(object(), "DNSMOS")

    def test_meta_utmos_is_unavailable_not_loaded(self, monkeypatch):
        """The issue's product state: _get_utmos returned normally, the
        predictor was a real UTMOS22Strong on meta, and _utmos_error was
        None. Now the loader refuses it and records why."""
        torch = pytest.importorskip("torch")

        import psdn_sonar.quality_models as qm

        broken = self._meta_module()
        monkeypatch.setattr(torch.hub, "load", lambda *a, **k: broken)
        monkeypatch.setattr(qm, "_utmos_predictor", None)
        monkeypatch.setattr(qm, "_utmos_available", None)
        monkeypatch.setattr(qm, "_utmos_error", None)

        assert qm._get_utmos() is None
        assert qm._utmos_available is False
        assert "meta device" in qm._utmos_error
        assert "issue #288" in qm._utmos_error
        assert score_utmos(np.zeros(16000, dtype=np.float32)) == {"utmos": None}

    def test_meta_squim_is_unavailable_not_loaded(self, monkeypatch):
        """The issue lists SQUIM as structurally identical and untested."""
        pytest.importorskip("torch")
        import sys
        from types import SimpleNamespace

        import psdn_sonar.quality_models as qm

        broken = self._meta_module()
        fake_pipelines = SimpleNamespace(SQUIM_OBJECTIVE=SimpleNamespace(get_model=lambda: broken, sample_rate=16000))
        monkeypatch.setitem(sys.modules, "torchaudio.pipelines", fake_pipelines)
        monkeypatch.setattr(qm, "_squim_model", None)
        monkeypatch.setattr(qm, "_squim_available", None)
        monkeypatch.setattr(qm, "_squim_error", None)

        assert qm._get_squim() is None
        assert qm._squim_available is False
        assert "meta device" in qm._squim_error

    def test_meta_failure_reaches_mos_warnings(self, monkeypatch):
        """End to end through compute_mos_metrics: the refusal must arrive in
        the artifact-bound mos_warnings, not just a debug line."""
        torch = pytest.importorskip("torch")

        import psdn_sonar.quality_models as qm

        broken = self._meta_module()
        monkeypatch.setattr(torch.hub, "load", lambda *a, **k: broken)
        monkeypatch.setattr(qm, "_utmos_predictor", None)
        monkeypatch.setattr(qm, "_utmos_available", None)
        monkeypatch.setattr(qm, "_utmos_error", None)
        monkeypatch.setattr(
            qm, "score_dnsmos", lambda audio, sr=16000: {"dnsmos_ovrl": 2.6, "dnsmos_sig": 3.0, "dnsmos_bak": 3.1}
        )
        monkeypatch.setattr(
            qm, "score_squim", lambda audio, sr=16000: {"squim_pesq": 2.4, "squim_stoi": 0.9, "squim_si_sdr": 18.0}
        )

        result = compute_mos_metrics(np.zeros(16000, dtype=np.float32))

        assert result["utmos"] is None
        assert len(result["mos_warnings"]) == 1
        assert result["mos_warnings"][0].startswith("utmos_unavailable:")
        assert "meta device" in result["mos_warnings"][0]


class TestPrewarmJoinsBeforeTheModelFactory:
    """Issue #288, the race itself: the prewarm thread's constructions must
    finish before the ASR factory can open transformers'
    init_empty_weights() window."""

    def test_quality_models_finish_loading_before_the_factory_runs(self, tmp_path, monkeypatch):
        import threading
        import time
        from types import SimpleNamespace

        import psdn_sonar.quality_models as qm
        from psdn_sonar.evaluators.single_speaker import SingleSpeakerEvaluator

        events = []
        lock = threading.Lock()

        def record(name, delay=0.0):
            time.sleep(delay)
            with lock:
                events.append(name)

        # The prewarm thread runs these three; give the first one enough
        # delay that, without the join, the factory would win the race.
        monkeypatch.setattr(qm, "_get_dnsmos", lambda: record("prewarm_dnsmos", delay=0.3))
        monkeypatch.setattr(qm, "_get_utmos", lambda: record("prewarm_utmos"))
        monkeypatch.setattr(qm, "_get_squim", lambda: record("prewarm_squim"))

        def factory(*args, **kwargs):
            record("factory")
            return SimpleNamespace(provider="test", provider_model_id=None)

        monkeypatch.setattr("psdn_sonar.evaluators.single_speaker.load_env", lambda: None)
        monkeypatch.setattr(
            SingleSpeakerEvaluator,
            "load_data",
            lambda *a, **k: [{"audio_path": "clip.wav", "ground_truth": "hello world"}],
        )
        monkeypatch.setattr("psdn_sonar.evaluators.single_speaker._model_factory", factory)
        monkeypatch.setattr(
            SingleSpeakerEvaluator,
            "evaluate_one",
            lambda *a, **k: {
                "model_name": "whisper_base_en",
                "results": [],
                "summary": {
                    "total_samples": 1,
                    "successful": 1,
                    "failed": 0,
                    "avg_wer": 0.1,
                    "avg_cer": 0.05,
                    "elapsed_time": 0.1,
                    "avg_latency_s": None,
                    "median_latency_s": None,
                    "p95_latency_s": None,
                },
            },
        )

        SingleSpeakerEvaluator.run_evaluation(
            tsv_path="eval.tsv",
            output_dir=str(tmp_path),
            models=["whisper_base_en"],
            language="en",
            write_scores=False,
            compute_sem=False,
        )

        assert "factory" in events
        factory_at = events.index("factory")
        assert set(events[:factory_at]) == {"prewarm_dnsmos", "prewarm_utmos", "prewarm_squim"}, (
            f"factory ran before the prewarm finished: {events}"
        )
