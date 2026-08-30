import logging
import shutil
import subprocess

import numpy as np
import pytest

from psdn_sonar import audio_quality
from psdn_sonar.audio_quality import (
    assign_snr_tier,
    calculate_clipping_ratio,
    calculate_silence_ratio,
    calculate_snr,
    compute_audio_quality_metrics,
    get_audio_quality_config,
    get_quality_warnings,
)


class TestCalculateSnr:
    def test_sine_wave_has_positive_snr(self, sine_wave_audio):
        snr = calculate_snr(sine_wave_audio)
        assert snr > 0.0

    def test_all_zeros_returns_none_not_inf(self):
        audio = np.zeros(16_000, dtype=np.float32)
        assert calculate_snr(audio) is None

    def test_empty_audio_returns_none(self):
        assert calculate_snr(np.array([], dtype=np.float32)) is None

    def test_noise_free_signal_capped_not_inf(self):
        # Half signal, half digital silence: noise floor is exactly 0 but the
        # signal is real, so SNR is capped rather than None or inf.
        audio = np.concatenate([np.full(16_000, 0.5, dtype=np.float32), np.zeros(16_000, dtype=np.float32)])
        assert calculate_snr(audio) == 100.0

    def test_never_returns_inf(self, sine_wave_audio):
        for audio in (sine_wave_audio, np.zeros(16_000, dtype=np.float32)):
            snr = calculate_snr(audio)
            assert snr is None or np.isfinite(snr)

    def test_constant_signal_returns_finite_snr(self):
        audio = np.full(16_000, 0.5, dtype=np.float32)
        snr = calculate_snr(audio)
        assert np.isfinite(snr)

    def test_white_noise_has_finite_snr(self):
        rng = np.random.default_rng(42)
        noise = rng.normal(0, 0.3, 16_000).astype(np.float32)
        snr = calculate_snr(noise)
        assert np.isfinite(snr)

    def test_returns_float(self, sine_wave_audio):
        assert isinstance(calculate_snr(sine_wave_audio), float)


class TestCalculateClippingRatio:
    def test_no_clipping(self, sine_wave_audio):
        ratio = calculate_clipping_ratio(sine_wave_audio)
        assert ratio == 0.0

    def test_full_clipping(self, clipped_audio):
        ratio = calculate_clipping_ratio(clipped_audio)
        assert ratio == 1.0

    def test_partial_clipping(self):
        audio = np.array([0.0, 0.5, 1.0, 0.0], dtype=np.float32)
        ratio = calculate_clipping_ratio(audio)
        assert ratio == 0.25

    def test_empty_audio(self):
        assert calculate_clipping_ratio(np.array([], dtype=np.float32)) == 0.0


class TestCalculateSilenceRatio:
    def test_fully_silent_audio_scores_one(self):
        # Regression for issue #105: relative-to-own-max scored this 0.0,
        # identical to all-speech, and it passed the max_silence_ratio gate.
        assert calculate_silence_ratio(np.zeros(16_000, dtype=np.float32)) == 1.0

    def test_near_silent_audio_scores_one(self):
        rng = np.random.default_rng(0)
        quiet = (rng.normal(0, 1e-6, 16_000)).astype(np.float32)
        assert calculate_silence_ratio(quiet) == 1.0

    def test_all_speech_scores_near_zero(self):
        rng = np.random.default_rng(42)
        speech = (rng.normal(0, 0.2, 16_000)).astype(np.float32)
        assert calculate_silence_ratio(speech) < 0.1

    def test_half_speech_half_silence_scores_near_half(self):
        rng = np.random.default_rng(42)
        audio = np.concatenate([(rng.normal(0, 0.2, 8_000)).astype(np.float32), np.zeros(8_000, dtype=np.float32)])
        ratio = calculate_silence_ratio(audio)
        assert 0.3 <= ratio <= 0.7

    def test_loud_audio_not_silent(self, sine_wave_audio):
        ratio = calculate_silence_ratio(sine_wave_audio)
        assert ratio < 0.5

    def test_returns_between_zero_and_one(self, sine_wave_audio):
        ratio = calculate_silence_ratio(sine_wave_audio)
        assert 0.0 <= ratio <= 1.0


class TestAssignSnrTier:
    def test_low_tier(self):
        assert assign_snr_tier(5.0) == "Low"
        assert assign_snr_tier(9.9) == "Low"

    def test_medium_tier(self):
        assert assign_snr_tier(10.0) == "Medium"
        assert assign_snr_tier(15.0) == "Medium"
        assert assign_snr_tier(19.9) == "Medium"

    def test_high_tier(self):
        assert assign_snr_tier(20.0) == "High"
        assert assign_snr_tier(50.0) == "High"

    def test_negative_snr_is_low(self):
        assert assign_snr_tier(-5.0) == "Low"

    def test_boundary_low_medium(self):
        assert assign_snr_tier(9.999) == "Low"
        assert assign_snr_tier(10.0) == "Medium"

    def test_boundary_medium_high(self):
        assert assign_snr_tier(19.999) == "Medium"
        assert assign_snr_tier(20.0) == "High"


class TestComputeAudioQualityMetrics:
    def test_valid_file_returns_all_keys(self, wav_file):
        result = compute_audio_quality_metrics(wav_file)
        expected_base = {
            "snr_db",
            "clipping_ratio",
            "silence_ratio",
            "snr_tier",
            "quality_warnings",
        }
        assert expected_base.issubset(set(result.keys()))
        assert result["snr_db"] is not None
        assert result["clipping_ratio"] is not None
        assert result["silence_ratio"] is not None
        assert result["snr_tier"] in ("Low", "Medium", "High")
        assert isinstance(result["quality_warnings"], str)

    def test_nonexistent_file_returns_safe_defaults_and_says_why(self):
        result = compute_audio_quality_metrics("/nonexistent/path.wav")
        assert result["snr_db"] is None
        assert result["clipping_ratio"] is None
        assert result["silence_ratio"] is None
        assert result["snr_tier"] is None
        # Issue #206: blank metrics used to come with an empty warnings
        # cell, indistinguishable from a complete result.
        assert result["quality_warnings"].startswith("quality_metrics_unavailable:")

    def test_values_are_rounded(self, wav_file):
        result = compute_audio_quality_metrics(wav_file)
        snr_str = str(result["snr_db"])
        if "." in snr_str:
            decimals = len(snr_str.split(".")[1])
            assert decimals <= 2

    def test_silent_file_flagged_not_clean(self, tmp_path):
        # End-to-end regression for issue #105: an all-zero WAV used to
        # report silence_ratio=0.0, snr_db=inf, snr_tier=High, no warnings.
        import soundfile as sf

        path = tmp_path / "silent.wav"
        sf.write(str(path), np.zeros(16_000, dtype=np.float32), 16_000)

        result = compute_audio_quality_metrics(str(path), include_mos=False)
        assert result["silence_ratio"] == 1.0
        assert result["snr_db"] is None
        assert result["snr_tier"] is None
        assert "high_silence" in result["quality_warnings"]


_FFMPEG = shutil.which("ffmpeg") is not None
requires_ffmpeg = pytest.mark.skipif(not _FFMPEG, reason="ffmpeg not on PATH")


def _write_wav(path, seconds=1.0, sr=16_000):
    import soundfile as sf

    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    sf.write(str(path), (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32), sr)
    return path


class TestDecodeParityWithTranscription:
    """Issue #206: transcription decodes with ffmpeg by path while the
    quality metrics decoded with librosa/libsndfile, which reads neither AAC
    nor ALAC — the same M4A transcribed fine and produced twelve blank
    quality columns with an empty warnings cell and a debug-only log line."""

    @requires_ffmpeg
    def test_fallback_decodes_what_libsndfile_cannot(self, tmp_path, monkeypatch):
        """When librosa cannot open the file, the ffmpeg fallback decodes it
        and real metrics come back."""
        wav = _write_wav(tmp_path / "a.wav")

        def _libsndfile_refuses(path, sr):
            raise RuntimeError(f"Error opening '{path}': Format not recognised.")

        monkeypatch.setattr(audio_quality.librosa, "load", _libsndfile_refuses)
        result = compute_audio_quality_metrics(str(wav), include_mos=False)
        assert result["snr_db"] is not None
        assert result["snr_tier"] in ("Low", "Medium", "High")
        assert "quality_metrics_unavailable" not in result["quality_warnings"]

    @requires_ffmpeg
    def test_real_m4a_gets_quality_metrics(self, tmp_path):
        """An AAC-in-M4A file — the container the issue reproduces with —
        yields populated quality columns."""
        m4a = tmp_path / "sample.m4a"
        proc = subprocess.run(
            ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=1", "-c:a", "aac", str(m4a)],
            capture_output=True,
        )
        if proc.returncode != 0:
            pytest.skip(f"ffmpeg cannot encode AAC here: {proc.stderr.decode(errors='replace')[:100]}")

        result = compute_audio_quality_metrics(str(m4a), include_mos=False)
        assert result["snr_db"] is not None
        assert result["clipping_ratio"] is not None
        assert result["silence_ratio"] is not None
        assert result["snr_tier"] in ("Low", "Medium", "High")

    def test_undecodable_file_warns_loudly_and_names_both_decoders(self, tmp_path, caplog):
        bad = tmp_path / "noise.m4a"
        bad.write_bytes(b"\x00not audio at all")

        with caplog.at_level(logging.WARNING, logger="psdn_sonar.audio_quality"):
            result = compute_audio_quality_metrics(str(bad), include_mos=True)

        assert result["snr_db"] is None
        warning_cell = result["quality_warnings"]
        assert warning_cell.startswith("quality_metrics_unavailable:")
        assert "librosa" in warning_cell
        assert "ffmpeg" in warning_cell
        assert any("Audio quality metrics unavailable" in r.message for r in caplog.records)
        # MOS columns are filled with their empty defaults, not omitted.
        assert "mos_tier" in result and result["mos_tier"] is None

    def test_mos_failure_is_named_next_to_the_base_metrics(self, tmp_path, monkeypatch, caplog):
        import psdn_sonar.quality_models as quality_models

        wav = _write_wav(tmp_path / "a.wav")

        def _boom(*args, **kwargs):
            raise RuntimeError("onnx runtime exploded")

        monkeypatch.setattr(quality_models, "compute_mos_metrics", _boom)
        with caplog.at_level(logging.WARNING, logger="psdn_sonar.audio_quality"):
            result = compute_audio_quality_metrics(str(wav), include_mos=True)

        assert result["snr_db"] is not None  # base metrics unaffected
        assert "mos_metrics_unavailable: onnx runtime exploded" in result["quality_warnings"]
        assert result["mos_tier"] is None


class TestGetAudioQualityConfig:
    def test_returns_config_with_defaults(self):
        cfg = get_audio_quality_config()
        assert cfg.snr_tier_low_db == 10.0
        assert cfg.snr_tier_high_db == 20.0
        assert cfg.silence_thresh_db == -40.0
        assert cfg.silence_floor_amplitude == 1e-3
        assert cfg.clipping_amplitude == 0.99
        assert cfg.min_snr_db == 10.0
        assert cfg.max_silence_ratio == 0.60
        assert cfg.max_clipping_ratio == 0.001


class TestGetQualityWarnings:
    def test_empty_when_all_pass(self):
        assert get_quality_warnings(25.0, 0.2, 0.0) == []

    def test_low_snr_warning(self):
        w = get_quality_warnings(5.0, 0.2, 0.0)
        assert len(w) == 1
        assert "low_snr" in w[0]

    def test_high_silence_warning(self):
        w = get_quality_warnings(25.0, 0.9, 0.0)
        assert len(w) == 1
        assert "high_silence" in w[0]

    def test_clipping_warning(self):
        w = get_quality_warnings(25.0, 0.2, 0.01)
        assert len(w) == 1
        assert "clipping" in w[0]

    def test_none_metrics_no_warnings(self):
        assert get_quality_warnings(None, None, None) == []

    def test_multiple_warnings(self):
        w = get_quality_warnings(5.0, 0.9, 0.01)
        assert len(w) >= 2
        assert any("low_snr" in x for x in w)
        assert any("high_silence" in x for x in w)
