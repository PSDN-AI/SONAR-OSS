import numpy as np

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

    def test_all_zeros_returns_inf(self):
        audio = np.zeros(16_000, dtype=np.float32)
        assert calculate_snr(audio) == float("inf")

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
    def test_near_silent_audio_detected(self):
        rng = np.random.default_rng(0)
        quiet = (rng.normal(0, 1e-5, 16_000)).astype(np.float32)
        ratio = calculate_silence_ratio(quiet)
        assert isinstance(ratio, float)
        assert 0.0 <= ratio <= 1.0

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

    def test_nonexistent_file_returns_safe_defaults(self):
        result = compute_audio_quality_metrics("/nonexistent/path.wav")
        assert result["snr_db"] is None
        assert result["clipping_ratio"] is None
        assert result["silence_ratio"] is None
        assert result["snr_tier"] is None
        assert result.get("quality_warnings") == ""

    def test_values_are_rounded(self, wav_file):
        result = compute_audio_quality_metrics(wav_file)
        snr_str = str(result["snr_db"])
        if "." in snr_str:
            decimals = len(snr_str.split(".")[1])
            assert decimals <= 2


class TestGetAudioQualityConfig:
    def test_returns_config_with_defaults(self):
        cfg = get_audio_quality_config()
        assert cfg.snr_tier_low_db == 10.0
        assert cfg.snr_tier_high_db == 20.0
        assert cfg.silence_thresh_db == -40.0
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
