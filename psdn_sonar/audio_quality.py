"""
Audio quality metrics: SNR, clipping ratio, silence ratio, and SNR tier assignment.

Computes per-utterance audio quality metrics inline during evaluation.
Thresholds are configurable via environment variables; see get_audio_quality_config().
"""

import logging
import os
from dataclasses import dataclass
from typing import List

import librosa
import numpy as np

logger = logging.getLogger(__name__)

# Defaults (used when env vars are not set)
_DEFAULT_SNR_TIER_LOW = 10.0
_DEFAULT_SNR_TIER_HIGH = 20.0
_DEFAULT_CLIPPING_AMPLITUDE = 0.99
_DEFAULT_SILENCE_THRESH_DB = -40.0
_DEFAULT_MAX_SILENCE_RATIO = 0.60
_DEFAULT_MAX_CLIPPING_RATIO = 0.001
_DEFAULT_MIN_SNR_DB = 10.0
SAMPLE_RATE = 16_000


def _safe_float(env_var: str, default: float) -> float:
    raw = os.getenv(env_var)
    if raw is None:
        return default
    try:
        return float(raw)
    except (ValueError, TypeError):
        logger.warning("Invalid %s=%r, using default %.2f", env_var, raw, default)
        return default


@dataclass(frozen=True)
class AudioQualityConfig:
    """Configurable thresholds for audio quality metrics. Read from env on first use."""

    snr_tier_low_db: float
    snr_tier_high_db: float
    silence_thresh_db: float
    max_silence_ratio: float
    max_clipping_ratio: float
    clipping_amplitude: float
    min_snr_db: float


_audio_quality_config: AudioQualityConfig | None = None


def get_audio_quality_config() -> AudioQualityConfig:
    """Return audio quality thresholds. Reads from environment on first access."""
    global _audio_quality_config
    if _audio_quality_config is None:
        _audio_quality_config = AudioQualityConfig(
            snr_tier_low_db=_safe_float("SONAR_SNR_TIER_LOW", _DEFAULT_SNR_TIER_LOW),
            snr_tier_high_db=_safe_float("SONAR_SNR_TIER_HIGH", _DEFAULT_SNR_TIER_HIGH),
            silence_thresh_db=_safe_float("SONAR_SILENCE_THRESH_DB", _DEFAULT_SILENCE_THRESH_DB),
            max_silence_ratio=_safe_float("SONAR_MAX_SILENCE_RATIO", _DEFAULT_MAX_SILENCE_RATIO),
            max_clipping_ratio=_safe_float("SONAR_MAX_CLIPPING_RATIO", _DEFAULT_MAX_CLIPPING_RATIO),
            clipping_amplitude=_safe_float("SONAR_CLIPPING_AMPLITUDE", _DEFAULT_CLIPPING_AMPLITUDE),
            min_snr_db=_safe_float("SONAR_MIN_SNR_DB", _DEFAULT_MIN_SNR_DB),
        )
    return _audio_quality_config


def get_quality_warnings(
    snr_db: float | None,
    silence_ratio: float | None,
    clipping_ratio: float | None,
) -> List[str]:
    """
    Return a list of human-readable warnings when metrics fail config thresholds.

    Used for optional quality_warnings column so downstream can filter or flag items.
    """
    cfg = get_audio_quality_config()
    warnings: List[str] = []
    if snr_db is not None and snr_db < cfg.min_snr_db:
        warnings.append(f"low_snr:{snr_db:.1f}dB_min_{cfg.min_snr_db}dB")
    if silence_ratio is not None and silence_ratio > cfg.max_silence_ratio:
        warnings.append(f"high_silence:{silence_ratio:.2%}_max_{cfg.max_silence_ratio:.0%}")
    if clipping_ratio is not None and clipping_ratio > cfg.max_clipping_ratio:
        warnings.append(f"clipping:{clipping_ratio:.4%}_max_{cfg.max_clipping_ratio:.3%}")
    return warnings


def calculate_snr(audio: np.ndarray, frame_length: int = 2048, hop_length: int = 512) -> float:
    """Estimate Signal-to-Noise Ratio in dB using frame-level RMS energy.

    Splits the audio into short frames, computes the RMS of each frame,
    then uses the bottom 10 % of frames (by energy) as a noise-floor
    estimate.  This is more robust than sample-level percentile methods
    because speech pauses map to low-energy frames where only background
    noise is present.
    """
    rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
    if len(rms) == 0:
        return float("inf")

    rms_sorted = np.sort(rms)
    n_noise = max(1, len(rms) // 10)
    noise_rms = float(np.mean(rms_sorted[:n_noise]))
    signal_rms = float(np.mean(rms))

    if noise_rms == 0:
        return float("inf")

    return float(20.0 * np.log10(signal_rms / noise_rms))


def calculate_clipping_ratio(audio: np.ndarray) -> float:
    """Fraction of samples whose absolute value exceeds the clipping threshold."""
    if len(audio) == 0:
        return 0.0
    cfg = get_audio_quality_config()
    return float(np.mean(np.abs(audio) > cfg.clipping_amplitude))


def calculate_silence_ratio(audio: np.ndarray, thresh_db: float | None = None) -> float:
    """Fraction of short-time frames whose RMS energy is below *thresh_db* (or config default)."""
    if thresh_db is None:
        thresh_db = get_audio_quality_config().silence_thresh_db
    rms = librosa.feature.rms(y=audio, frame_length=2048, hop_length=512)[0]
    if len(rms) == 0:
        return 0.0
    rms_db = librosa.amplitude_to_db(rms, ref=np.max)
    return float(np.mean(rms_db < thresh_db))


def assign_snr_tier(snr_db: float) -> str:
    """Map an SNR value to Low / Medium / High quality tier using config thresholds."""
    cfg = get_audio_quality_config()
    if snr_db < cfg.snr_tier_low_db:
        return "Low"
    if snr_db < cfg.snr_tier_high_db:
        return "Medium"
    return "High"


def compute_audio_quality_metrics(audio_path: str, include_mos: bool = True) -> dict:
    """
    Load an audio file and return all audio-quality metrics as a dict.

    Keys: snr_db, clipping_ratio, silence_ratio, snr_tier, quality_warnings.
    When *include_mos* is True (default), also includes DNSMOS, UTMOS, and
    SQUIM reference-free quality scores via :func:`quality_models.compute_mos_metrics`.
    Returns safe defaults on any error so evaluation is never blocked.
    """
    from psdn_sonar.quality_models import _EMPTY_MOS, compute_mos_metrics

    base_empty: dict = {
        "snr_db": None,
        "clipping_ratio": None,
        "silence_ratio": None,
        "snr_tier": None,
        "quality_warnings": "",
    }

    audio = None
    try:
        audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE)
        snr = calculate_snr(audio)
        clipping = calculate_clipping_ratio(audio)
        silence = calculate_silence_ratio(audio)
        result = {
            "snr_db": round(snr, 2),
            "clipping_ratio": round(clipping, 6),
            "silence_ratio": round(silence, 4),
            "snr_tier": assign_snr_tier(snr),
            "quality_warnings": "; ".join(get_quality_warnings(snr, silence, clipping)) or "",
        }
    except Exception as exc:
        logger.debug("Audio quality extraction failed for %s: %s", audio_path, exc)
        result = dict(base_empty)

    if include_mos:
        try:
            result.update(compute_mos_metrics(audio if audio is not None else audio_path, sr=SAMPLE_RATE))
        except Exception as exc:
            logger.debug("MOS metrics failed for %s: %s", audio_path, exc)
            result.update(_EMPTY_MOS)
    return result
