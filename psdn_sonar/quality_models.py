"""
Reference-free speech quality models: DNSMOS, UTMOS, SQUIM.

Each scorer is lazily loaded on first use and cached for the session.
All public functions return ``None`` on failure so evaluation is never blocked.
"""

import logging
import sys
import threading
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

MOS_TIER_HIGH_THRESHOLD = 3.5
MOS_TIER_LOW_THRESHOLD = 2.5

SAMPLE_RATE = 16_000


def assign_mos_tier(mos: Optional[float]) -> Optional[str]:
    if mos is None:
        return None
    if mos >= MOS_TIER_HIGH_THRESHOLD:
        return "High"
    if mos >= MOS_TIER_LOW_THRESHOLD:
        return "Medium"
    return "Low"


# ---------------------------------------------------------------------------
# DNSMOS  (speechmos / ONNX)
# ---------------------------------------------------------------------------

_dnsmos_model = None
_dnsmos_available: Optional[bool] = None
_dnsmos_lock = threading.Lock()


def _get_dnsmos():
    global _dnsmos_model, _dnsmos_available
    if _dnsmos_available is not None:
        return _dnsmos_model
    with _dnsmos_lock:
        if _dnsmos_available is not None:
            return _dnsmos_model
        try:
            from speechmos import dnsmos  # noqa: F811

            _dnsmos_model = dnsmos
            _dnsmos_available = True
            logger.debug("DNSMOS model loaded (speechmos)")
        except Exception as exc:
            logger.warning("DNSMOS unavailable: %s", exc)
            _dnsmos_available = False
    return _dnsmos_model


def score_dnsmos(audio: np.ndarray, sr: int = SAMPLE_RATE) -> Dict[str, Optional[float]]:
    """Return OVRL / SIG / BAK scores (1-5 scale) or Nones."""
    empty = {"dnsmos_ovrl": None, "dnsmos_sig": None, "dnsmos_bak": None}
    model = _get_dnsmos()
    if model is None:
        return empty
    try:
        result = model.run(audio, sr)
        return {
            "dnsmos_ovrl": round(float(result["ovrl_mos"]), 3),
            "dnsmos_sig": round(float(result["sig_mos"]), 3),
            "dnsmos_bak": round(float(result["bak_mos"]), 3),
        }
    except Exception as exc:
        logger.debug("DNSMOS scoring failed: %s", exc)
        return empty


# ---------------------------------------------------------------------------
# UTMOS  (torch.hub / SpeechMOS)
# ---------------------------------------------------------------------------

_utmos_predictor = None
_utmos_available: Optional[bool] = None
_utmos_lock = threading.Lock()


def _get_utmos():
    global _utmos_predictor, _utmos_available
    if _utmos_available is not None:
        return _utmos_predictor
    with _utmos_lock:
        if _utmos_available is not None:  # re-check after acquiring lock
            return _utmos_predictor
        try:
            import torch

            # Microsoft's `speechmos` (installed for DNSMOS) is already in
            # sys.modules and shadows tarepan's speechmos.utmos22 that lives
            # inside the torch.hub cache directory.  Temporarily evict it so
            # torch.hub can resolve the correct submodule, then restore it.
            _saved = {
                k: sys.modules.pop(k) for k in list(sys.modules) if k == "speechmos" or k.startswith("speechmos.")
            }
            try:
                _utmos_predictor = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)
                _utmos_available = True
                logger.debug("UTMOS model loaded (torch.hub)")
            finally:
                for k in list(sys.modules):
                    if k == "speechmos" or k.startswith("speechmos."):
                        del sys.modules[k]
                sys.modules.update(_saved)
        except Exception as exc:
            logger.warning("UTMOS unavailable: %s", exc)
            _utmos_available = False
    return _utmos_predictor


def score_utmos(audio: np.ndarray, sr: int = SAMPLE_RATE) -> Dict[str, Optional[float]]:
    """Return UTMOS MOS score (1-5 scale) or None."""
    empty = {"utmos": None}
    predictor = _get_utmos()
    if predictor is None:
        return empty
    try:
        import torch

        wave = torch.from_numpy(audio).unsqueeze(0).float()
        score = predictor(wave, sr)
        return {"utmos": round(float(score.item()), 3)}
    except Exception as exc:
        logger.debug("UTMOS scoring failed: %s", exc)
        return empty


# ---------------------------------------------------------------------------
# SQUIM Objective  (torchaudio — reference-free PESQ, STOI, SI-SDR)
# ---------------------------------------------------------------------------

_squim_model = None
_squim_available: Optional[bool] = None
_squim_sr: int = SAMPLE_RATE
_squim_lock = threading.Lock()


def _get_squim():
    global _squim_model, _squim_available, _squim_sr
    if _squim_available is not None:
        return _squim_model
    with _squim_lock:
        if _squim_available is not None:
            return _squim_model
        try:
            from torchaudio.pipelines import SQUIM_OBJECTIVE

            _squim_model = SQUIM_OBJECTIVE.get_model()
            _squim_sr = SQUIM_OBJECTIVE.sample_rate
            _squim_available = True
            logger.debug("SQUIM Objective model loaded (torchaudio)")
        except Exception as exc:
            logger.warning("SQUIM unavailable: %s", exc)
            _squim_available = False
    return _squim_model


def score_squim(audio: np.ndarray, sr: int = SAMPLE_RATE) -> Dict[str, Optional[float]]:
    """Return reference-free PESQ, STOI, SI-SDR estimates or Nones."""
    empty = {"squim_pesq": None, "squim_stoi": None, "squim_si_sdr": None}
    model = _get_squim()
    if model is None:
        return empty
    try:
        import torch
        import torchaudio

        wave = torch.from_numpy(audio).unsqueeze(0).float()
        if sr != _squim_sr:
            wave = torchaudio.functional.resample(wave, sr, _squim_sr)

        stoi, pesq, si_sdr = model(wave)
        return {
            "squim_pesq": round(float(pesq.item()), 3),
            "squim_stoi": round(float(stoi.item()), 3),
            "squim_si_sdr": round(float(si_sdr.item()), 3),
        }
    except Exception as exc:
        logger.debug("SQUIM scoring failed: %s", exc)
        return empty


# ---------------------------------------------------------------------------
# Combined entry point
# ---------------------------------------------------------------------------

_EMPTY_MOS: Dict[str, "float | str | None"] = {
    "dnsmos_ovrl": None,
    "dnsmos_sig": None,
    "dnsmos_bak": None,
    "utmos": None,
    "squim_pesq": None,
    "squim_stoi": None,
    "squim_si_sdr": None,
    "mos_tier": None,
}


def compute_mos_metrics(
    audio_or_path,
    sr: int = SAMPLE_RATE,
) -> Dict[str, "float | str | None"]:
    """Compute all reference-free quality scores.

    Parameters
    ----------
    audio_or_path : np.ndarray | str
        Pre-loaded audio waveform (preferred) or a file path.  When a
        path is given the file is loaded with librosa — but callers
        that already have the waveform in memory should pass it
        directly to avoid redundant I/O.
    sr : int
        Sample rate of the waveform (ignored when a path is given
        because librosa resamples to ``SAMPLE_RATE``).

    Returns
    -------
    dict with keys: dnsmos_ovrl, dnsmos_sig, dnsmos_bak,
    utmos, squim_pesq, squim_stoi, squim_si_sdr, mos_tier.
    """
    if isinstance(audio_or_path, np.ndarray):
        audio = audio_or_path
    else:
        try:
            import librosa

            audio, _ = librosa.load(audio_or_path, sr=SAMPLE_RATE)
            sr = SAMPLE_RATE
        except Exception as exc:
            logger.debug("Failed to load audio for MOS metrics %s: %s", audio_or_path, exc)
            return dict(_EMPTY_MOS)

    scores: Dict[str, Optional[float]] = {}
    scores.update(score_dnsmos(audio, sr))
    scores.update(score_utmos(audio, sr))
    scores.update(score_squim(audio, sr))

    result: Dict[str, "float | str | None"] = dict(scores)
    result["mos_tier"] = assign_mos_tier(scores.get("dnsmos_ovrl"))
    return result
