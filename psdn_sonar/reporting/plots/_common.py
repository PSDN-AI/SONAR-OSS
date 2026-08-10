"""Shared helpers for reporting plot modules."""

import logging
from typing import List, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

_API_MODEL_NAMES = {
    "whisper_api": "Whisper API",
    "elevenlabs_api": "ElevenLabs API",
    "assemblyai_api": "AssemblyAI API",
}

_NUMERIC_COLUMNS = ("snr_db", "wer", "cer", "clipping_ratio", "silence_ratio", "inference_latency_s")


def prettify_model_name(raw: str) -> str:
    """Turn CSV-derived model names into readable labels."""
    if raw in _API_MODEL_NAMES:
        return _API_MODEL_NAMES[raw]
    if raw.startswith("custom_"):
        parts = raw[len("custom_") :].split("_")
        return "/".join(parts[:2]) if len(parts) >= 2 else raw
    return raw.replace("_", " ").replace("-", " ").title()


def load_and_tag_results(results_csvs: List[Tuple[str, str]]) -> pd.DataFrame:
    """Load every result CSV, tag with a *model* column, and concatenate.

    Multi-speaker CSVs (``wer_conv`` / ``cer_conv``) are normalised so
    downstream plot functions always see ``wer`` and ``cer``. Known numeric
    columns are coerced, with unparseable values becoming NaN.
    """
    frames = []
    for model_name, csv_path in results_csvs:
        try:
            df = pd.read_csv(csv_path)
            df["model"] = prettify_model_name(model_name)
            if "wer" not in df.columns and "wer_conv" in df.columns:
                df["wer"] = df["wer_conv"]
            if "cer" not in df.columns and "cer_conv" in df.columns:
                df["cer"] = df["cer_conv"]
            frames.append(df)
        except Exception as exc:
            logger.warning("Could not load %s: %s", csv_path, exc)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    for col in _NUMERIC_COLUMNS:
        if col in combined.columns:
            combined[col] = pd.to_numeric(combined[col], errors="coerce")
    return combined
