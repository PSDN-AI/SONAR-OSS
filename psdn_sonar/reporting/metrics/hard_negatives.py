"""Hard-negative statistics from an evaluation results CSV."""

from typing import Dict

import pandas as pd


def calculate_hard_negatives(results_csv: str, percentile_threshold: float = 0.75) -> Dict:
    """Compare overall vs. hard-negative mean WER/CER for a results CSV.

    Hard negatives are rows at or above the *percentile_threshold* quantile of
    either metric. The first columns whose names contain "wer" and "cer" are
    used; raises ``ValueError`` when they cannot be found.
    """
    df = pd.read_csv(results_csv)

    wer_col = None
    cer_col = None

    for col in df.columns:
        if "wer" in col.lower() and wer_col is None:
            wer_col = col
        if "cer" in col.lower() and cer_col is None:
            cer_col = col

    if not wer_col or not cer_col:
        raise ValueError(f"Could not find WER/CER columns in {results_csv}")

    df_clean = df[df[wer_col].notna() & df[cer_col].notna()].copy()

    wer_threshold = df_clean[wer_col].quantile(percentile_threshold)
    cer_threshold = df_clean[cer_col].quantile(percentile_threshold)

    hard_negatives = df_clean[(df_clean[wer_col] >= wer_threshold) | (df_clean[cer_col] >= cer_threshold)]

    return {
        "wer": {
            "overall": df_clean[wer_col].mean(),
            "hard": hard_negatives[wer_col].mean() if len(hard_negatives) > 0 else 0.0,
        },
        "cer": {
            "overall": df_clean[cer_col].mean(),
            "hard": hard_negatives[cer_col].mean() if len(hard_negatives) > 0 else 0.0,
        },
    }
