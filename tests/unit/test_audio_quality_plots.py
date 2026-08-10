"""Tests for audio-quality plot generation and the shared results loader."""

import json

import numpy as np
import pandas as pd

from psdn_sonar.reporting.plots._common import load_and_tag_results, prettify_model_name
from psdn_sonar.reporting.plots.audio_quality import compute_quality_summary, generate_audio_quality_plots

rng = np.random.default_rng(42)


def _make_results_csv(tmp_path, name="results.csv", n=30, **extra_cols):
    df = pd.DataFrame(
        {
            "snr_db": rng.uniform(5, 30, n),
            "snr_tier": rng.choice(["Low", "Medium", "High"], n),
            "wer": rng.uniform(0.0, 0.8, n),
            "clipping_ratio": rng.uniform(0.0, 0.005, n),
            "silence_ratio": rng.uniform(0.1, 0.5, n),
            **extra_cols,
        }
    )
    path = tmp_path / name
    df.to_csv(path, index=False)
    return str(path)


class TestPrettifyModelName:
    def test_api_names(self):
        assert prettify_model_name("whisper_api") == "Whisper API"
        assert prettify_model_name("elevenlabs_api") == "ElevenLabs API"

    def test_custom_prefix_joins_org_and_model(self):
        assert prettify_model_name("custom_openai_whisper-large") == "openai/whisper-large"

    def test_fallback_title_case(self):
        assert prettify_model_name("my_test-model") == "My Test Model"


class TestLoadAndTagResults:
    def test_conv_columns_normalised(self, tmp_path):
        df = pd.DataFrame({"wer_conv": [0.1, 0.2], "cer_conv": [0.05, 0.1]})
        path = tmp_path / "r.csv"
        df.to_csv(path, index=False)
        combined = load_and_tag_results([("m", str(path))])
        assert list(combined["wer"]) == [0.1, 0.2]
        assert list(combined["cer"]) == [0.05, 0.1]

    def test_unreadable_csv_skipped(self, tmp_path):
        combined = load_and_tag_results([("m", str(tmp_path / "missing.csv"))])
        assert combined.empty

    def test_non_numeric_coerced_to_nan(self, tmp_path):
        df = pd.DataFrame({"wer": ["0.1", "bad"], "snr_db": ["10", "20"]})
        path = tmp_path / "r.csv"
        df.to_csv(path, index=False)
        combined = load_and_tag_results([("m", str(path))])
        assert combined["wer"].isna().sum() == 1


class TestQualitySummary:
    def test_summary_written_as_json(self, tmp_path):
        df = pd.DataFrame(
            {
                "snr_db": [15.0, 25.0, 5.0],
                "clipping_ratio": [0.0, 0.02, 0.0],
                "silence_ratio": [0.2, 0.3, 0.9],
            }
        )
        out = tmp_path / "summary.json"
        summary = compute_quality_summary(df, str(out))
        assert summary["total_utterances"] == 3
        assert json.loads(out.read_text()) == summary

    def test_clipped_percentage(self, tmp_path):
        df = pd.DataFrame({"snr_db": [15.0, 15.0], "clipping_ratio": [0.02, 0.0], "silence_ratio": [0.1, 0.1]})
        summary = compute_quality_summary(df, str(tmp_path / "s.json"))
        assert summary["pct_clipped"] == 50.0


class TestGenerateAudioQualityPlots:
    def test_default_outputs(self, tmp_path):
        csv = _make_results_csv(tmp_path)
        out = tmp_path / "plots"
        generate_audio_quality_plots([("test_model", csv)], str(out))
        assert (out / "snr_vs_wer_scatter.png").exists()
        assert (out / "snr_distribution.png").exists()
        assert (out / "quality_summary.json").exists()
        assert not (out / "wer_by_snr_tier.png").exists()

    def test_tier_plots_opt_in(self, tmp_path):
        csv = _make_results_csv(tmp_path)
        out = tmp_path / "plots"
        generate_audio_quality_plots([("test_model", csv)], str(out), include_tier_plots=True)
        assert (out / "wer_by_snr_tier.png").exists()
        assert (out / "quality_tier_composition.png").exists()
        assert (out / "model_tier_heatmap.png").exists()

    def test_missing_columns_skips_everything(self, tmp_path):
        df = pd.DataFrame({"wer": [0.1, 0.2]})
        path = tmp_path / "r.csv"
        df.to_csv(path, index=False)
        out = tmp_path / "plots"
        generate_audio_quality_plots([("m", str(path))], str(out))
        assert not (out / "snr_vs_wer_scatter.png").exists()
