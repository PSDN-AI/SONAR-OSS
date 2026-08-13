"""Tests for scripts/snr_vs_wer.py."""

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

SCRIPTS_DIR = Path(__file__).parent.parent.parent / "scripts"


@pytest.fixture(scope="module")
def script():
    spec = importlib.util.spec_from_file_location("snr_vs_wer", SCRIPTS_DIR / "snr_vs_wer.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestParseCsvSpecs:
    def test_parses_model_path_pairs(self, script):
        result = script.parse_csv_specs(["whisper_api=a.csv", "assemblyai_api=b.csv"])

        assert result == {"whisper_api": Path("a.csv"), "assemblyai_api": Path("b.csv")}

    def test_rejects_missing_separator(self, script):
        with pytest.raises(ValueError, match="MODEL=PATH"):
            script.parse_csv_specs(["just_a_path.csv"])


class TestAddQualityMetrics:
    def test_metrics_added_for_existing_audio(self, script, tmp_path, monkeypatch):
        (tmp_path / "a.wav").write_bytes(b"fake")
        monkeypatch.setattr(
            script,
            "compute_audio_quality_metrics",
            lambda path, include_mos: {"snr_db": 20.0, "clipping_ratio": 0.0, "silence_ratio": 0.1, "snr_tier": "high"},
        )
        df = pd.DataFrame({"path": ["a.wav"], "wer_conv": [0.2]})

        result = script.add_quality_metrics(df, tmp_path)

        assert result["snr_db"].iloc[0] == 20.0
        assert result["snr_tier"].iloc[0] == "high"
        assert result["wer"].iloc[0] == 0.2

    def test_missing_audio_gets_nan(self, script, tmp_path):
        df = pd.DataFrame({"audio_path": ["missing.wav"]})

        result = script.add_quality_metrics(df, tmp_path)

        assert pd.isna(result["snr_db"].iloc[0])
        assert result["snr_tier"].iloc[0] is None

    def test_failed_computation_gets_nan(self, script, tmp_path, monkeypatch):
        (tmp_path / "a.wav").write_bytes(b"fake")

        def boom(path, include_mos):
            raise RuntimeError("corrupt file")

        monkeypatch.setattr(script, "compute_audio_quality_metrics", boom)
        df = pd.DataFrame({"path": ["a.wav"]})

        result = script.add_quality_metrics(df, tmp_path)

        assert pd.isna(result["snr_db"].iloc[0])

    def test_absolute_paths_used_verbatim(self, script, tmp_path, monkeypatch):
        audio = tmp_path / "abs.wav"
        audio.write_bytes(b"fake")
        seen = []

        def record(path, include_mos):
            seen.append(path)
            return {"snr_db": 1.0, "clipping_ratio": 0.0, "silence_ratio": 0.0, "snr_tier": "low"}

        monkeypatch.setattr(script, "compute_audio_quality_metrics", record)
        df = pd.DataFrame({"path": [str(audio)]})

        script.add_quality_metrics(df, tmp_path / "elsewhere")

        assert seen == [str(audio)]

    def test_no_audio_column_raises(self, script, tmp_path):
        df = pd.DataFrame({"other": [1]})

        with pytest.raises(ValueError, match="No audio column"):
            script.add_quality_metrics(df, tmp_path)
