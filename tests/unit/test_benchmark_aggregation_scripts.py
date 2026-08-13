"""Tests for scripts/extract_benchmarks.py and scripts/build_macro_summary.py."""

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

SCRIPTS_DIR = Path(__file__).parent.parent.parent / "scripts"


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def extract():
    return _load_script("extract_benchmarks")


@pytest.fixture(scope="module")
def macro():
    return _load_script("build_macro_summary")


def _write_results(path: Path, cer=(0.1, 0.2), wer=(0.3, 0.4)):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "cer_conv": cer,
            "wer_conv": wer,
            "semantic_similarity_conv": [0.9] * len(cer),
        }
    ).to_csv(path, index=False)


class TestSummarizeCsv:
    def test_aggregates_means_and_stds(self, extract, tmp_path):
        csv = tmp_path / "whisper_api" / "fleurs.csv"
        _write_results(csv)

        row = extract.summarize_csv(csv, model="whisper_api", dataset="fleurs")

        assert row["model"] == "whisper_api"
        assert row["dataset"] == "fleurs"
        assert row["n_samples"] == 2
        assert row["cer_conv_mean"] == pytest.approx(0.15)
        assert row["wer_conv_mean"] == pytest.approx(0.35)
        assert row["poseidon_score_mean"] is not None

    def test_single_row_std_is_zero(self, extract, tmp_path):
        csv = tmp_path / "m" / "d.csv"
        _write_results(csv, cer=(0.1,), wer=(0.3,))

        row = extract.summarize_csv(csv, model="m", dataset="d")

        assert row["cer_conv_std"] == 0.0

    def test_no_known_metrics_returns_empty(self, extract, tmp_path):
        csv = tmp_path / "m" / "d.csv"
        csv.parent.mkdir(parents=True)
        pd.DataFrame({"other": [1, 2]}).to_csv(csv, index=False)

        assert extract.summarize_csv(csv, model="m", dataset="d") == {}

    def test_unreadable_csv_returns_empty(self, extract, tmp_path):
        assert extract.summarize_csv(tmp_path / "missing.csv", model="m", dataset="d") == {}


class TestExtractBenchmarks:
    def test_scans_model_dataset_layout(self, extract, tmp_path):
        _write_results(tmp_path / "whisper_api" / "fleurs.csv")
        _write_results(tmp_path / "whisper_api" / "commonvoice.csv")
        _write_results(tmp_path / "assemblyai_api" / "fleurs.csv")

        summary = extract.extract_benchmarks(tmp_path)

        assert len(summary) == 3
        assert set(summary["model"]) == {"whisper_api", "assemblyai_api"}
        assert set(summary["dataset"]) == {"fleurs", "commonvoice"}

    def test_empty_dir_returns_empty_frame(self, extract, tmp_path):
        assert extract.extract_benchmarks(tmp_path).empty


@pytest.fixture
def benchmarks_csv(tmp_path):
    csv = tmp_path / "public_benchmarks.csv"
    csv.write_text(
        "\n".join(
            [
                "model,dataset,cer_conv_mean,wer_conv_mean",
                "whisper-1,fleurs,0.10,0.30",
                "assemblyai,fleurs,0.12,0.35",
                "whisper-1,openslr_bd,0.20,0.40",
                "assemblyai,openslr_bd,0.22,0.42",
                "whisper-1,commonvoice,0.30,0.50",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return csv


_METRIC_FLAGS = ["--metric", "cer_conv_mean", "--metric", "wer_conv_mean"]


class TestBuildMacroSummaryCli:
    def test_csv_writes_macro_summary_per_model(self, macro, tmp_path, benchmarks_csv):
        out = tmp_path / "macro.json"

        rc = macro.main(["--input", str(benchmarks_csv), "--output", str(out), *_METRIC_FLAGS])

        assert rc == 0
        payload = json.loads(out.read_text(encoding="utf-8"))
        assert set(payload) == {"whisper-1", "assemblyai"}
        assert payload["whisper-1"]["macro"]["cer_conv_mean"] == pytest.approx(0.20)
        assert payload["whisper-1"]["n_locales"] == 3
        assert payload["assemblyai"]["macro"]["wer_conv_mean"] == pytest.approx((0.35 + 0.42) / 2)
        assert payload["assemblyai"]["n_locales"] == 2

    def test_json_input_passthrough(self, macro, tmp_path):
        src = tmp_path / "per_locale.json"
        src.write_text(
            json.dumps(
                {
                    "fleurs": {"whisper-1": {"wer": 0.3}, "assemblyai": {"wer": 0.35}},
                    "openslr_bd": {"whisper-1": {"wer": 0.4}, "assemblyai": {"wer": 0.42}},
                }
            ),
            encoding="utf-8",
        )
        out = tmp_path / "macro.json"

        rc = macro.main(["--input", str(src), "--output", str(out)])

        assert rc == 0
        payload = json.loads(out.read_text(encoding="utf-8"))
        assert payload["whisper-1"]["macro"]["wer"] == pytest.approx(0.35)

    def test_missing_input_exits(self, macro, tmp_path):
        with pytest.raises(SystemExit):
            macro.main(["--input", str(tmp_path / "nope.csv")])

    def test_missing_columns_exits(self, macro, tmp_path):
        bad = tmp_path / "bad.csv"
        bad.write_text("model,dataset\nfoo,fleurs\n", encoding="utf-8")

        with pytest.raises(SystemExit, match="missing required columns"):
            macro.main(["--input", str(bad)])


@pytest.fixture
def csv_with_duplicates(tmp_path):
    csv = tmp_path / "duplicates.csv"
    csv.write_text(
        "\n".join(
            [
                "model,dataset,cer_conv_mean,wer_conv_mean",
                "whisper-1,fleurs,0.10,0.30",
                "whisper-1,fleurs,0.99,0.99",
                "whisper-1,openslr_bd,0.20,0.40",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return csv


class TestDuplicateHandling:
    def test_duplicates_error_by_default(self, macro, csv_with_duplicates):
        with pytest.raises(SystemExit) as exc_info:
            macro.main(["--input", str(csv_with_duplicates), *_METRIC_FLAGS])

        msg = str(exc_info.value)
        assert "duplicate" in msg.lower()
        assert "fleurs" in msg
        assert "whisper-1" in msg

    def test_allow_duplicates_warns_last_row_wins(self, macro, tmp_path, csv_with_duplicates, caplog):
        out = tmp_path / "macro.json"

        with caplog.at_level("WARNING"):
            rc = macro.main(
                ["--input", str(csv_with_duplicates), "--output", str(out), "--allow-duplicates", *_METRIC_FLAGS]
            )

        assert rc == 0
        payload = json.loads(out.read_text(encoding="utf-8"))
        assert payload["whisper-1"]["per_locale"]["fleurs"]["cer_conv_mean"] == pytest.approx(0.99)
        assert any("duplicate" in r.message.lower() for r in caplog.records)
