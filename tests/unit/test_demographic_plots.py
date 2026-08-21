"""Tests for demographic plot generation and the Poseidon score helper."""

import json

import numpy as np
import pandas as pd

from psdn_sonar.reporting.plots.demographic import (
    create_age_groups,
    generate_demographic_plots,
    load_all_multispeaker_with_metadata,
)
from psdn_sonar.utils.metrics import ensure_poseidon_score

rng = np.random.default_rng(7)


def _build_dataset(tmp_path, n=12):
    """Create metadata folders plus one per-model results CSV; returns (results_dir, dataset_dir)."""
    dataset_dir = tmp_path / "dataset"
    results_dir = tmp_path / "results"
    results_dir.mkdir()

    genders = ["Female", "Male"]
    regions = ["North", "South"]
    rows = []
    for i in range(n):
        audio_id = f"conv_{i:03d}"
        meta_dir = dataset_dir / audio_id
        meta_dir.mkdir(parents=True)
        metadata = {
            "speaker_a": {"age": 19 + i, "gender": genders[i % 2], "region": regions[i % 2]},
            "speaker_b": {"age": 22 + i, "gender": genders[(i + 1) % 2], "region": regions[(i + 1) % 2]},
        }
        (meta_dir / "metadata.json").write_text(json.dumps(metadata))
        rows.append({"audio_id": audio_id, "speaker": "A", "wer_conv": float(rng.uniform(0.05, 0.6))})

    pd.DataFrame(rows).to_csv(results_dir / "results_test_model_manifest.csv", index=False)
    return str(results_dir), str(dataset_dir)


class TestEnsurePoseidonScore:
    def test_derives_score_from_conv_columns(self):
        df = pd.DataFrame({"wer_conv": [0.2], "cer_conv": [0.1], "semantic_similarity_conv": [0.9]})
        out = ensure_poseidon_score(df)
        assert 0.0 <= out["poseidon_score"].iloc[0] <= 1.0

    def test_nan_metrics_excluded_not_substituted(self):
        # Issue #107: an uncomputable metric is excluded (NaN poseidon that
        # aggregations skip), never scored as the worst case.
        df = pd.DataFrame({"wer_conv": [np.nan], "cer_conv": [np.nan], "semantic_similarity_conv": [np.nan]})
        out = ensure_poseidon_score(df)
        assert np.isnan(out["poseidon_score"].iloc[0])

    def test_single_missing_metric_excludes_row(self):
        df = pd.DataFrame({"wer_conv": [0.2], "cer_conv": [0.1], "semantic_similarity_conv": [np.nan]})
        assert np.isnan(ensure_poseidon_score(df)["poseidon_score"].iloc[0])

    def test_mixed_rows_scored_independently(self):
        df = pd.DataFrame(
            {
                "wer_conv": [0.2, np.nan],
                "cer_conv": [0.1, 0.1],
                "semantic_similarity_conv": [0.9, 0.9],
            }
        )
        out = ensure_poseidon_score(df)
        assert 0.0 <= out["poseidon_score"].iloc[0] <= 1.0
        assert np.isnan(out["poseidon_score"].iloc[1])

    def test_existing_column_untouched(self):
        df = pd.DataFrame({"poseidon_score": [0.42]})
        assert ensure_poseidon_score(df)["poseidon_score"].iloc[0] == 0.42

    def test_missing_sources_returns_unchanged(self):
        df = pd.DataFrame({"wer_conv": [0.2]})
        assert "poseidon_score" not in ensure_poseidon_score(df).columns


class TestCreateAgeGroups:
    def test_bins_and_labels(self):
        df = pd.DataFrame({"age": [18, 22, 25, 40]})
        out = create_age_groups(df)
        assert list(out["age_group"].astype(str)) == ["≤20", "21-23", "24-26", ">26"]


class TestLoadMultispeakerWithMetadata:
    def test_joins_metadata_for_speaker(self, tmp_path):
        results_dir, dataset_dir = _build_dataset(tmp_path, n=4)
        df = load_all_multispeaker_with_metadata(results_dir, dataset_dir)
        assert len(df) == 4
        assert {"age", "gender", "region", "model", "model_label"}.issubset(df.columns)
        assert df["model"].iloc[0] == "test_model"
        assert df["model_label"].iloc[0] == "Test Model"

    def test_missing_metadata_yields_nan(self, tmp_path):
        results_dir, dataset_dir = _build_dataset(tmp_path, n=2)
        df_rows = pd.DataFrame([{"audio_id": "conv_999", "speaker": "A", "wer_conv": 0.3}])
        df_rows.to_csv(f"{results_dir}/results_other_model_manifest.csv", index=False)
        df = load_all_multispeaker_with_metadata(results_dir, dataset_dir)
        other = df[df["model"] == "other_model"]
        assert other["gender"].isna().all()

    def test_empty_results_dir(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        assert load_all_multispeaker_with_metadata(str(empty), str(empty)).empty


class TestGenerateDemographicPlots:
    def test_generates_boxplots_and_summary(self, tmp_path):
        results_dir, dataset_dir = _build_dataset(tmp_path)
        out = tmp_path / "plots"
        generate_demographic_plots(results_dir, dataset_dir, str(out))
        assert (out / "gender_wer_conv.png").exists()
        assert (out / "age_wer_conv.png").exists()
        assert (out / "region_wer_conv.png").exists()
        summary = pd.read_csv(out / "demographic_summary_statistics.csv")
        assert set(summary["demographic"]) == {"Gender", "Age Group", "Region"}

    def test_no_data_no_output(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        out = tmp_path / "plots"
        generate_demographic_plots(str(empty), str(empty), str(out))
        assert not any(out.glob("*.png"))
