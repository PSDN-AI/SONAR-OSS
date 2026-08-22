"""Tests for the ``psdn-sonar leaderboard`` command and its aggregation.

Issue #117 established the ground rule these tests enforce: a leaderboard may
only show numbers that were measured by real runs. Metrics absent from the
run artifacts must stay absent — never derived, back-solved, or substituted.
"""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from psdn_sonar.benchmark.leaderboard import (
    LoadedRun,
    build_leaderboard,
    collect_scores,
    render_leaderboard,
    rows_as_json,
)
from psdn_sonar.benchmark.scores import (
    RunAggregate,
    RunScoresArtifact,
    scores_json_path,
    write_scores_json,
)
from psdn_sonar.benchmark.submission import SubmissionConfig
from psdn_sonar.cli import main


def run_cli(*argv):
    with patch("sys.argv", ["psdn-sonar", *argv]):
        main()


def make_artifact(
    model="wav2vec2_bengali",
    language="bn",
    wer=0.3,
    cer=0.15,
    semantic=0.9,
    poseidon=0.8,
    successful=10,
    failed=0,
    warnings=(),
):
    submission = SubmissionConfig(
        provider="local",
        model_snapshot=model,
        region="local",
        protocol="batch",
        inference_params={"language_code": language} if language else {},
        seed=42,
        git_sha="testsha",
        package_version="0.0.0-test",
        timestamp_utc="2026-08-22T00:00:00Z",
    )
    aggregate = RunAggregate(
        wer_mean=wer,
        cer_mean=cer,
        semantic_similarity_mean=semantic,
        poseidon_score_mean=poseidon,
        total_samples=successful + failed,
        successful=successful,
        failed=failed,
        elapsed_time_s=1.0,
    )
    return RunScoresArtifact(
        submission=submission,
        model_name=model,
        aggregate=aggregate,
        warnings=list(warnings),
    )


def write_artifact(directory, artifact, filename=None):
    path = scores_json_path(directory, artifact.model_name) if filename is None else directory / filename
    return write_scores_json(path, artifact)


def loaded(*artifacts):
    return [LoadedRun(path=Path("unused.json"), artifact=a) for a in artifacts]


class TestCollectScores:
    def test_finds_artifacts_recursively(self, tmp_path):
        write_artifact(tmp_path / "run1", make_artifact(model="model_a"))
        write_artifact(tmp_path / "nested" / "run2", make_artifact(model="model_b"))
        runs, skipped = collect_scores([tmp_path])
        assert sorted(r.artifact.model_name for r in runs) == ["model_a", "model_b"]
        assert skipped == []

    def test_skips_invalid_json_with_message(self, tmp_path):
        (tmp_path / "scores_broken.json").write_text("{not json", encoding="utf-8")
        write_artifact(tmp_path, make_artifact(model="model_a"))
        runs, skipped = collect_scores([tmp_path])
        assert [r.artifact.model_name for r in runs] == ["model_a"]
        assert len(skipped) == 1
        assert "scores_broken.json" in skipped[0]

    def test_skips_json_that_is_not_a_scores_artifact(self, tmp_path):
        (tmp_path / "scores_other.json").write_text('{"something": "else"}', encoding="utf-8")
        runs, skipped = collect_scores([tmp_path])
        assert runs == []
        assert len(skipped) == 1

    def test_empty_directory(self, tmp_path):
        assert collect_scores([tmp_path]) == ([], [])


class TestBuildLeaderboard:
    def test_averages_multiple_runs_of_one_model(self):
        rows = build_leaderboard(
            loaded(
                make_artifact(wer=0.2, cer=0.1, successful=10, failed=1),
                make_artifact(wer=0.4, cer=0.3, successful=5, failed=0),
            )
        )
        assert len(rows) == 1
        row = rows[0]
        assert row.runs == 2
        assert row.wer == pytest.approx(0.3)
        assert row.cer == pytest.approx(0.2)
        assert row.successful == 15
        assert row.failed == 1

    def test_missing_metrics_stay_missing_never_derived(self):
        """The core #117 rule: a WER without a measured CER must not conjure
        a CER (the old site published CER = 0.5 x WER by construction)."""
        rows = build_leaderboard(loaded(make_artifact(wer=0.4, cer=None, semantic=None, poseidon=None)))
        row = rows[0]
        assert row.wer == pytest.approx(0.4)
        assert row.cer is None
        assert row.semantic is None
        assert row.poseidon is None

    def test_metric_average_ignores_runs_that_did_not_measure_it(self):
        rows = build_leaderboard(
            loaded(
                make_artifact(semantic=0.9, poseidon=0.8),
                make_artifact(semantic=None, poseidon=None),
            )
        )
        row = rows[0]
        assert row.semantic == pytest.approx(0.9)
        assert row.poseidon == pytest.approx(0.8)

    def test_groups_same_model_per_language(self):
        rows = build_leaderboard(
            loaded(
                make_artifact(model="whisper_api", language="bn"),
                make_artifact(model="whisper_api", language="hi"),
            )
        )
        assert {(r.model_name, r.language) for r in rows} == {("whisper_api", "bn"), ("whisper_api", "hi")}

    def test_language_filter(self):
        rows = build_leaderboard(
            loaded(
                make_artifact(model="model_bn", language="bn"),
                make_artifact(model="model_hi", language="hi"),
            ),
            language="hi",
        )
        assert [r.model_name for r in rows] == ["model_hi"]

    def test_warnings_flagged(self):
        rows = build_leaderboard(
            loaded(
                make_artifact(model="suspect", warnings=["reference script contradicts --language"]),
                make_artifact(model="clean"),
            )
        )
        by_name = {r.model_name: r for r in rows}
        assert by_name["suspect"].has_warnings is True
        assert by_name["clean"].has_warnings is False

    def test_default_sort_poseidon_desc_missing_last(self):
        rows = build_leaderboard(
            loaded(
                make_artifact(model="mid", poseidon=0.5),
                make_artifact(model="best", poseidon=0.9),
                make_artifact(model="unmeasured", poseidon=None),
            )
        )
        assert [r.model_name for r in rows] == ["best", "mid", "unmeasured"]

    def test_sort_by_wer_asc_missing_last(self):
        rows = build_leaderboard(
            loaded(
                make_artifact(model="worse", wer=0.5),
                make_artifact(model="better", wer=0.1),
                make_artifact(model="unmeasured", wer=None),
            ),
            sort="wer",
        )
        assert [r.model_name for r in rows] == ["better", "worse", "unmeasured"]

    def test_unknown_sort_key_raises(self):
        with pytest.raises(ValueError, match="Unknown sort key"):
            build_leaderboard(loaded(make_artifact()), sort="vibes")


class TestRenderLeaderboard:
    def test_missing_metric_renders_as_dash(self):
        rows = build_leaderboard(loaded(make_artifact(cer=None, semantic=None, poseidon=None)))
        table = render_leaderboard(rows)
        assert "—" in table
        assert "never derived" in table

    def test_warning_marker_and_footnote(self):
        rows = build_leaderboard(loaded(make_artifact(model="suspect", warnings=["script mismatch"])))
        table = render_leaderboard(rows)
        assert "suspect !" in table
        assert "warnings" in table

    def test_no_warning_footnote_for_clean_rows(self):
        rows = build_leaderboard(loaded(make_artifact()))
        table = render_leaderboard(rows)
        assert "! =" not in table

    def test_json_rendering_round_trips(self):
        rows = build_leaderboard(loaded(make_artifact(model="model_a", poseidon=0.8)))
        payload = json.loads(rows_as_json(rows))
        assert payload[0]["model_name"] == "model_a"
        assert payload[0]["poseidon"] == pytest.approx(0.8)
        assert payload[0]["cer"] == pytest.approx(0.15)


class TestLeaderboardCLI:
    def test_happy_path_prints_table(self, tmp_path, capsys):
        write_artifact(tmp_path, make_artifact(model="model_a", poseidon=0.9))
        write_artifact(tmp_path / "sub", make_artifact(model="model_b", poseidon=0.5))
        run_cli("leaderboard", "--runs", str(tmp_path))
        out = capsys.readouterr().out
        assert "model_a" in out
        assert "model_b" in out
        assert out.index("model_a") < out.index("model_b")  # POSEIDON desc

    def test_json_output(self, tmp_path, capsys):
        write_artifact(tmp_path, make_artifact(model="model_a"))
        run_cli("leaderboard", "--runs", str(tmp_path), "--json")
        payload = json.loads(capsys.readouterr().out)
        assert payload[0]["model_name"] == "model_a"

    def test_empty_directory_exits_one_with_actionable_error(self, tmp_path, caplog):
        with caplog.at_level("ERROR"):
            with pytest.raises(SystemExit) as exc_info:
                run_cli("leaderboard", "--runs", str(tmp_path))
        assert exc_info.value.code == 1
        assert "No scores_*.json artifacts found" in caplog.text
        assert "--output" in caplog.text

    def test_nonexistent_directory_exits_one(self, tmp_path, caplog):
        with caplog.at_level("ERROR"):
            with pytest.raises(SystemExit) as exc_info:
                run_cli("leaderboard", "--runs", str(tmp_path / "nope"))
        assert exc_info.value.code == 1
        assert "Not a directory" in caplog.text

    def test_language_filter_excluding_everything_names_present_languages(self, tmp_path, caplog):
        write_artifact(tmp_path, make_artifact(language="bn"))
        with caplog.at_level("ERROR"):
            with pytest.raises(SystemExit) as exc_info:
                run_cli("leaderboard", "--runs", str(tmp_path), "--language", "ko")
        assert exc_info.value.code == 1
        assert "Languages present: bn" in caplog.text

    def test_malformed_artifact_warns_but_run_succeeds(self, tmp_path, caplog, capsys):
        write_artifact(tmp_path, make_artifact(model="model_a"))
        (tmp_path / "scores_broken.json").write_text("{oops", encoding="utf-8")
        with caplog.at_level("WARNING"):
            run_cli("leaderboard", "--runs", str(tmp_path))
        assert "scores_broken.json" in caplog.text
        assert "model_a" in capsys.readouterr().out
