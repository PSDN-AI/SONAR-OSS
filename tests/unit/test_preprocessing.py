"""Tests for the preprocessing package: audio utils, methods, config loader."""

import numpy as np
import pytest
import soundfile as sf

from psdn_sonar.loaders.manifest import ManifestEntry
from psdn_sonar.preprocessing.audio_utils import (
    get_audio_duration,
    get_combined_audio_path,
    parse_timestamp,
    trim_by_timestamps,
    trim_silence,
)
from psdn_sonar.preprocessing.config_loader import (
    DEFAULT_METHODS,
    DEFAULT_SETTINGS,
    load_multi_speaker_config,
)
from psdn_sonar.preprocessing.methods import (
    dual_assignment_score,
    preprocess_energy_trim,
    preprocess_no_trim,
)

SR = 16_000


def _write_wav(path, pieces):
    """Write a wav built from ("tone"|"silence", seconds) pieces."""
    chunks = []
    for kind, seconds in pieces:
        n = int(SR * seconds)
        if kind == "tone":
            t = np.linspace(0, seconds, n, endpoint=False)
            chunks.append((0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32))
        else:
            chunks.append(np.zeros(n, dtype=np.float32))
    sf.write(str(path), np.concatenate(chunks), SR)
    return path


class TestParseTimestamp:
    def test_hh_mm_ss(self):
        assert parse_timestamp("01:02:03") == 3723.0

    def test_mm_ss(self):
        assert parse_timestamp("02:30") == 150.0

    def test_float_seconds(self):
        assert parse_timestamp("12.5") == 12.5

    def test_invalid_raises(self):
        with pytest.raises(ValueError, match="Invalid timestamp"):
            parse_timestamp("1:2:3:4")


class TestAudioDurationAndTrimming:
    def test_get_audio_duration(self, tmp_path):
        wav = _write_wav(tmp_path / "a.wav", [("tone", 2.0)])
        assert get_audio_duration(wav) == pytest.approx(2.0, abs=0.05)

    def test_trim_silence_shortens_audio(self, tmp_path):
        wav = _write_wav(tmp_path / "a.wav", [("tone", 1.0), ("silence", 2.0), ("tone", 1.0)])
        out, original_s, trimmed_s = trim_silence(wav, output_path=tmp_path / "out.wav")
        assert original_s == pytest.approx(4.0, abs=0.05)
        assert trimmed_s < original_s
        assert get_audio_duration(out) == pytest.approx(trimmed_s, abs=0.05)

    def test_trim_silence_all_silent_returns_input(self, tmp_path):
        wav = _write_wav(tmp_path / "a.wav", [("silence", 1.0)])
        out, original_s, trimmed_s = trim_silence(wav)
        assert out == wav
        assert original_s == trimmed_s

    def test_trim_by_timestamps_extracts_speaker_segments(self, tmp_path):
        wav = _write_wav(tmp_path / "a.wav", [("tone", 4.0)])
        segments = [
            {"speaker": "speaker_a", "start": "00:00:00", "end": "00:00:01"},
            {"speaker": "speaker_b", "start": "00:00:01", "end": "00:00:03"},
            {"speaker": "SPEAKER_A", "start": "3.0", "end": "3.5"},
        ]
        out, original_s, trimmed_s = trim_by_timestamps(wav, segments, "A", output_path=tmp_path / "out.wav")
        assert original_s == pytest.approx(4.0, abs=0.05)
        # ~1.5s of speaker A audio plus padding, well under the original
        assert 1.0 < trimmed_s < 2.5

    def test_trim_by_timestamps_no_match_returns_input(self, tmp_path):
        wav = _write_wav(tmp_path / "a.wav", [("tone", 1.0)])
        out, original_s, trimmed_s = trim_by_timestamps(wav, [], "A")
        assert out == wav
        assert original_s == trimmed_s


class TestGetCombinedAudioPath:
    def _entry(self, tmp_path, audio_filepaths):
        return ManifestEntry(
            audio_id="conv_001",
            audio_filepaths=audio_filepaths,
            transcript_filepath="t.json",
            num_speakers=2,
            base_dir=tmp_path,
        )

    def test_found_next_to_speaker_audio(self, tmp_path):
        clip_dir = tmp_path / "data" / "conv_001"
        clip_dir.mkdir(parents=True)
        (clip_dir / "conv_001_Combined_Audio.wav").touch()
        entry = self._entry(tmp_path, {"speaker_a": "data/conv_001/a.wav"})
        assert get_combined_audio_path(entry) == clip_dir / "conv_001_Combined_Audio.wav"

    def test_missing_returns_none(self, tmp_path):
        entry = self._entry(tmp_path, {"speaker_a": "data/conv_001/a.wav"})
        assert get_combined_audio_path(entry) is None


class TestMethods:
    def test_no_trim_returns_duration(self, tmp_path):
        wav = _write_wav(tmp_path / "a.wav", [("tone", 1.0)])
        out, original_s, trimmed_s = preprocess_no_trim(wav)
        assert out == wav
        assert original_s == trimmed_s == pytest.approx(1.0, abs=0.05)

    def test_energy_trim_delegates(self, tmp_path):
        wav = _write_wav(tmp_path / "a.wav", [("tone", 0.8), ("silence", 1.5), ("tone", 0.8)])
        _, original_s, trimmed_s = preprocess_energy_trim(wav)
        assert trimmed_s < original_s


def _metric_exact(ref, hyp):
    """(cer, wer, similarity, poseidon): perfect on match, worst otherwise."""
    if ref == hyp:
        return 0.0, 0.0, 1.0, 1.0
    return 1.0, 1.0, 0.0, 0.0


class TestDualAssignmentScore:
    def test_no_speakers(self):
        a, b = dual_assignment_score({}, "ref a", "ref b", _metric_exact)
        assert a["error"] == b["error"] == "No speakers detected"

    def test_single_speaker_assigned_to_best_side(self):
        a, b = dual_assignment_score({"s0": "ref b"}, "ref a", "ref b", _metric_exact)
        assert "error" in a and a["text"] == ""
        assert b["text"] == "ref b" and b["similarity"] == 1.0

    def test_correct_assignment_kept(self):
        texts = {"s0": "ref a", "s1": "ref b"}
        a, b = dual_assignment_score(texts, "ref a", "ref b", _metric_exact)
        assert a["text"] == "ref a" and a["wer"] == 0.0
        assert b["text"] == "ref b" and b["wer"] == 0.0

    def test_swapped_assignment_corrected(self):
        texts = {"s0": "ref b", "s1": "ref a"}
        a, b = dual_assignment_score(texts, "ref a", "ref b", _metric_exact)
        assert a["text"] == "ref a"
        assert b["text"] == "ref b"

    def test_perfect_scores_beat_higher_similarity_with_errors(self):
        # Regression for issue #106: `or` fallbacks turned a perfect CER/WER
        # of 0.0 into worst-case 1.0, so a swapped pairing with real errors
        # but slightly higher cosine similarity outscored the correct one
        # and both speakers were charged the swapped pairing's error rates.
        table = {
            ("ref a", "hyp a"): (0.0, 0.0, 0.90),
            ("ref b", "hyp b"): (0.0, 0.0, 0.90),
            ("ref a", "hyp b"): (0.4, 0.4, 0.95),
            ("ref b", "hyp a"): (0.4, 0.4, 0.95),
        }

        def metric_fn(ref, hyp):
            c, w, s = table[(ref, hyp)]
            return (c, w, s, 0.0)

        a, b = dual_assignment_score({"s0": "hyp a", "s1": "hyp b"}, "ref a", "ref b", metric_fn)
        assert a["text"] == "hyp a" and a["cer"] == 0.0 and a["wer"] == 0.0
        assert b["text"] == "hyp b" and b["cer"] == 0.0 and b["wer"] == 0.0

    def test_none_metrics_still_count_as_worst_case(self):
        # None (metric unavailable) must keep its worst-case default: a
        # pairing with real scores beats one with missing scores.
        def metric_fn(ref, hyp):
            if (ref, hyp) in {("ref a", "hyp a"), ("ref b", "hyp b")}:
                return (0.2, 0.3, 0.8, 0.0)
            return (None, None, None, None)

        a, b = dual_assignment_score({"s0": "hyp a", "s1": "hyp b"}, "ref a", "ref b", metric_fn)
        assert a["text"] == "hyp a" and a["cer"] == 0.2
        assert b["text"] == "hyp b" and b["wer"] == 0.3

    def test_single_speaker_zero_similarity_assigned_to_a(self):
        # Legitimate 0.0 similarity on both sides: the tie goes to A and no
        # None-vs-0.0 confusion creeps in.
        def metric_fn(ref, hyp):
            return (0.5, 0.5, 0.0, 0.0)

        a, b = dual_assignment_score({"s0": "some text"}, "ref a", "ref b", metric_fn)
        assert a["text"] == "some text" and a["similarity"] == 0.0
        assert "error" in b


class TestLoadMultiSpeakerConfig:
    def test_default_package_config_loads(self):
        cfg = load_multi_speaker_config()
        assert cfg["methods"] == ["no_trim"]
        assert cfg["silence"]["max_silence_ms"] == 400

    def test_missing_file_uses_defaults(self, tmp_path):
        cfg = load_multi_speaker_config(str(tmp_path / "nope.yaml"))
        assert cfg["methods"] == DEFAULT_METHODS
        assert cfg["pyannote"] == DEFAULT_SETTINGS["pyannote"]

    def test_unknown_methods_skipped(self, tmp_path):
        p = tmp_path / "c.yaml"
        p.write_text("methods:\n  - no_trim\n  - bogus_method\n")
        assert load_multi_speaker_config(str(p))["methods"] == ["no_trim"]

    def test_all_unknown_falls_back_to_defaults(self, tmp_path):
        p = tmp_path / "c.yaml"
        p.write_text("methods:\n  - bogus\n")
        assert load_multi_speaker_config(str(p))["methods"] == DEFAULT_METHODS

    def test_settings_merge_with_defaults(self, tmp_path):
        p = tmp_path / "c.yaml"
        p.write_text("methods:\n  - energy_trim\nsilence:\n  silence_thresh: -35\n")
        cfg = load_multi_speaker_config(str(p))
        assert cfg["silence"]["silence_thresh"] == -35
        assert cfg["silence"]["max_silence_ms"] == 400  # default preserved
