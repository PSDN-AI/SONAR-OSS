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


def _write_freq_wav(path, pieces):
    """Write a wav from ``(freq_hz_or_0_for_silence, seconds)`` pieces."""
    chunks = []
    for freq, seconds in pieces:
        n = int(SR * seconds)
        if freq:
            t = np.linspace(0, seconds, n, endpoint=False)
            chunks.append((0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32))
        else:
            chunks.append(np.zeros(n, dtype=np.float32))
    sf.write(str(path), np.concatenate(chunks), SR)
    return path


def _dominant_freq(path) -> float:
    audio, sr = sf.read(str(path))
    spectrum = np.abs(np.fft.rfft(audio))
    return float(np.fft.rfftfreq(audio.size, 1 / sr)[int(np.argmax(spectrum))])


class TestTimestampTrimCombinedTimeline:
    """Issue #205: segment offsets are combined-timeline, but they were
    clamped against each speaker's own channel file. For the speaker who
    talks second, the start lay past the end of their file, every segment
    was dropped, and 100 ms of padding was exported and scored as their turn
    — silently, with exit 0. Speakers A and B carry different tone
    frequencies here so the tests can assert whose audio actually ends up in
    the trimmed output."""

    def _fixture(self, tmp_path):
        # The conversation: A speaks (440 Hz) for 2.0 s, a 0.3 s gap, then B
        # speaks (880 Hz) for 1.5 s. Channel files hold only each speaker's
        # own turn — the shipped fixtures' layout.
        combined = _write_freq_wav(tmp_path / "combined.wav", [(440, 2.0), (0, 0.3), (880, 1.5)])
        channel_a = _write_freq_wav(tmp_path / "speaker_a.wav", [(440, 2.0)])
        channel_b = _write_freq_wav(tmp_path / "speaker_b.wav", [(880, 1.5)])
        segments = [
            {"speaker": "speaker_a", "start": 0.0, "end": 2.0},
            {"speaker": "speaker_b", "start": 2.3, "end": 3.8},
        ]
        return combined, channel_a, channel_b, segments

    def test_second_speaker_is_cut_from_the_combined_recording(self, tmp_path):
        combined, _, channel_b, segments = self._fixture(tmp_path)
        out, orig_s, trim_s = trim_by_timestamps(
            channel_b, segments, "B", output_path=tmp_path / "out.wav", combined_audio_path=combined
        )
        assert orig_s == pytest.approx(3.8, abs=0.05)  # cut from the combined file
        assert 1.4 < trim_s < 2.1  # B's 1.5 s turn plus padding — not 0.1 s
        assert _dominant_freq(out) == pytest.approx(880, abs=5)  # B's audio, not A's

    def test_first_speaker_keeps_the_channel_source(self, tmp_path):
        """Offsets that fit the channel file keep trimming it — channel
        isolation is strictly better audio when the timeline allows it."""
        combined, channel_a, _, segments = self._fixture(tmp_path)
        out, orig_s, trim_s = trim_by_timestamps(
            channel_a, segments, "A", output_path=tmp_path / "out.wav", combined_audio_path=combined
        )
        assert orig_s == pytest.approx(2.0, abs=0.05)  # the channel file itself
        assert _dominant_freq(out) == pytest.approx(440, abs=5)

    def test_full_timeline_channel_file_is_trimmed_in_place(self, tmp_path):
        """A channel file spanning the combined timeline (true stereo split)
        holds the offsets, so the channel — not the combined mix — is cut."""
        combined, _, _, segments = self._fixture(tmp_path)
        channel_b_full = _write_freq_wav(tmp_path / "speaker_b_full.wav", [(0, 2.3), (880, 1.5)])
        out, orig_s, trim_s = trim_by_timestamps(
            channel_b_full, segments, "B", output_path=tmp_path / "out.wav", combined_audio_path=combined
        )
        assert orig_s == pytest.approx(3.8, abs=0.05)
        assert 1.4 < trim_s < 2.1
        assert _dominant_freq(out) == pytest.approx(880, abs=5)

    def test_mismatch_without_combined_raises_instead_of_scoring_padding(self, tmp_path):
        _, _, channel_b, segments = self._fixture(tmp_path)
        with pytest.raises(RuntimeError) as excinfo:
            trim_by_timestamps(channel_b, segments, "B", output_path=tmp_path / "out.wav")
        message = str(excinfo.value)
        assert "speaker_b.wav" in message
        assert "2.30 s" in message  # the offending segment start
        assert "combined-recording timeline" in message

    def test_no_overlap_anywhere_raises(self, tmp_path):
        combined, _, channel_b, _ = self._fixture(tmp_path)
        segments = [{"speaker": "speaker_b", "start": 50.0, "end": 60.0}]
        with pytest.raises(RuntimeError, match="nothing to score"):
            trim_by_timestamps(channel_b, segments, "B", output_path=tmp_path / "out.wav", combined_audio_path=combined)


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

    def test_packaged_config_missing_falls_back_to_defaults(self, monkeypatch):
        """The no-argument path stays lenient: a damaged install still runs."""
        monkeypatch.setattr("psdn_sonar.preprocessing.config_loader.os.path.exists", lambda _: False)
        cfg = load_multi_speaker_config()
        assert cfg["methods"] == DEFAULT_METHODS
        assert cfg["pyannote"] == DEFAULT_SETTINGS["pyannote"]

    def test_fallback_method_set_matches_the_packaged_config(self):
        """Issue #210: the missing-file fallback declared three methods while
        the packaged config lists one, so the set a run swept depended on
        whether the file could be read. A run that cannot read its config
        must behave like a run that can."""
        assert load_multi_speaker_config()["methods"] == list(DEFAULT_METHODS)

    def test_named_file_that_is_missing_raises(self, tmp_path):
        """A caller that names a file gets that file or an error. Falling back
        would evaluate with a configuration nobody asked for and report the
        result as the requested one."""
        with pytest.raises(FileNotFoundError, match="Preprocessing config not found"):
            load_multi_speaker_config(str(tmp_path / "nope.yaml"))

    def test_unknown_methods_skipped(self, tmp_path):
        p = tmp_path / "c.yaml"
        p.write_text("methods:\n  - no_trim\n  - bogus_method\n")
        assert load_multi_speaker_config(str(p))["methods"] == ["no_trim"]

    def test_named_file_with_no_known_methods_raises(self, tmp_path):
        p = tmp_path / "c.yaml"
        p.write_text("methods:\n  - bogus\n")
        with pytest.raises(ValueError, match="no known methods"):
            load_multi_speaker_config(str(p))

    @pytest.mark.parametrize(
        "body,match",
        [
            ("methods: 5\n", "'methods' must be a list of strings"),
            ("methods:\n  - 5\n", "'methods' must be a list of strings"),
            ("silence: oops\n", "'silence' must be a mapping"),
            ("timestamp: 3\n", "'timestamp' must be a mapping"),
            ("- a\n- b\n", "top level must be a mapping"),
        ],
    )
    def test_malformed_structure_raises_instead_of_a_raw_typeerror(self, tmp_path, body, match):
        """``methods: 5`` and ``silence: oops`` used to escape as a bare
        ``TypeError`` from the iteration and the settings merge."""
        p = tmp_path / "c.yaml"
        p.write_text(body)
        with pytest.raises(ValueError, match=match):
            load_multi_speaker_config(str(p))

    @pytest.mark.parametrize("body", ["[]\n", "false\n", "0\n", '""\n'])
    def test_falsy_yaml_document_is_not_an_empty_config(self, tmp_path, body):
        """``yaml.safe_load(f) or {}`` turned every falsy document into ``{}``,
        which then passed the mapping check and silently produced the default
        configuration — the failure the strict path exists to prevent."""
        p = tmp_path / "c.yaml"
        p.write_text(body)
        with pytest.raises(ValueError, match="top level must be a mapping"):
            load_multi_speaker_config(str(p))

    @pytest.mark.parametrize("body", ["", "null\n"])
    def test_empty_document_specifies_nothing_and_takes_the_defaults(self, tmp_path, body):
        """Only a genuinely empty document is an empty config."""
        p = tmp_path / "c.yaml"
        p.write_text(body)
        cfg = load_multi_speaker_config(str(p))
        assert cfg["methods"] == DEFAULT_METHODS
        assert cfg["silence"] == DEFAULT_SETTINGS["silence"]

    def test_methods_not_required_lets_an_override_past_a_stale_method_list(self, tmp_path):
        """A caller replacing the list does not need the file to carry a usable
        one — blocking there made ``--methods``/``--method`` unusable against a
        config with a stale method list, which is one of the things an override
        is for. The file's settings still apply."""
        p = tmp_path / "c.yaml"
        p.write_text("methods:\n  - bogus\nsilence:\n  silence_thresh: -35\n")

        cfg = load_multi_speaker_config(str(p), methods_required=False)
        assert cfg["methods"] == []
        assert cfg["silence"]["silence_thresh"] == -35

        with pytest.raises(ValueError, match="no known methods"):
            load_multi_speaker_config(str(p))

    def test_settings_merge_with_defaults(self, tmp_path):
        p = tmp_path / "c.yaml"
        p.write_text("methods:\n  - energy_trim\nsilence:\n  silence_thresh: -35\n")
        cfg = load_multi_speaker_config(str(p))
        assert cfg["silence"]["silence_thresh"] == -35
        assert cfg["silence"]["max_silence_ms"] == 400  # default preserved
