"""Tests for the data preparation scripts (prepare_data, converters)."""

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
def prep():
    return _load_script("prepare_data")


@pytest.fixture(scope="module")
def zeroth():
    return _load_script("convert_zeroth_korean_to_tsv")


@pytest.fixture(scope="module")
def commonvoice():
    return _load_script("convert_commonvoice_to_tsv")


def _write_delivery(root: Path, audio_id: str, num_speakers: int = 1, texts=("hello", "world")):
    transcripts = root / "transcripts"
    audio = root / "audio"
    transcripts.mkdir(exist_ok=True)
    audio.mkdir(exist_ok=True)

    (transcripts / f"{audio_id}.json").write_text(
        json.dumps(
            {
                "audio_id": audio_id,
                "num_speakers": num_speakers,
                "topic": "daily life",
                "duration": 12.5,
                "segments": [{"text": t} for t in texts],
            }
        ),
        encoding="utf-8",
    )
    suffix = ".wav" if num_speakers == 1 else "_combined.wav"
    (audio / f"{audio_id}{suffix}").write_bytes(b"fake")


class TestTranscriptJsonPrep:
    def test_single_speaker_rows_written(self, prep, tmp_path):
        _write_delivery(tmp_path, "rec1")
        _write_delivery(tmp_path, "rec2", texts=("only",))

        out, meta, n_rows, skipped_multi, skipped_no_audio = prep.TranscriptJsonPrep.create_asr_eval_tsv(
            transcripts_dir=str(tmp_path / "transcripts"),
            audio_dir=str(tmp_path / "audio"),
            output_tsv=str(tmp_path / "eval.tsv"),
        )

        df = pd.read_csv(out, sep="\t")
        assert n_rows == 2
        assert list(df.columns) == ["audio_path", "transcription"]
        assert df["transcription"].tolist() == ["hello world", "only"]

        meta_df = pd.read_csv(meta, sep="\t")
        assert "topic" in meta_df.columns
        assert meta_df["topic"].tolist() == ["daily life"] * 2

    def test_multi_speaker_skipped_by_default(self, prep, tmp_path):
        _write_delivery(tmp_path, "solo", num_speakers=1)
        _write_delivery(tmp_path, "duo", num_speakers=2)

        _, _, n_rows, skipped_multi, _ = prep.TranscriptJsonPrep.create_asr_eval_tsv(
            transcripts_dir=str(tmp_path / "transcripts"),
            audio_dir=str(tmp_path / "audio"),
            output_tsv=str(tmp_path / "eval.tsv"),
        )

        assert n_rows == 1
        assert skipped_multi == 1

    def test_multi_speaker_included_uses_combined_audio(self, prep, tmp_path):
        _write_delivery(tmp_path, "duo", num_speakers=2)

        out, _, n_rows, _, _ = prep.TranscriptJsonPrep.create_asr_eval_tsv(
            transcripts_dir=str(tmp_path / "transcripts"),
            audio_dir=str(tmp_path / "audio"),
            output_tsv=str(tmp_path / "eval.tsv"),
            single_speaker_only=False,
        )

        df = pd.read_csv(out, sep="\t")
        assert n_rows == 1
        assert df["audio_path"].iloc[0].endswith("duo_combined.wav")

    def test_missing_audio_skipped(self, prep, tmp_path):
        _write_delivery(tmp_path, "rec1")
        (tmp_path / "audio" / "rec1.wav").unlink()

        _, _, n_rows, _, skipped_no_audio = prep.TranscriptJsonPrep.create_asr_eval_tsv(
            transcripts_dir=str(tmp_path / "transcripts"),
            audio_dir=str(tmp_path / "audio"),
            output_tsv=str(tmp_path / "eval.tsv"),
        )

        assert n_rows == 0
        assert skipped_no_audio == 1

    def test_missing_dirs_raise(self, prep, tmp_path):
        with pytest.raises(FileNotFoundError):
            prep.TranscriptJsonPrep.create_asr_eval_tsv(
                transcripts_dir=str(tmp_path / "nope"),
                audio_dir=str(tmp_path / "nope"),
                output_tsv=str(tmp_path / "eval.tsv"),
            )


def _write_zeroth_layout(root: Path):
    chapter_dir = root / "003" / "104_003"
    chapter_dir.mkdir(parents=True)
    (chapter_dir / "104_003.trans.txt").write_text(
        "104_003_0001 안녕하세요\n104_003_0002 반갑습니다\n", encoding="utf-8"
    )
    (chapter_dir / "104_003_0001.flac").write_bytes(b"fake")
    (chapter_dir / "104_003_0002.flac").write_bytes(b"fake")


class TestZerothConverter:
    def test_converts_trans_files(self, zeroth, tmp_path):
        _write_zeroth_layout(tmp_path)
        out = tmp_path / "zeroth.tsv"

        result = zeroth.convert(str(tmp_path), str(out))

        df = pd.read_csv(result, sep="\t")
        assert len(df) == 2
        assert df["transcription"].tolist() == ["안녕하세요", "반갑습니다"]
        assert all(p.endswith(".flac") for p in df["audio_path"])

    def test_max_samples_limits(self, zeroth, tmp_path):
        _write_zeroth_layout(tmp_path)

        result = zeroth.convert(str(tmp_path), str(tmp_path / "zeroth.tsv"), max_samples=1)

        assert len(pd.read_csv(result, sep="\t")) == 1

    def test_missing_audio_skipped(self, zeroth, tmp_path):
        _write_zeroth_layout(tmp_path)
        (tmp_path / "003" / "104_003" / "104_003_0002.flac").unlink()

        result = zeroth.convert(str(tmp_path), str(tmp_path / "zeroth.tsv"))

        assert len(pd.read_csv(result, sep="\t")) == 1

    def test_empty_dir_returns_none(self, zeroth, tmp_path):
        assert zeroth.convert(str(tmp_path), str(tmp_path / "zeroth.tsv")) is None


def _write_classic_commonvoice(root: Path):
    corpus = root / "cv-corpus" / "ko"
    clips = corpus / "clips"
    clips.mkdir(parents=True)
    pd.DataFrame({"path": ["a.mp3", "b.mp3", "missing.mp3"], "sentence": ["one", "two", "three"]}).to_csv(
        corpus / "test.tsv", sep="\t", index=False
    )
    (clips / "a.mp3").write_bytes(b"fake")
    (clips / "b.mp3").write_bytes(b"fake")


def _write_spontaneous_commonvoice(root: Path):
    corpus = root / "sps-corpus-en"
    audios = corpus / "audios"
    audios.mkdir(parents=True)
    pd.DataFrame(
        {
            "audio_file": ["x.wav", "y.wav"],
            "transcription": ["alpha", "beta"],
            "split": ["test", "train"],
        }
    ).to_csv(corpus / "ss-corpus-en.tsv", sep="\t", index=False)
    (audios / "x.wav").write_bytes(b"fake")
    (audios / "y.wav").write_bytes(b"fake")


class TestCommonVoiceConverter:
    def test_classic_layout(self, commonvoice, tmp_path):
        _write_classic_commonvoice(tmp_path)
        out = tmp_path / "cv.tsv"

        result = commonvoice.convert(tmp_path, out)

        df = pd.read_csv(result, sep="\t")
        assert list(df.columns) == ["audio_path", "transcription"]
        assert df["transcription"].tolist() == ["one", "two"]

    def test_spontaneous_layout_filters_split(self, commonvoice, tmp_path):
        _write_spontaneous_commonvoice(tmp_path)
        out = tmp_path / "cv.tsv"

        result = commonvoice.convert(tmp_path, out, split="test")

        df = pd.read_csv(result, sep="\t")
        assert df["transcription"].tolist() == ["alpha"]

    def test_max_samples_limits(self, commonvoice, tmp_path):
        _write_classic_commonvoice(tmp_path)

        result = commonvoice.convert(tmp_path, tmp_path / "cv.tsv", max_samples=1)

        assert len(pd.read_csv(result, sep="\t")) == 1

    def test_unrecognized_columns_returns_none(self, commonvoice, tmp_path):
        corpus = tmp_path / "weird"
        corpus.mkdir()
        pd.DataFrame({"foo": [1]}).to_csv(corpus / "test.tsv", sep="\t", index=False)

        assert commonvoice.convert(tmp_path, tmp_path / "cv.tsv") is None

    def test_no_tsv_returns_none(self, commonvoice, tmp_path):
        assert commonvoice.convert(tmp_path, tmp_path / "cv.tsv") is None

    def test_prefers_split_named_tsv(self, commonvoice, tmp_path):
        _write_classic_commonvoice(tmp_path)
        corpus = tmp_path / "cv-corpus" / "ko"
        pd.DataFrame({"path": ["a.mp3"], "sentence": ["dev row"]}).to_csv(corpus / "dev.tsv", sep="\t", index=False)

        source = commonvoice.find_source_tsv(tmp_path, "test")

        assert source.name == "test.tsv"
