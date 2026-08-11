"""Dataset loader contract and shared row-shaping logic.

A loader knows how to read one dataset layout from disk: its metadata file(s),
where the audio lives, and how to shape a per-utterance evaluation output row.
Loaders are stateless — every method is a class/static method.
"""

import glob
import os
from typing import Callable, ClassVar, List, Protocol, runtime_checkable


@runtime_checkable
class DatasetLoader(Protocol):
    """Interface that all dataset loaders must implement."""

    def load_metadata(self, data_dir: str, **kwargs) -> dict:
        """Load dataset metadata, returning {file_id: metadata_dict}."""
        ...

    def get_output_fieldnames(self) -> list:
        """Return CSV/TSV column names for output."""
        ...

    def get_transcription_from_metadata(self, metadata: dict) -> str:
        """Extract transcription text from a metadata entry."""
        ...

    def create_output_row(
        self,
        metadata,
        transcription,
        transcription_norm,
        asr_transcription,
        asr_norm_non,
        asr_norm_conv,
        cer_non,
        wer_non,
        sem_non,
        poseidon_non,
        cer_conv,
        wer_conv,
        sem_conv,
        poseidon_conv,
        path_from_root,
        inference_latency_s=None,
    ) -> dict:
        """Create a dict representing one output row."""
        ...


def _fmt(val) -> str:
    return f"{val:.4f}" if val is not None else ""


class SingleSpeakerLoaderBase:
    """Shared row shaping for single-speaker loaders.

    Output rows are identical across datasets except for the leading ID
    column; subclasses set ``ID_FIELD`` (both the column name and the
    metadata key) and ``TRANSCRIPTION_KEY`` (the metadata key holding the
    reference text).
    """

    ID_FIELD: ClassVar[str] = "file_id"
    TRANSCRIPTION_KEY: ClassVar[str] = "transcription"

    _METRIC_FIELDNAMES: ClassVar[List[str]] = [
        "transcription",
        "transcription_norm",
        "asr_transcription",
        "asr_transcription_norm_non",
        "asr_transcription_norm_conv",
        "cer_non",
        "wer_non",
        "semantic_similarity_non",
        "poseidon_score_non",
        "cer_conv",
        "wer_conv",
        "semantic_similarity_conv",
        "poseidon_score_conv",
        "inference_latency_s",
    ]

    @classmethod
    def get_output_fieldnames(cls) -> List[str]:
        return [cls.ID_FIELD, "path", *cls._METRIC_FIELDNAMES]

    @classmethod
    def get_transcription_from_metadata(cls, metadata: dict) -> str:
        return metadata.get(cls.TRANSCRIPTION_KEY, "").strip()

    @classmethod
    def create_output_row(
        cls,
        metadata,
        transcription,
        transcription_norm,
        asr_transcription,
        asr_norm_non,
        asr_norm_conv,
        cer_non,
        wer_non,
        sem_non,
        poseidon_non,
        cer_conv,
        wer_conv,
        sem_conv,
        poseidon_conv,
        path_from_root,
        inference_latency_s=None,
    ) -> dict:
        return {
            cls.ID_FIELD: metadata.get(cls.ID_FIELD, ""),
            "path": path_from_root,
            "transcription": transcription,
            "transcription_norm": transcription_norm,
            "asr_transcription": asr_transcription,
            "asr_transcription_norm_non": asr_norm_non,
            "asr_transcription_norm_conv": asr_norm_conv,
            "cer_non": _fmt(cer_non),
            "wer_non": _fmt(wer_non),
            "semantic_similarity_non": _fmt(sem_non),
            "poseidon_score_non": _fmt(poseidon_non),
            "cer_conv": _fmt(cer_conv),
            "wer_conv": _fmt(wer_conv),
            "semantic_similarity_conv": _fmt(sem_conv),
            "poseidon_score_conv": _fmt(poseidon_conv),
            "inference_latency_s": _fmt(inference_latency_s),
        }


def _asr_bengali_filter(x: str) -> bool:
    """Match asr_bengali_* subdirs (OpenSLR 37/53 use these; 53 has asr_bengali_0..8)."""
    return x.startswith("asr_bengali")


class OpenSLRBaseLoader(SingleSpeakerLoaderBase):
    """Shared metadata/audio scanning for OpenSLR-style corpora.

    Reads ``utt_spk_text.tsv`` (root and matching subdirs) and finds
    ``*.flac`` audio recursively under subdirs accepted by ``filter_func``.
    """

    @staticmethod
    def _load_metadata(openslr_dir: str, filter_func: Callable[[str], bool]) -> dict:
        metadata_map = {}
        files_to_check = [os.path.join(openslr_dir, "utt_spk_text.tsv")]
        for item in os.listdir(openslr_dir):
            if os.path.isdir(os.path.join(openslr_dir, item)) and filter_func(item):
                p = os.path.join(openslr_dir, item, "utt_spk_text.tsv")
                if os.path.exists(p):
                    files_to_check.append(p)

        for p in files_to_check:
            if not os.path.exists(p):
                continue
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) >= 3:
                        metadata_map[parts[0].strip()] = {"file_id": parts[0], "transcription": parts[2].strip()}
        return metadata_map

    @staticmethod
    def _find_audio(openslr_dir: str, filter_func: Callable[[str], bool]) -> list:
        audio_files = []
        for item in os.listdir(openslr_dir):
            item_path = os.path.join(openslr_dir, item)
            if os.path.isdir(item_path) and filter_func(item):
                search_pattern = os.path.join(item_path, "**", "*.flac")
                for flac in glob.glob(search_pattern, recursive=True):
                    fid = os.path.splitext(os.path.basename(flac))[0]
                    rel = os.path.relpath(flac, openslr_dir).replace("\\", "/")
                    audio_files.append((item, flac, fid, rel))
        return audio_files
