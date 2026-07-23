"""OpenSLR corpus loaders (Bengali SLR37/SLR53 layouts)."""

import glob
import os

from .base import OpenSLRBaseLoader, SingleSpeakerLoaderBase, _asr_bengali_filter


def _load_line_index_tsv(root_dir) -> dict:
    """Load ``line_index.tsv`` (``filename\\ttranscription``) into a metadata map.

    Entries are keyed by both the raw filename and its extension-less stem so
    audio discovery can match either form.
    """
    metadata_map = {}
    tsv_path = os.path.abspath(os.path.normpath(os.path.join(root_dir, "line_index.tsv")))
    if not os.path.isfile(tsv_path):
        return metadata_map
    with open(tsv_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = s.split("\t", 1)
            if len(parts) < 2:
                continue
            file_id = parts[0].strip()
            transcription = parts[1].strip()
            if not file_id or not transcription:
                continue
            metadata_map[file_id] = {"file_id": file_id, "transcription": transcription}
            stem = os.path.splitext(file_id)[0]
            if stem and stem != file_id and stem not in metadata_map:
                metadata_map[stem] = {"file_id": file_id, "transcription": transcription}
    return metadata_map


class OpenSLRLineIndexLoader(SingleSpeakerLoaderBase):
    """OpenSLR datasets with ``line_index.tsv`` and ``wavs/*.wav``."""

    @staticmethod
    def load_metadata(root_dir) -> dict:
        return _load_line_index_tsv(root_dir)

    @staticmethod
    def find_audio_files(root_dir) -> list:
        """Return [(subdir, abs_path, file_id, rel_path)] for wavs with metadata."""
        meta = OpenSLRLineIndexLoader.load_metadata(root_dir)
        audio_files = []
        wavs_dir = os.path.join(root_dir, "wavs")
        if not os.path.isdir(wavs_dir):
            return audio_files
        for wav_path in glob.glob(os.path.join(wavs_dir, "*.wav")):
            basename = os.path.basename(wav_path)
            stem = os.path.splitext(basename)[0]
            rel = os.path.relpath(wav_path, root_dir).replace("\\", "/")
            file_id = basename if basename in meta else (stem if stem in meta else None)
            if file_id:
                audio_files.append(("wavs", wav_path, file_id, rel))
        return audio_files


class OpenSLR37BDLoader(OpenSLRBaseLoader):
    @staticmethod
    def load_metadata(d) -> dict:
        return OpenSLRBaseLoader._load_metadata(d, _asr_bengali_filter)

    @staticmethod
    def find_audio_files(d) -> list:
        return OpenSLRBaseLoader._find_audio(d, _asr_bengali_filter)


class OpenSLR53Loader(OpenSLRBaseLoader):
    @staticmethod
    def load_metadata(d) -> dict:
        return OpenSLRBaseLoader._load_metadata(d, _asr_bengali_filter)

    @staticmethod
    def find_audio_files(d) -> list:
        return OpenSLRBaseLoader._find_audio(d, _asr_bengali_filter)
