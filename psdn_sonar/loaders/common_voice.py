"""Mozilla Common Voice loader (``<lang>/test.tsv`` + ``<lang>/clips/*.mp3``)."""

import csv
import glob
import os

from .base import SingleSpeakerLoaderBase


class CommonVoiceLoader(SingleSpeakerLoaderBase):
    ID_FIELD = "client_id"
    TRANSCRIPTION_KEY = "sentence"

    DEFAULT_LANGUAGE = "bn"

    @classmethod
    def load_metadata(cls, cv_dir, language=None) -> dict:
        """Return {file_id: {path, sentence, client_id}} from ``<lang>/test.tsv``."""
        metadata_map = {}
        lang = language or cls.DEFAULT_LANGUAGE
        tsv_path = os.path.join(cv_dir, lang, "test.tsv")
        if not os.path.exists(tsv_path):
            return metadata_map

        with open(tsv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                path = (row.get("path") or "").strip()
                sentence = (row.get("sentence") or "").strip()
                if path and sentence:
                    file_id = os.path.splitext(path)[0]
                    metadata_map[file_id] = {"path": path, "sentence": sentence, "client_id": row.get("client_id", "")}
        return metadata_map

    @classmethod
    def find_audio_files(cls, cv_dir, language=None) -> list:
        """Return [(abs_path, file_id, rel_path)] for ``<lang>/clips/*.mp3``."""
        audio_files = []
        lang = language or cls.DEFAULT_LANGUAGE
        clips_dir = os.path.join(cv_dir, lang, "clips")
        if os.path.exists(clips_dir):
            for mp3_file in glob.glob(os.path.join(clips_dir, "*.mp3")):
                file_id = os.path.splitext(os.path.basename(mp3_file))[0]
                rel_path = os.path.relpath(mp3_file, cv_dir).replace("\\", "/")
                audio_files.append((mp3_file, file_id, rel_path))
        return audio_files
