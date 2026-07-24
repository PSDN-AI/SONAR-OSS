"""Google FLEURS loader (``test/test.tsv`` + ``test/audio/*.wav``)."""

import csv
import glob
import os

from .base import SingleSpeakerLoaderBase


class FleursLoader(SingleSpeakerLoaderBase):
    ID_FIELD = "id"

    @staticmethod
    def load_metadata(fleurs_dir) -> dict:
        """Return {audio_file: metadata} from ``test/test.tsv``.

        Handles both the headerless upstream export (``id\\taudio\\ttext``)
        and a headered TSV with an ``audio_file``/``path`` column.
        """
        metadata_map = {}
        tsv_path = os.path.join(fleurs_dir, "test", "test.tsv")
        if not os.path.exists(tsv_path):
            return metadata_map
        with open(tsv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            first = next(reader, None)
            if not first:
                return metadata_map
            if len(first) >= 3 and first[1].endswith(".wav") and "audio_file" not in first:
                row = {"id": first[0], "audio_file": first[1].strip().replace("\\", "/"), "transcription": first[2]}
                metadata_map[row["audio_file"]] = row
                for parts in reader:
                    if len(parts) >= 3:
                        af = parts[1].strip().replace("\\", "/")
                        metadata_map[af] = {"id": parts[0], "audio_file": af, "transcription": parts[2]}
                return metadata_map
        with open(tsv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                af = (row.get("audio_file") or row.get("path") or "").strip().replace("\\", "/")
                if af:
                    metadata_map[af] = row
        return metadata_map

    @staticmethod
    def find_audio_files(fleurs_dir) -> list:
        """Return [(split, abs_path, rel_path)] for wavs under ``test/``."""
        audio_files = []
        split_dir = os.path.join(fleurs_dir, "test")
        if not os.path.exists(split_dir):
            return audio_files
        # Metadata keys are sometimes a bare filename, sometimes a relative path.
        search_paths = [os.path.join(split_dir, "audio", "*.wav"), os.path.join(split_dir, "*.wav")]
        for sp in search_paths:
            for af in glob.glob(sp):
                rel_path = os.path.relpath(af, split_dir).replace("\\", "/")
                if os.path.basename(af) not in audio_files:
                    audio_files.append(("test", af, rel_path))
        return audio_files
