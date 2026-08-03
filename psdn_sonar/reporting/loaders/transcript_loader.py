"""Load reference transcripts from TSV or JSONL dataset files."""

import csv
import json
import logging
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


def load_transcripts_from_file(file_path: str, dataset_dir: Optional[str] = None) -> List[str]:
    """Load transcripts from a ``.tsv`` or ``.jsonl`` file by extension."""
    path = Path(file_path)

    if path.suffix == ".tsv":
        return load_transcripts_from_tsv(path)
    elif path.suffix == ".jsonl":
        return load_transcripts_from_jsonl(path, dataset_dir)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")


def load_transcripts_from_tsv(tsv_path: Path) -> List[str]:
    """Extract non-empty transcripts from a TSV file.

    Tries, in order: a ``sentence`` or ``transcription`` header column, a
    headerless two-column layout (transcript second), then a three-column
    layout (transcript third). Raises ``ValueError`` when no strategy yields
    any transcripts.
    """
    transcripts = []

    try:
        with open(tsv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            if reader.fieldnames:
                if "sentence" in reader.fieldnames:
                    for row in reader:
                        transcript = row.get("sentence", "").strip()
                        if transcript:
                            transcripts.append(transcript)
                    if transcripts:
                        return transcripts

                if "transcription" in reader.fieldnames:
                    for row in reader:
                        transcript = row.get("transcription", "").strip()
                        if transcript:
                            transcripts.append(transcript)
                    if transcripts:
                        return transcripts
    except Exception as exc:
        logger.debug("TSV header-based parsing failed for %s: %s", tsv_path, exc)

    transcripts = []
    try:
        with open(tsv_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t", 1)
                if len(parts) >= 2:
                    transcript = parts[1].strip()
                    if transcript:
                        transcripts.append(transcript)
        if transcripts:
            return transcripts
    except Exception as exc:
        logger.debug("TSV two-column parsing failed for %s: %s", tsv_path, exc)

    transcripts = []
    try:
        with open(tsv_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) >= 3:
                    transcript = parts[2].strip()
                    if transcript:
                        transcripts.append(transcript)
        if transcripts:
            return transcripts
    except Exception as exc:
        logger.debug("TSV three-column parsing failed for %s: %s", tsv_path, exc)

    raise ValueError(f"Could not parse TSV file format: {tsv_path}")


def load_transcripts_from_jsonl(jsonl_path: Path, dataset_dir: Optional[str] = None) -> List[str]:
    """Read transcripts referenced by ``transcript_path`` entries in a JSONL manifest.

    Relative paths resolve against *dataset_dir* (default: the manifest's
    directory); missing transcript files are skipped.
    """
    transcripts = []
    base_dir = Path(dataset_dir) if dataset_dir else jsonl_path.parent

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            transcript_path = base_dir / entry["transcript_path"]

            if transcript_path.exists():
                with open(transcript_path, "r", encoding="utf-8") as tf:
                    content = tf.read().strip()
                    if content:
                        transcripts.append(content)

    return transcripts
