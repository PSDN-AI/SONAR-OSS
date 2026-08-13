#!/usr/bin/env python3
"""Convert a Common Voice download to the toolkit TSV format.

Handles both the classic corpus layout (``path``/``sentence`` columns with a
``clips/`` audio dir) and the Spontaneous Speech corpus layout
(``audio_file``/``transcription`` columns with an ``audios/`` dir and a
``split`` column). Pass a ``.tar.gz`` archive or an already-extracted directory.

Usage:
    python scripts/convert_commonvoice_to_tsv.py --archive Common_Voice_English_1.0.tar.gz \\
        --output data/commonvoice_test.tsv
    python scripts/convert_commonvoice_to_tsv.py --data-dir extracted/ --split test \\
        --output data/commonvoice_test.tsv
"""

import argparse
import logging
import sys
import tarfile
from pathlib import Path
from typing import Optional

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

_AUDIO_COLUMNS = ["audio_path", "audio_file", "path"]
_TEXT_COLUMNS = ["transcription", "sentence", "transcript", "text"]
_AUDIO_DIRS = ["clips", "audios", "audio"]


def find_source_tsv(data_dir: Path, split: str) -> Optional[Path]:
    """Locate the corpus TSV, preferring one whose name contains *split*."""
    tsv_files = sorted(data_dir.rglob("*.tsv"))
    if not tsv_files:
        return None
    for tsv_file in tsv_files:
        if split in tsv_file.name.lower():
            return tsv_file
    return tsv_files[0]


def _resolve_column(df: pd.DataFrame, candidates: list) -> Optional[str]:
    return next((c for c in candidates if c in df.columns), None)


def convert(data_dir: Path, output_path: Path, split: str = "test", max_samples: int = 0) -> Optional[Path]:
    """Convert an extracted Common Voice directory to (audio_path, transcription) TSV.

    Returns the output path, or None when the layout is not recognized or no
    referenced audio files exist.
    """
    source_tsv = find_source_tsv(data_dir, split)
    if source_tsv is None:
        logger.error("No TSV files found under %s", data_dir)
        return None
    logger.info("Using source TSV: %s", source_tsv)

    df = pd.read_csv(source_tsv, sep="\t")

    audio_col = _resolve_column(df, _AUDIO_COLUMNS)
    text_col = _resolve_column(df, _TEXT_COLUMNS)
    if audio_col is None or text_col is None:
        logger.error("Unrecognized columns in %s: %s", source_tsv, list(df.columns))
        return None

    if "split" in df.columns:
        df = df[df["split"] == split]
        logger.info("Filtered to %d rows for split=%s", len(df), split)

    if max_samples > 0:
        df = df.head(max_samples)

    audio_dir = next((source_tsv.parent / d for d in _AUDIO_DIRS if (source_tsv.parent / d).is_dir()), None)
    if audio_dir is None:
        logger.error("No audio directory (%s) found near %s", "/".join(_AUDIO_DIRS), source_tsv)
        return None

    rows = []
    for audio_file, text in zip(df[audio_col], df[text_col]):
        text = str(text).strip()
        audio_path = audio_dir / str(audio_file)
        if text and audio_path.exists():
            rows.append({"audio_path": str(audio_path), "transcription": text})

    if not rows:
        logger.error("No rows with existing audio files found")
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, sep="\t", index=False)
    logger.info("Wrote %d samples to %s", len(rows), output_path)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Convert a Common Voice download to the toolkit TSV format")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--archive", type=Path, help="Common Voice .tar.gz archive to extract and convert")
    source.add_argument("--data-dir", type=Path, help="Already-extracted Common Voice directory")
    parser.add_argument("--output", type=Path, required=True, help="Output TSV path")
    parser.add_argument("--split", default="test", help="Corpus split to keep (default: test)")
    parser.add_argument("--max-samples", type=int, default=0, help="Limit samples (0 = all)")
    args = parser.parse_args()

    data_dir = args.data_dir
    if args.archive:
        data_dir = args.archive.parent / "extracted"
        data_dir.mkdir(exist_ok=True)
        logger.info("Extracting %s ...", args.archive)
        with tarfile.open(args.archive, "r:gz") as tar:
            tar.extractall(data_dir)

    if convert(data_dir, args.output, split=args.split, max_samples=args.max_samples) is None:
        sys.exit(1)


if __name__ == "__main__":
    main()
