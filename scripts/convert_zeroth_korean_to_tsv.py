#!/usr/bin/env python3
"""Convert an extracted Zeroth-Korean (OpenSLR SLR40) directory to the toolkit TSV format.

Expects the ``{chapter}/{speaker}_{chapter}/`` layout with ``.flac`` audio and
``*.trans.txt`` files ("utterance_id transcript" per line).

Usage:
    python scripts/convert_zeroth_korean_to_tsv.py --data-dir path/to/test_data_01 \\
        --output data/zeroth_korean_test.tsv
"""

import argparse
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def convert(data_dir: str, output_path: str, max_samples: Optional[int] = None) -> Optional[Path]:
    """Walk ``*.trans.txt`` files and write (audio_path, transcription) rows.

    Returns the output path, or None when no samples were found.
    """
    root = Path(data_dir)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for trans_file in sorted(root.rglob("*.trans.txt")):
        speaker_chapter_dir = trans_file.parent

        with open(trans_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split(" ", 1)
                if len(parts) != 2:
                    continue

                utt_id, transcript = parts
                audio_file = speaker_chapter_dir / f"{utt_id}.flac"
                if not audio_file.exists():
                    continue

                rows.append({"audio_path": str(audio_file.resolve()), "transcription": transcript.strip()})
                if max_samples and len(rows) >= max_samples:
                    break

        if max_samples and len(rows) >= max_samples:
            break

    if not rows:
        logger.error("No samples found. Check --data-dir path.")
        return None

    df = pd.DataFrame(rows)
    df.to_csv(out_path, sep="\t", index=False)
    logger.info("Converted %d samples to %s", len(df), out_path)
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Zeroth-Korean to the toolkit TSV format")
    parser.add_argument("--data-dir", required=True, help="Extracted Zeroth-Korean directory (e.g. test_data_01)")
    parser.add_argument("--output", required=True, help="Output TSV file path")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit number of samples (default: all)")
    args = parser.parse_args()

    convert(args.data_dir, args.output, args.max_samples)
