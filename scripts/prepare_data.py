#!/usr/bin/env python3
"""Convert a transcript-JSON delivery (audio + per-recording JSON) to eval TSVs.

Usage:
    python scripts/prepare_data.py --transcripts-dir data/transcripts \\
        --audio-dir data/audio --output data/asr_eval.tsv
"""

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class TranscriptJsonPrep:
    """Builds ASR-evaluation TSVs from per-recording transcript JSON files.

    Each JSON is expected to carry ``audio_id``, ``num_speakers``, and a
    ``segments`` list with ``text`` entries; single-speaker recordings map to
    ``<audio_id>.wav`` and multi-speaker ones to ``<audio_id>_combined.wav``.
    """

    @staticmethod
    def _read_transcript(json_path: Path) -> Optional[dict]:
        """Load one transcript JSON; None on read/parse errors."""
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return None

    @staticmethod
    def _extract_transcript_text(segments: List[dict]) -> str:
        """Concatenate segment texts into a single transcript string."""
        texts = [seg.get("text", "").strip() for seg in (segments or []) if seg.get("text", "").strip()]
        return " ".join(texts)

    @staticmethod
    def _collect_rows(
        transcripts_dir: Path,
        audio_dir: Path,
        single_speaker_only: bool,
    ) -> Tuple[List[dict], int, int]:
        """Build row dicts from transcript JSONs; returns (rows, skipped_multi, skipped_no_audio)."""
        rows = []
        skipped_multi = 0
        skipped_no_audio = 0

        for json_file in sorted(transcripts_dir.glob("*.json")):
            data = TranscriptJsonPrep._read_transcript(json_file)
            if data is None:
                continue

            audio_id = data.get("audio_id", "")
            num_speakers = data.get("num_speakers", 1)

            if single_speaker_only and num_speakers > 1:
                skipped_multi += 1
                continue

            audio_filename = f"{audio_id}.wav" if num_speakers == 1 else f"{audio_id}_combined.wav"
            audio_path = audio_dir / audio_filename

            if not audio_path.exists():
                skipped_no_audio += 1
                continue

            full_transcript = TranscriptJsonPrep._extract_transcript_text(data.get("segments", []))
            if not full_transcript:
                continue

            rows.append(
                {
                    "audio_path": str(audio_path.absolute()),
                    "transcription": full_transcript,
                    "audio_id": audio_id,
                    "num_speakers": num_speakers,
                    "topic": data.get("topic", ""),
                    "duration": data.get("duration", 0),
                }
            )

        return rows, skipped_multi, skipped_no_audio

    @staticmethod
    def _write_tsv(path: Path, rows: List[dict], columns: List[str]) -> None:
        """Write rows to a TSV file with the given column order."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns, delimiter="\t", extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def create_asr_eval_tsv(
        transcripts_dir: str,
        audio_dir: str,
        output_tsv: str,
        single_speaker_only: bool = True,
        write_metadata_tsv: bool = True,
    ) -> Tuple[Path, Path, int, int, int]:
        """Create eval TSV (audio_path, transcription) and an optional metadata TSV.

        Returns:
            (output_path, metadata_path, num_rows, skipped_multi, skipped_no_audio).

        Raises:
            FileNotFoundError: transcripts_dir or audio_dir does not exist.
        """
        t_dir = Path(transcripts_dir)
        a_dir = Path(audio_dir)

        if not t_dir.exists():
            raise FileNotFoundError(f"Transcripts directory not found: {t_dir}")
        if not a_dir.exists():
            raise FileNotFoundError(f"Audio directory not found: {a_dir}")

        rows, skipped_multi, skipped_no_audio = TranscriptJsonPrep._collect_rows(t_dir, a_dir, single_speaker_only)

        out_path = Path(output_tsv)
        TranscriptJsonPrep._write_tsv(out_path, rows, ["audio_path", "transcription"])

        meta_path = out_path
        if write_metadata_tsv:
            meta_path = out_path.with_name(out_path.stem + "_metadata.tsv")
            TranscriptJsonPrep._write_tsv(
                meta_path,
                rows,
                ["audio_id", "audio_path", "transcription", "num_speakers", "topic", "duration"],
            )

        return out_path, meta_path, len(rows), skipped_multi, skipped_no_audio


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a transcript-JSON delivery to ASR eval TSVs")
    parser.add_argument("--transcripts-dir", required=True, help="Directory of per-recording transcript JSON files")
    parser.add_argument("--audio-dir", required=True, help="Directory of WAV audio files")
    parser.add_argument("--output", required=True, help="Output TSV path")
    parser.add_argument(
        "--include-multi-speaker", action="store_true", help="Include multi-speaker recordings (default: skip)"
    )
    parser.add_argument("--no-metadata", action="store_true", help="Skip writing the companion metadata TSV")
    args = parser.parse_args()

    out_path, meta_path, n_rows, skipped_multi, skipped_no_audio = TranscriptJsonPrep.create_asr_eval_tsv(
        transcripts_dir=args.transcripts_dir,
        audio_dir=args.audio_dir,
        output_tsv=args.output,
        single_speaker_only=not args.include_multi_speaker,
        write_metadata_tsv=not args.no_metadata,
    )

    logger.info("Wrote %d rows to %s", n_rows, out_path)
    if meta_path != out_path:
        logger.info("Metadata TSV: %s", meta_path)
    if skipped_multi or skipped_no_audio:
        logger.info("Skipped: %d multi-speaker, %d missing audio", skipped_multi, skipped_no_audio)


if __name__ == "__main__":
    main()
