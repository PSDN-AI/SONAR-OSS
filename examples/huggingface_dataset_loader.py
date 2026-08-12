"""Load and convert audio datasets from HuggingFace Hub.

Downloads any HF audio dataset, extracts the audio to WAV files, and writes
a psdn-sonar TSV (``audio_path`` + ``transcription``) ready for
``psdn-sonar single``. Common audio/text column names are auto-detected.
"""

import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

_AUDIO_COLUMN_CANDIDATES = ["audio", "audio_file", "file", "path"]
_TEXT_COLUMN_CANDIDATES = ["transcription", "transcript", "text", "sentence", "raw_transcription"]


def _resolve_column(dataset, requested: str, candidates: list, kind: str) -> str:
    """Return *requested* if present, otherwise the first matching candidate."""
    if requested in dataset.column_names:
        return requested
    logger.warning("%s column '%s' not found. Available: %s", kind, requested, dataset.column_names)
    for candidate in candidates:
        if candidate in dataset.column_names:
            logger.info("Using %s column: %s", kind, candidate)
            return candidate
    logger.error("Could not find %s column. Specify with --%s-column", kind, kind.lower())
    sys.exit(1)


def load_hf_dataset_to_tsv(
    dataset_name: str,
    output_tsv: Path,
    split: str = "test",
    audio_column: str = "audio",
    text_column: str = "transcription",
    max_samples: int = 0,
    config: Optional[str] = None,
) -> Path:
    """Download a HF dataset and write a psdn-sonar TSV next to extracted audio.

    Returns the output TSV path.
    """
    import io

    import librosa
    import soundfile as sf
    from datasets import load_dataset

    logger.info("Loading dataset: %s (config=%s, split=%s)", dataset_name, config or "default", split)

    try:
        load_args = [dataset_name, config] if config else [dataset_name]
        dataset = load_dataset(*load_args, split=split)
        logger.info("Loaded %d samples", len(dataset))
    except Exception as e:
        logger.error("Failed to load dataset: %s", e)
        logger.error("Troubleshooting:")
        logger.error("  1. Check the dataset name is correct")
        logger.error("  2. Check if the dataset requires a config (e.g. 'bn_in' for FLEURS)")
        logger.error("  3. Some datasets require trust_remote_code=True")
        sys.exit(1)

    audio_column = _resolve_column(dataset, audio_column, _AUDIO_COLUMN_CANDIDATES, "audio")
    text_column = _resolve_column(dataset, text_column, _TEXT_COLUMN_CANDIDATES, "text")

    output_tsv.parent.mkdir(parents=True, exist_ok=True)
    audio_dir = output_tsv.parent / "audio"
    audio_dir.mkdir(exist_ok=True)

    if max_samples > 0 and len(dataset) > max_samples:
        dataset = dataset.select(range(max_samples))
        logger.info("Limited to %d samples", max_samples)

    logger.info("Extracting audio files and creating TSV")

    # Read the raw Arrow table so audio bytes are not eagerly decoded for
    # every row; each sample is decoded and resampled individually.
    table = dataset.data
    audio_col_idx = dataset.column_names.index(audio_column)
    text_col_idx = dataset.column_names.index(text_column)

    rows = []
    failed = 0

    for idx in range(len(dataset)):
        try:
            audio_data = table.column(audio_col_idx)[idx].as_py()
            text_data = table.column(text_col_idx)[idx].as_py()

            transcription = text_data.strip() if text_data else ""
            if not transcription:
                logger.warning("Empty transcription at index %d", idx)
                failed += 1
                continue

            if isinstance(audio_data, dict) and "bytes" in audio_data:
                audio_array, sampling_rate = librosa.load(io.BytesIO(audio_data["bytes"]), sr=16000)
            elif isinstance(audio_data, dict) and "path" in audio_data:
                audio_array, sampling_rate = librosa.load(audio_data["path"], sr=16000)
            else:
                logger.warning("Unexpected audio format at index %d", idx)
                failed += 1
                continue

            audio_filename = f"sample_{idx:06d}.wav"
            sf.write(str(audio_dir / audio_filename), audio_array, sampling_rate)
            rows.append({"audio_path": f"audio/{audio_filename}", "transcription": transcription})

            if (idx + 1) % 100 == 0:
                logger.info("Processed %d/%d samples", idx + 1, len(dataset))

        except Exception as e:
            logger.error("Error processing sample %d: %s", idx, e)
            failed += 1

    with open(output_tsv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["audio_path", "transcription"], delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Dataset conversion complete: %d samples (%d failed) -> %s", len(rows), failed, output_tsv)
    return output_tsv


def main():
    parser = argparse.ArgumentParser(
        description="Load a dataset from HuggingFace Hub and convert it to psdn-sonar format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Load FLEURS Bengali
  python examples/huggingface_dataset_loader.py \\
      --dataset google/fleurs \\
      --config bn_in \\
      --split test \\
      --output data/fleurs-bengali/test.tsv

  # Load a custom dataset with a sample limit
  python examples/huggingface_dataset_loader.py \\
      --dataset username/my-bengali-dataset \\
      --split test \\
      --max-samples 100 \\
      --output data/my-dataset/test.tsv

  # Then evaluate
  psdn-sonar single \\
      --input data/my-dataset/test.tsv \\
      --hf-model openai/whisper-small \\
      --language bn
        """,
    )

    parser.add_argument(
        "--dataset", required=True, help='HuggingFace dataset ID (e.g., "google/fleurs", "username/dataset")'
    )
    parser.add_argument("--output", type=Path, required=True, help="Output TSV path")
    parser.add_argument("--split", default="test", help="Dataset split to load (default: test)")
    parser.add_argument("--config", type=str, help='Dataset configuration/subset (e.g., "bn_in" for FLEURS)')
    parser.add_argument("--audio-column", default="audio", help="Name of audio column (default: audio)")
    parser.add_argument(
        "--text-column", default="transcription", help="Name of text/transcription column (default: transcription)"
    )
    parser.add_argument("--max-samples", type=int, default=0, help="Maximum samples to load (0 = all)")

    args = parser.parse_args()

    output_tsv = load_hf_dataset_to_tsv(
        dataset_name=args.dataset,
        output_tsv=args.output,
        split=args.split,
        audio_column=args.audio_column,
        text_column=args.text_column,
        max_samples=args.max_samples,
        config=args.config,
    )

    logger.info("Next steps:")
    logger.info("  psdn-sonar single --input %s --hf-model openai/whisper-small --language bn", output_tsv)


if __name__ == "__main__":
    main()
