"""Download, clean, split, and export public datasets for a language."""

from __future__ import annotations

import csv
import errno
import json
import logging
import random
import shutil
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Optional, cast

from .catalog import validate_huggingface_revision
from .registry import DATASET_REGISTRY, AvailableDataset, resolve_config

if TYPE_CHECKING:
    from datasets import Dataset

logger = logging.getLogger(__name__)

# huggingface_hub._check_disk_space warns with this prefix *before* downloading
# a file that will not fit on disk. Promoted to an error so the run stops at
# that point instead of downloading for hours until ENOSPC (issue #183).
_DISK_SPACE_WARNING = "Not enough free disk space"

# ENOSPC as rendered by layers that drop the errno: pyarrow/C++ keeps the
# strerror text, hf_transfer/xet (Rust) renders "... (os error 28)".
_ENOSPC_TEXT_MARKERS = ("No space left on device", "os error 28")


def _hf_cache_home() -> Path:
    """Best-effort location of the HuggingFace cache (holds partial downloads)."""
    try:
        from huggingface_hub.constants import HF_HOME

        return Path(HF_HOME)
    except Exception:
        return Path.home() / ".cache" / "huggingface"


def _disk_full_evidence(exc: BaseException) -> Optional[BaseException]:
    """Return the exception in *exc*'s cause chain evidencing a full disk, else None.

    datasets/pyarrow/huggingface_hub wrap the original ENOSPC in their own
    exception types (often losing the errno), so check each link for the errno,
    the strerror text, or the promoted hub disk-space warning.
    """
    seen: set[int] = set()
    current: Optional[BaseException] = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, OSError) and current.errno == errno.ENOSPC:
            return current
        if isinstance(current, Warning) and str(current).startswith(_DISK_SPACE_WARNING):
            return current
        text = str(current)
        if any(marker in text for marker in _ENOSPC_TEXT_MARKERS):
            return current
        current = current.__cause__ or current.__context__
    return None


def _raise_if_disk_full(exc: BaseException, hf_id: str) -> None:
    """Re-raise *exc* as an actionable ENOSPC OSError if it evidences a full disk.

    Continuing past a full disk cannot succeed, so this must escape the
    per-split downgrade-and-continue handling. Raising OSError (not
    RuntimeError) routes it to the CLI's clean one-line error path (#149),
    and the message names what the bare "os error 28" does not: the partial
    download sitting in the HuggingFace cache and how to reclaim it.
    """
    evidence = _disk_full_evidence(exc)
    if evidence is None:
        return
    cache = _hf_cache_home()
    try:
        free_note = f"; that disk has {shutil.disk_usage(cache).free / 1e9:.1f} GB free now"
    except OSError:
        free_note = ""
    raise OSError(
        errno.ENOSPC,
        f"the disk filled up while preparing {hf_id} ({evidence}). "
        f"The partial download is kept in the HuggingFace cache at {cache}{free_note}. "
        f"Reclaim the space by deleting the dataset's directories under that cache "
        f"({cache / 'hub'} holds the raw download, {cache / 'datasets'} the prepared data). "
        "Note that each requested split is fetched in full on first run; --max-samples "
        "bounds what is prepared, not the download (see 'What a first run costs' in the README).",
    ) from exc


def _audio_to_path(audio: dict | str | None, index: int, audio_dir: Path) -> str:
    """Resolve audio to a file path string for the TSV. Writes to audio_dir if only array is given."""
    if audio is None:
        return f"row_{index}"
    if isinstance(audio, dict):
        path = audio.get("path")
        arr = audio.get("array")
        sr = audio.get("sampling_rate")
        if path and isinstance(path, str) and path.strip():
            return path
        if arr is not None:
            # Write array to WAV so evaluation can read it
            import numpy as np
            import soundfile as sf

            wav_path = audio_dir / f"audio_{index}.wav"
            sr = int(sr) if sr is not None else 16000
            arr = np.asarray(arr, dtype=np.float32)
            if arr.ndim > 1:
                arr = arr.mean(axis=1)
            sf.write(str(wav_path), arr, sr)
            return str(wav_path)
        return f"row_{index}"
    if isinstance(audio, str) and audio.strip():
        return audio
    return f"row_{index}"


def prepare_dataset(
    dataset_name: str,
    language: str,
    split: str,
    output_dir: Path,
    max_samples: int = 0,
) -> Path:
    """Download a registered dataset for the given language/split and write a TSV.

    Returns the path to the generated TSV. For datasets with no config (e.g. Zeroth),
    load_dataset is called without a config argument.
    """
    name = dataset_name.strip().lower()
    lang = language.strip().lower()
    spec = DATASET_REGISTRY.get(name)
    if not spec:
        raise ValueError(f"Unknown dataset: {name}. Known: {list(DATASET_REGISTRY.keys())}")
    config = resolve_config(spec, lang)
    if config is None:
        raise ValueError(f"Dataset {name} does not support language: {lang}")

    if split not in spec.splits:
        raise ValueError(f"Dataset {name} has splits {spec.splits}, not {split}")

    from datasets import load_dataset

    # A concrete split= always yields a Dataset, but the stubs declare the
    # full DatasetDict/IterableDataset union.
    if config:
        ds = cast("Dataset", load_dataset(spec.hf_id, config, split=split, revision=spec.revision))
    else:
        ds = cast("Dataset", load_dataset(spec.hf_id, split=split, revision=spec.revision))

    total = len(ds)
    if max_samples and max_samples < total:
        ds = ds.select(range(max_samples))
        total = max_samples

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    tsv_path = output_dir / f"{name}_{lang}_{split}.tsv"

    with open(tsv_path, "w", encoding="utf-8") as f:
        f.write("audio_path\ttranscription\n")
        for i in range(len(ds)):
            row = ds[i]
            audio = row.get(spec.audio_column)
            text = row.get(spec.text_column, "")
            if isinstance(text, bytes):
                text = text.decode("utf-8", errors="replace")
            path = _audio_to_path(audio, i, audio_dir)
            f.write(f"{path}\t{text}\n")

    logger.info("Wrote %s (%d rows) to %s", tsv_path.name, total, tsv_path)
    return tsv_path


class DatasetPreparer:
    """Downloads a discovered dataset, normalises text, and writes train/val/test TSVs."""

    TSV_COLUMNS = (
        "audio_path",
        "transcription",
        "transcription_norm",
        "duration_s",
        "snr_db",
    )

    def __init__(
        self,
        dataset: AvailableDataset,
        language: str,
        output_dir: str | Path,
        max_samples: int = 0,
        split_ratio: tuple[int, int, int] = (80, 10, 10),
        skip_audio_validation: bool = False,
        seed: int | None = None,
    ):
        from psdn_sonar.config_loader import get_run_seed

        try:
            validate_huggingface_revision(dataset.revision)
        except ValueError as exc:
            raise ValueError("DatasetPreparer requires an immutable source revision") from exc
        self.dataset = dataset
        self.language = language
        self.output_dir = Path(output_dir) / dataset.name
        self.max_samples = max_samples
        self.split_ratio = split_ratio
        self.skip_audio_validation = skip_audio_validation
        self._rng = random.Random(get_run_seed() if seed is None else seed)

    def prepare(self) -> Path:
        """Run the full pipeline: download -> clean -> split -> export.

        Returns the output directory containing the TSVs.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        audio_dir = self.output_dir / "audio"
        audio_dir.mkdir(exist_ok=True)

        from datasets import load_dataset

        hf_splits = self.dataset.splits
        has_predefined_splits = len(hf_splits) > 1

        all_records: dict[str, list[dict]] = {}
        last_error: Optional[BaseException] = None

        for split in hf_splits:
            logger.info(
                "Loading %s/%s split=%s …",
                self.dataset.hf_id,
                self.dataset.config,
                split,
            )
            try:
                with warnings.catch_warnings():
                    # Stop at the first file that will not fit on disk instead
                    # of downloading past the hub's warning until ENOSPC (#183).
                    warnings.filterwarnings("error", message=_DISK_SPACE_WARNING)
                    if self.dataset.config:
                        ds = load_dataset(
                            self.dataset.hf_id,
                            self.dataset.config,
                            split=split,
                            revision=self.dataset.revision,
                        )
                    else:
                        ds = load_dataset(
                            self.dataset.hf_id,
                            split=split,
                            revision=self.dataset.revision,
                        )
            except Exception as exc:
                # A full disk fails every remaining split too — abort, do not
                # downgrade it to a per-split warning (#183).
                _raise_if_disk_full(exc, self.dataset.hf_id)
                last_error = exc
                logger.warning("Could not load split '%s': %s", split, exc)
                continue

            try:
                records = self._process_split(ds, split, audio_dir)
            except OSError as exc:
                _raise_if_disk_full(exc, self.dataset.hf_id)
                raise
            all_records[split] = records
            logger.info("  Processed %d samples for split '%s'", len(records), split)

        if not any(all_records.values()):
            message = f"No samples could be loaded from {self.dataset.hf_id}/{self.dataset.config}"
            if last_error is not None:
                # Chain and name the real failure: an unchained RuntimeError
                # hides the cause from the reader and from the CLI (#183).
                raise RuntimeError(f"{message} (last failure: {last_error})") from last_error
            raise RuntimeError(message)

        if has_predefined_splits:
            split_map = self._use_predefined_splits(all_records)
        else:
            combined = []
            for recs in all_records.values():
                combined.extend(recs)
            split_map = self._create_splits(combined)

        for split_name, records in split_map.items():
            tsv_path = self.output_dir / f"{split_name}.tsv"
            self._write_tsv(tsv_path, records)
            logger.info("  Wrote %s (%d samples)", tsv_path.name, len(records))

        self._write_metadata(split_map)
        logger.info("Dataset ready at %s", self.output_dir)
        return self.output_dir

    def _process_split(
        self,
        ds,
        split_name: str,
        audio_dir: Path,
    ) -> list[dict]:
        """Convert one HF dataset split into a list of record dicts."""
        import soundfile as sf

        from psdn_sonar.utils.text_processing import normalize_text_unified

        records: list[dict] = []
        limit = min(len(ds), self.max_samples) if self.max_samples > 0 else len(ds)

        for i in range(limit):
            item = ds[i]
            text = item.get(self.dataset.text_column, "")
            if not text or not str(text).strip():
                continue

            text = str(text).strip()
            text_norm = normalize_text_unified(text, self.language)

            audio_data = item.get(self.dataset.audio_column)
            audio_path = ""
            duration_s = 0.0

            if audio_data and isinstance(audio_data, dict) and "array" in audio_data:
                wav_path = audio_dir / f"{split_name}_{i:06d}.wav"
                sr = audio_data.get("sampling_rate", 16000)
                sf.write(str(wav_path), audio_data["array"], sr)
                audio_path = str(wav_path.resolve())
                duration_s = len(audio_data["array"]) / sr
            elif audio_data and isinstance(audio_data, dict) and "path" in audio_data:
                audio_path = audio_data["path"]

            snr_db = ""
            if audio_path and not self.skip_audio_validation:
                snr_db = self._compute_snr(audio_path)

            records.append(
                {
                    "audio_path": audio_path,
                    "transcription": text,
                    "transcription_norm": text_norm,
                    "duration_s": f"{duration_s:.3f}" if duration_s else "",
                    "snr_db": snr_db,
                }
            )

            if (i + 1) % 500 == 0:
                logger.info("  Processed %d/%d in split '%s'", i + 1, limit, split_name)

        return records

    @staticmethod
    def _use_predefined_splits(all_records: dict[str, list[dict]]) -> dict[str, list[dict]]:
        """Map HF split names to standard train/val/test."""
        split_map: dict[str, list[dict]] = {}
        name_map = {"train": "train", "validation": "val", "test": "test"}

        for hf_name, canonical in name_map.items():
            if hf_name in all_records and all_records[hf_name]:
                split_map[canonical] = all_records[hf_name]

        for hf_name, recs in all_records.items():
            if hf_name not in name_map and recs:
                split_map[hf_name] = recs

        return split_map

    def _create_splits(self, records: list[dict]) -> dict[str, list[dict]]:
        """Deterministically split records into train/val/test."""
        shuffled = list(records)
        self._rng.shuffle(shuffled)

        total = len(shuffled)
        ratio_sum = sum(self.split_ratio)
        train_end = int(total * self.split_ratio[0] / ratio_sum)
        val_end = train_end + int(total * self.split_ratio[1] / ratio_sum)

        return {
            "train": shuffled[:train_end],
            "val": shuffled[train_end:val_end],
            "test": shuffled[val_end:],
        }

    def _write_tsv(self, path: Path, records: list[dict]) -> None:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(self.TSV_COLUMNS), delimiter="\t")
            writer.writeheader()
            writer.writerows(records)

    def _write_metadata(self, split_map: dict[str, list[dict]]) -> None:
        meta = {
            "source": self.dataset.hf_id,
            "source_revision": self.dataset.revision,
            "config": self.dataset.config,
            "language": self.language,
            "split_sizes": {k: len(v) for k, v in split_map.items()},
            "download_date": datetime.now(timezone.utc).isoformat(),
            "text_column": self.dataset.text_column,
            "audio_column": self.dataset.audio_column,
            "max_samples_per_split": self.max_samples or "all",
            "audio_validation": not self.skip_audio_validation,
        }
        meta_path = self.output_dir / "metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

    @staticmethod
    def _compute_snr(audio_path: str) -> str:
        """Compute SNR for a single audio file, returning empty string on failure."""
        try:
            import librosa

            from psdn_sonar.audio_quality import calculate_snr

            audio, _ = librosa.load(audio_path, sr=16000)
            snr = calculate_snr(audio)
            return f"{snr:.2f}" if snr is not None else ""
        except Exception:
            return ""
