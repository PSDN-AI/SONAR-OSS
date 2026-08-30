"""Core dataset-evaluation loops.

``process_dataset_with_asr`` evaluates an ASR model over a public benchmark
dataset (Common Voice, FLEURS, OpenSLR) and writes per-utterance dual-variant
scores plus a summary stats file. ``process_manifest_with_asr`` does the same
for multi-speaker manifest datasets, with preprocessing-method selection and
parallel audio-quality scoring.
"""

import csv
import glob
import logging
import math
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from .audio_quality import compute_audio_quality_metrics
from .config_loader import get_run_seed
from .evaluators.utterance import UtteranceEvaluator
from .loaders.common_voice import CommonVoiceLoader
from .loaders.fleurs import FleursLoader
from .loaders.openslr import OpenSLR53Loader, OpenSLRLineIndexLoader
from .models.base import ASRModel, unpack_transcription
from .utils.text_processing import normalize_text_unified

logger = logging.getLogger(__name__)

_DATASET_LOADERS: Dict[str, Any] = {
    "commonvoice": CommonVoiceLoader,
    "fleurs": FleursLoader,
    "openslr37_bd": OpenSLRLineIndexLoader,
    "openslr37_in": OpenSLRLineIndexLoader,
    "openslr53": OpenSLR53Loader,
}

_DATASET_NAME_ALIASES = {"common_voice": "commonvoice"}

# Accumulator keys: non-conversion and conversion variants of each metric.
_METRIC_KEYS = ("cer_n", "wer_n", "sem_n", "poseidon_n", "cer_c", "wer_c", "sem_c", "poseidon_c")

_EMPTY_AUDIO_QUALITY = {
    "snr_db": None,
    "clipping_ratio": None,
    "silence_ratio": None,
    "snr_tier": None,
    "quality_warnings": "",
}


def _mean_std(values: list) -> tuple[float, Optional[float]]:
    """Mean and sample standard deviation, ``(0.0, None)`` for empty input.

    The standard deviation is ``None`` below two samples because it is
    undefined there. It used to be reported as ``0.0000``, which asserts no
    spread from a single sample and disagreed with the CLI summary printing
    ``nan`` for the same quantity (issue #189).
    """
    if not values:
        return 0.0, None
    m = sum(values) / len(values)
    if len(values) < 2:
        return m, None
    s = math.sqrt(sum((x - m) ** 2 for x in values) / (len(values) - 1))
    return m, s


def _fmt_std(value: Optional[float]) -> str:
    """Format a standard deviation, naming an undefined one instead of faking it."""
    if value is None or not math.isfinite(value):
        return "n/a (n<2)"
    return f"{value:.4f}"


def _scored_metric_values(scored) -> tuple:
    """Dual-variant metric values in ``_METRIC_KEYS`` order."""
    return (
        scored.non.cer,
        scored.non.wer,
        scored.non.semantic_similarity,
        scored.non.poseidon_score,
        scored.conv.cer,
        scored.conv.wer,
        scored.conv.semantic_similarity,
        scored.conv.poseidon_score,
    )


def _accumulate(acc: Dict[str, list], values: tuple) -> None:
    # Accumulate at the CSV's own 4-decimal precision so the .txt summary,
    # the CLI table (which reads the written CSV), and anything a reader
    # recomputes from the artifact all agree — the unrounded accumulator
    # drifted from the CSV-derived mean in the fourth decimal (issue #212).
    for key, value in zip(_METRIC_KEYS, values):
        if value is not None:
            acc[key].append(round(value, 4))


def _stats_lines(acc: Dict[str, list], suffix: str) -> List[str]:
    """Formatted mean/std lines for one metric variant (``n`` or ``c``)."""
    lines = []
    for label, key in (("CER", f"cer_{suffix}"), ("WER", f"wer_{suffix}"), ("Sem", f"sem_{suffix}")):
        m, s = _mean_std(acc[key])
        lines.append(f"{label}: Mean {m:.4f}, Std {_fmt_std(s)}")
    m, s = _mean_std(acc[f"poseidon_{suffix}"])
    lines.append(f"POSEIDON: Mean {m:.4f}, Std {_fmt_std(s)}")
    return lines


@contextmanager
def _timed_per_clip_calls(asr_model, latencies: list):
    """Record ASR latency for the per-clip (diarization) strategies.

    Those strategies call ``transcribe_diarized`` /
    ``transcribe_with_word_timestamps`` directly on the adapter, so their
    ASR calls were never timed and ``inference_latency_s`` stayed empty for
    ``scribe_diarize``/``pyannote_diarize`` while the same adapter's
    per-channel runs populated it (issue #212). Exactly the model call is
    measured — pyannote diarization is not inference — into the same
    accumulator the per-channel wrapper appends to. The instrumentation is
    removed on exit so a reused adapter never reports into a stale list.
    """

    def _timed(bound):
        def timed(*args, **kwargs):
            # Same stale-cause clearing as the per-channel wrapper (issue #170).
            if hasattr(asr_model, "last_transcribe_error"):
                asr_model.last_transcribe_error = None
            t0 = time.perf_counter()
            out = bound(*args, **kwargs)
            latencies.append(round(time.perf_counter() - t0, 4))
            return out

        return timed

    wrapped = []
    for name in ("transcribe_diarized", "transcribe_with_word_timestamps"):
        bound = getattr(asr_model, name, None)
        if callable(bound):
            # Instance attributes are restored to their original value;
            # class-provided methods are un-shadowed with delattr.
            was_instance_attr = name in getattr(asr_model, "__dict__", {})
            wrapped.append((name, was_instance_attr, bound))
            setattr(asr_model, name, _timed(bound))
    try:
        yield
    finally:
        for name, was_instance_attr, bound in wrapped:
            if was_instance_attr:
                setattr(asr_model, name, bound)
            else:
                delattr(asr_model, name)


def process_dataset_with_asr(
    dataset_name: str,
    dataset_dir: str,
    asr_model: ASRModel,
    output_tsv: str,
    root_dir: Optional[str] = None,
    max_samples: int = 0,
    asr_model_name: Optional[str] = None,
    language: str = "bn",
):
    """Evaluate *asr_model* over a public dataset and write per-utterance scores.

    Writes a TSV/CSV of dual-variant scores (with and without script
    conversion) plus a ``.txt`` stats summary next to it. ``max_samples > 0``
    evaluates a seeded random subset.
    """
    dataset_key = dataset_name.lower()
    dataset_key = _DATASET_NAME_ALIASES.get(dataset_key, dataset_key)
    loader = _DATASET_LOADERS.get(dataset_key)
    if not loader:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    if root_dir is None:
        root_dir = os.path.dirname(os.path.abspath(output_tsv))

    logger.info(f"Loading metadata and audio files from {dataset_dir}")
    if dataset_key == "commonvoice":
        all_metadata = loader.load_metadata(dataset_dir, language=language)
    else:
        all_metadata = loader.load_metadata(dataset_dir)

    if dataset_key == "commonvoice":
        audio_files = []
        for f in glob.glob(os.path.join(dataset_dir, language, "clips", "*.mp3")):
            fid = os.path.splitext(os.path.basename(f))[0]
            if fid in all_metadata:
                audio_files.append((f, fid, os.path.relpath(f, dataset_dir).replace("\\", "/")))
    elif dataset_key == "fleurs":
        audio_files = []
        for rel in all_metadata:
            p = os.path.join(dataset_dir, "test", "audio", rel)
            if not os.path.exists(p):
                p = os.path.join(dataset_dir, "test", rel)
            if os.path.exists(p):
                audio_files.append(("test", p, rel))
    else:
        all_audio = loader.find_audio_files(dataset_dir)
        audio_files = [x for x in all_audio if x[2] in all_metadata]

    if max_samples > 0 and len(audio_files) > max_samples:
        audio_files = random.Random(get_run_seed()).sample(audio_files, max_samples)

    logger.info(f"Processing {len(audio_files)} samples")

    fieldnames = loader.get_output_fieldnames()
    os.makedirs(os.path.dirname(output_tsv) or ".", exist_ok=True)
    delimiter = "," if output_tsv.endswith(".csv") else "\t"

    processed_count = 0
    acc: Dict[str, list] = {k: [] for k in _METRIC_KEYS}

    with open(output_tsv, "w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames, delimiter=delimiter)
        writer.writeheader()

        for idx, info in enumerate(tqdm(audio_files, desc="  Evaluating", unit="sample", leave=True)):
            if dataset_key == "commonvoice":
                path, fid, rel = info
                meta = all_metadata[fid]
            elif dataset_key == "fleurs":
                _, path, rel = info
                meta = all_metadata[rel]
            else:
                _, path, fid, rel = info
                meta = all_metadata[fid]

            transcription = loader.get_transcription_from_metadata(meta)
            if not transcription:
                continue

            try:
                logger.info(f"[{idx + 1}/{len(audio_files)}] Processing {rel}")

                t0 = time.perf_counter()
                raw_result = asr_model.transcribe(path)
                latency = round(time.perf_counter() - t0, 4)
                asr_raw, _ = unpack_transcription(raw_result)
                asr_raw = asr_raw or ""

                scored = UtteranceEvaluator.score_dual_variant(transcription, asr_raw, language=language)
                values = _scored_metric_values(scored)
                cer_n, wer_n, sem_n, poseidon_n, cer_c, wer_c, sem_c, poseidon_c = values

                writer.writerow(
                    loader.create_output_row(
                        meta,
                        transcription,
                        scored.ref_norm,
                        asr_raw,
                        scored.hyp_norm_non,
                        scored.hyp_norm_conv,
                        cer_n,
                        wer_n,
                        sem_n,
                        poseidon_n,
                        cer_c,
                        wer_c,
                        sem_c,
                        poseidon_c,
                        rel,
                        inference_latency_s=latency,
                    )
                )
                processed_count += 1
                _accumulate(acc, values)

            except Exception as e:
                logger.error(f"Error processing {rel}: {e}", exc_info=True)

    if processed_count > 0:
        m_cer_n, _ = _mean_std(acc["cer_n"])
        m_wer_n, _ = _mean_std(acc["wer_n"])
        m_cer_c, _ = _mean_std(acc["cer_c"])
        m_wer_c, _ = _mean_std(acc["wer_c"])
        logger.info(f"[Non-Conversion] CER: {m_cer_n:.4f}, WER: {m_wer_n:.4f}")
        logger.info(f"[Conversion]     CER: {m_cer_c:.4f}, WER: {m_wer_c:.4f}")

        stats_file = output_tsv.rsplit(".", 1)[0] + ".txt"
        with open(stats_file, "w", encoding="utf-8") as f:
            f.write(f"ASR Evaluation Results\n{'=' * 30}\n")
            f.write(f"Model: {asr_model_name}\nDataset: {dataset_name}\nSamples: {processed_count}\n\n")
            f.write("--- Results (Without Script Conversion) ---\n")
            f.write("\n".join(_stats_lines(acc, "n")) + "\n\n")
            f.write("--- Results (With Script Conversion) ---\n")
            f.write("\n".join(_stats_lines(acc, "c")) + "\n")

        logger.info(f"Stats saved to: {stats_file}")


def process_manifest_with_asr(
    manifest_path: str,
    asr_model: ASRModel,
    output_csv: str,
    max_samples: int = 0,
    asr_model_name: Optional[str] = None,
    language: str = "bn",
    methods: Optional[List[str]] = None,
    config_settings: Optional[Dict] = None,
    sweep: bool = False,
    method: Optional[str] = None,
):
    """Evaluate *asr_model* over a multi-speaker manifest dataset.

    Args:
        manifest_path: Path to manifest.jsonl file
        asr_model: Initialized ASR model
        output_csv: Path for output CSV file
        max_samples: Max clips to process (0 = all)
        asr_model_name: Model name for reporting
        methods: List of preprocessing method names (candidate pool)
        config_settings: Dict with silence/timestamp/pyannote settings
        sweep: If True, run all methods and pick the best using ground truth
               (oracle bias — inflates metrics; use only for ablations).
        method: Explicit method name to use for all clips. If None and sweep
                is False, method is auto-selected per clip based on available data.
    """
    from .loaders.manifest import (
        create_output_row,
        get_clip_files,
        get_output_fieldnames,
        load_manifest,
        load_transcript_with_segments,
    )
    from .preprocessing.audio_utils import get_combined_audio_path
    from .preprocessing.methods import PER_CLIP_METHODS, PYANNOTE_METHODS
    from .preprocessing.preprocessing_selector import run_single_method, run_sweep
    from .preprocessing.pyannote_utils import PYANNOTE_AVAILABLE

    if sweep:
        logger.warning(
            "Running in sweep mode (--sweep): all methods are scored against ground truth to select "
            "the best per clip. This introduces oracle bias and will inflate reported metrics. "
            "Do not use sweep results as production benchmarks."
        )

    if methods is None:
        methods = ["energy_trim", "timestamp_trim", "no_trim"]

    config_settings = config_settings or {}
    silence_settings = config_settings.get("silence", {})
    timestamp_settings = config_settings.get("timestamp", {})
    pyannote_settings = config_settings.get("pyannote", {})

    active_methods = []
    for m in methods:
        if m in PYANNOTE_METHODS and not PYANNOTE_AVAILABLE:
            logger.warning(
                f"Skipping {m}: pyannote.audio not installed. Install with: pip install 'psdn-sonar[pyannote]'"
            )
            continue
        if m in PER_CLIP_METHODS and not getattr(asr_model, "supports_diarization", False):
            logger.warning(f"Skipping {m}: {asr_model_name or type(asr_model).__name__} does not support diarization")
            continue
        active_methods.append(m)

    if not active_methods:
        raise ValueError("No valid preprocessing methods available")

    entries = load_manifest(manifest_path)
    logger.info(f"Loaded {len(entries)} clips from manifest: {manifest_path}")

    if max_samples > 0 and len(entries) > max_samples:
        total_entries = len(entries)
        entries = random.Random(get_run_seed()).sample(entries, max_samples)
        logger.info(f"Sampled {max_samples} clips from {total_entries} total")

    transcribe_latencies: list[float] = []

    def transcribe_with_latency(audio_path: str) -> str:
        # Clear any stale failure cause so a previous call's error is never
        # attributed to this one (same protocol as the single-speaker path,
        # issue #170).
        if hasattr(asr_model, "last_transcribe_error"):
            asr_model.last_transcribe_error = None
        t0 = time.perf_counter()
        raw_result = asr_model.transcribe(audio_path)
        transcribe_latencies.append(round(time.perf_counter() - t0, 4))
        text, _ = unpack_transcription(raw_result)
        if not text:
            # Adapters record the cause and return None on failure (issue
            # #178). A failed transcription is not an empty hypothesis, so it
            # must not be scored as if the model heard silence (issue #181):
            # raising routes the cause into the per-speaker error plumbing,
            # and the row comes out failed with the reason in its error
            # column. A genuinely empty prediction (no recorded cause) still
            # scores WER/CER 1.0 per the benchmark README convention.
            cause = getattr(asr_model, "last_transcribe_error", None)
            if cause:
                raise RuntimeError(f"Transcription failed: {cause}")
        return text or ""

    def compute_metrics(ref_text: str, hyp_text: str):
        scored = UtteranceEvaluator.score_normalized_pair(
            normalize_text_unified(ref_text, language=language),
            normalize_text_unified(hyp_text, language=language),
        )
        return scored.cer, scored.wer, scored.semantic_similarity, scored.poseidon_score

    fieldnames = get_output_fieldnames()
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)

    processed_count = 0
    failed_count = 0
    acc: Dict[str, list] = {k: [] for k in _METRIC_KEYS}

    with (
        open(output_csv, "w", newline="", encoding="utf-8") as outfile,
        _timed_per_clip_calls(asr_model, transcribe_latencies),
    ):
        writer = csv.DictWriter(outfile, fieldnames=fieldnames, delimiter=",")
        writer.writeheader()

        def write_failure_row(
            clip_id: str,
            speaker: str,
            ref_text: str,
            aq: dict,
            best_method: str = "",
            path: str = "",
            transcription_norm: str = "",
            latency: Optional[float] = None,
            error: str = "",
        ) -> None:
            """Row with audio-quality data but no scores, for clips that could not be evaluated.

            Counted in the failed total and never in ``processed_count``, with
            the reason in the ``error`` column — the same convention the
            single-speaker path applies (issue #181).
            """
            nonlocal failed_count
            failed_count += 1
            writer.writerow(
                create_output_row(
                    audio_id=clip_id,
                    speaker=speaker,
                    best_method=best_method,
                    path=path,
                    transcription=ref_text,
                    transcription_norm=transcription_norm,
                    asr_transcription="",
                    asr_norm_non="",
                    asr_norm_conv="",
                    cer_non=None,
                    wer_non=None,
                    sem_non=None,
                    poseidon_non=None,
                    cer_conv=None,
                    wer_conv=None,
                    sem_conv=None,
                    poseidon_conv=None,
                    original_duration_s=0.0,
                    trimmed_duration_s=0.0,
                    snr_db=aq.get("snr_db"),
                    clipping_ratio=aq.get("clipping_ratio"),
                    silence_ratio=aq.get("silence_ratio"),
                    snr_tier=aq.get("snr_tier"),
                    quality_warnings=aq.get("quality_warnings", ""),
                    inference_latency_s=latency,
                    error=error,
                )
            )

        for entry in tqdm(entries, desc="  Evaluating", unit="clip", leave=True):
            clip_id = entry.audio_id
            logger.debug(f"Processing clip: {clip_id}")

            audio_a, audio_b, transcript_path = get_clip_files(entry)

            if not transcript_path or not transcript_path.exists():
                logger.warning(f"Skipping {clip_id}: could not load transcript")
                continue
            if (not audio_a or not audio_a.exists()) and (not audio_b or not audio_b.exists()):
                logger.warning(f"Skipping {clip_id}: both audio files missing")
                continue

            try:
                ref_a, ref_b, segments = load_transcript_with_segments(transcript_path)
            except Exception as e:
                logger.warning(f"Skipping {clip_id}: could not load transcript: {e}")
                continue

            combined_audio = get_combined_audio_path(entry)

            # Audio quality runs in background threads while preprocessing + ASR
            # runs on the main thread.
            aq_futures = {}
            lat_start_idx = len(transcribe_latencies)
            eval_error = None

            with ThreadPoolExecutor(max_workers=2) as aq_exec:
                if audio_a and audio_a.exists():
                    aq_futures["A"] = aq_exec.submit(compute_audio_quality_metrics, str(audio_a))
                if audio_b and audio_b.exists():
                    aq_futures["B"] = aq_exec.submit(compute_audio_quality_metrics, str(audio_b))

                shared_kwargs: Dict[str, Any] = dict(
                    entry=entry,
                    asr_model=asr_model,
                    ref_a=ref_a,
                    ref_b=ref_b,
                    segments=segments,
                    audio_a=audio_a,
                    audio_b=audio_b,
                    combined_audio=combined_audio,
                    metric_fn=compute_metrics,
                    transcribe_fn=transcribe_with_latency,
                    silence_settings=silence_settings,
                    timestamp_settings=timestamp_settings,
                    pyannote_settings=pyannote_settings,
                )
                try:
                    if sweep:
                        all_results, best_a, best_b = run_sweep(methods=active_methods, **shared_kwargs)
                    else:
                        all_results, best_a, best_b = run_single_method(
                            active_methods=active_methods, method_name=method, **shared_kwargs
                        )
                except Exception as e:
                    eval_error = e
                    logger.error(f"Error processing {clip_id}: {e}", exc_info=True)

                aq_map = {}
                for spk in ("A", "B"):
                    if spk in aq_futures:
                        try:
                            aq_map[spk] = aq_futures[spk].result(timeout=120)
                        except Exception:
                            aq_map[spk] = _EMPTY_AUDIO_QUALITY.copy()
                    else:
                        aq_map[spk] = _EMPTY_AUDIO_QUALITY.copy()

            if eval_error is not None:
                for speaker, ref_text in [("A", ref_a), ("B", ref_b)]:
                    write_failure_row(clip_id, speaker, ref_text, aq_map[speaker], error=str(eval_error))
                continue

            clip_latencies = transcribe_latencies[lat_start_idx:]
            clip_latency = round(sum(clip_latencies), 4) if clip_latencies else None

            if best_a is None and best_b is None:
                logger.warning(f"No valid methods for {clip_id}, skipping")
                for speaker, ref_text in [("A", ref_a), ("B", ref_b)]:
                    # The selector records the actionable per-speaker reason
                    # (e.g. the pyannote install hint) in all_results; surface
                    # it instead of a generic placeholder.
                    reason = next((r.get("error") for r in all_results.get(speaker, []) if r.get("error")), None)
                    write_failure_row(
                        clip_id,
                        speaker,
                        ref_text,
                        aq_map[speaker],
                        latency=clip_latency,
                        error=reason or "No preprocessing method produced a result for this clip",
                    )
                continue

            for speaker, best_result, ref_text, audio_path in [
                ("A", best_a, ref_a, audio_a),
                ("B", best_b, ref_b, audio_b),
            ]:
                aq = aq_map[speaker]
                if best_result is None or best_result.get("error"):
                    write_failure_row(
                        clip_id,
                        speaker,
                        ref_text,
                        aq,
                        best_method=best_result.get("method", "") if best_result else "",
                        path=str(audio_path or ""),
                        transcription_norm=normalize_text_unified(ref_text, language=language),
                        latency=clip_latency,
                        error=(best_result.get("error") or "") if best_result else "No result for this speaker",
                    )
                    continue

                asr_raw = best_result.get("text", "")

                scored = UtteranceEvaluator.score_dual_variant(ref_text, asr_raw, language=language)
                values = _scored_metric_values(scored)

                # Per-method score summary; missing error rates count as worst
                # case (1.0) via explicit None checks so a legitimate 0.0 survives.
                method_scores = {}
                for r in all_results.get(speaker, []):
                    if r.get("error") is None and r.get("cer") is not None:
                        c = r["cer"]
                        w = r.get("wer") if r.get("wer") is not None else 1.0
                        s = r.get("similarity") if r.get("similarity") is not None else 0.0
                        method_scores[r.get("method", "unknown")] = round(((1 - c) + (1 - w) + s) / 3, 4)

                writer.writerow(
                    create_output_row(
                        audio_id=clip_id,
                        speaker=speaker,
                        best_method=best_result.get("method", ""),
                        path=str(audio_path or ""),
                        transcription=ref_text,
                        transcription_norm=scored.ref_norm,
                        asr_transcription=asr_raw,
                        asr_norm_non=scored.hyp_norm_non,
                        asr_norm_conv=scored.hyp_norm_conv,
                        cer_non=values[0],
                        wer_non=values[1],
                        sem_non=values[2],
                        poseidon_non=values[3],
                        cer_conv=values[4],
                        wer_conv=values[5],
                        sem_conv=values[6],
                        poseidon_conv=values[7],
                        original_duration_s=best_result.get("original_duration_s", 0.0),
                        trimmed_duration_s=best_result.get("trimmed_duration_s", 0.0),
                        snr_db=aq.get("snr_db"),
                        clipping_ratio=aq.get("clipping_ratio"),
                        inference_latency_s=clip_latency,
                        silence_ratio=aq.get("silence_ratio"),
                        snr_tier=aq.get("snr_tier"),
                        quality_warnings=aq.get("quality_warnings", ""),
                        all_method_scores=method_scores if method_scores else None,
                    )
                )

                processed_count += 1
                _accumulate(acc, values)

    if failed_count:
        logger.warning(
            f"Successful: {processed_count}, Failed: {failed_count} — "
            f"per-row reasons are in the error column of {output_csv}"
        )

    if processed_count > 0:
        m_cer_c, _ = _mean_std(acc["cer_c"])
        m_wer_c, _ = _mean_std(acc["wer_c"])
        logger.info(f"[Combined] CER: {m_cer_c:.4f}, WER: {m_wer_c:.4f}")

        stats_file = output_csv.rsplit(".", 1)[0] + ".txt"
        with open(stats_file, "w", encoding="utf-8") as f:
            f.write(f"Multi-Speaker ASR Evaluation Results\n{'=' * 40}\n")
            f.write(f"Model: {asr_model_name}\nManifest: {manifest_path}\n")
            if sweep:
                mode_str = f"sweep ({', '.join(active_methods)}) [ORACLE BIAS]"
            elif method:
                mode_str = f"fixed:{method}"
            else:
                mode_str = f"auto ({', '.join(active_methods)})"
            f.write(f"Mode: {mode_str}\nSamples: {processed_count}\nFailed: {failed_count}\n\n")
            f.write("--- Combined ---\n")
            f.write("\n".join(_stats_lines(acc, "c")) + "\n")

        logger.info(f"Stats saved to: {stats_file}")
    else:
        # A run that scored nothing must not look like a completed evaluation
        # (issue #102): raise so callers (and the CLI) exit non-zero. The CSV
        # written above still carries the header plus any per-clip error rows.
        raise RuntimeError(
            "No clips were successfully processed — 0 evaluated rows. "
            f"See the warnings above and the error column of the per-clip rows in {output_csv} "
            "for the reasons (transcription failures, missing transcripts/audio, "
            "preprocessing failures, or missing optional dependencies)."
        )
