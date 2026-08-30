"""
Single speaker audio dataset evaluation: Evaluate multiple models on single-speaker datasets.
"""

from __future__ import annotations

import csv
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, cast

from psdn_sonar.audio_quality import compute_audio_quality_metrics
from psdn_sonar.config import load_env
from psdn_sonar.evaluators.utterance import UtteranceEvaluator
from psdn_sonar.models.base import unpack_transcription
from psdn_sonar.models.registry import LANGUAGE_DEFAULT_MODELS as _LANG_DEFAULTS
from psdn_sonar.models.registry import UnknownModelError, create_model, list_models
from psdn_sonar.quality_models import _EMPTY_MOS
from psdn_sonar.utils.metrics import (
    DEFAULT_SIGNIFICANT_WER_THRESHOLD,
    calculate_poseidon_score,
    clamp_similarity,
    compute_latency_summary,
    compute_protocol_latency_summary,
    is_significant_wer,
)

if TYPE_CHECKING:
    from psdn_sonar.benchmark.submission import SubmissionConfig

logger = logging.getLogger(__name__)


class NoSamplesEvaluatedError(RuntimeError):
    """Raised when an evaluation run finishes with zero successful samples.

    Artifacts (per-utterance CSV, ``scores_*.json``) are still written before
    this is raised, so per-row errors remain inspectable; the exception exists
    so a run that scored nothing cannot exit 0 and masquerade as a clean run.
    """


# One line with the known remedy, matching the other dependency paths
# (#169/#177): without it a core-only install ran to completion with the
# headline metrics silently null and nothing in any artifact saying why
# (issue #191).
_SEMANTICS_MISSING_EXTRA_WARNING = (
    "semantic_similarity and poseidon_score were not computed: "
    "sentence-transformers is not installed (a core install without the [ml] extra). "
    'WER/CER are unaffected. Install with: pip install "psdn-sonar[ml]"'
)


def _semantics_dependency_missing() -> bool:
    """True when the [ml] extra needed for semantic similarity is absent."""
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        return True
    return False


_EMPTY_AUDIO_QUALITY = {
    "snr_db": None,
    "clipping_ratio": None,
    "silence_ratio": None,
    "snr_tier": None,
    "quality_warnings": "",
    **_EMPTY_MOS,
}


def _default_submission_for_model(
    model,
    model_name: str,
    *,
    language: str,
) -> SubmissionConfig:
    """Build the submission block from the model that actually ran (issue #184).

    - ``provider`` comes from the adapter (``openai``/``elevenlabs``/
      ``assemblyai`` for hosted APIs, ``local`` for in-process inference); it
      used to come from the undocumented ``SONAR_PROVIDER`` env var with a
      ``local`` default, so a hosted-API run recorded ``provider: local``.
      The env var remains as an explicit override (documented in .env.example).
    - ``region`` is null unless ``SONAR_REGION`` is set — hosted providers do
      not disclose one and a local run has none, so the old ``local`` default
      asserted a region nobody measured.
    - ``model_snapshot`` records the provider-side model id actually requested
      (e.g. ``whisper-1``) when the adapter pins one, falling back to the
      registry alias; ``scores.json`` carries the alias separately as
      ``model_name``.
    - ``judge_model``/``prompt_version`` are never set here: this pipeline
      computes semantic similarity locally and runs no LLM judge, and the old
      ``compute_sem`` gate stamped the judge rubric hash onto every POSEIDON
      run — asserting LLM-judged metrics that were never computed. A caller
      that actually runs the judge passes its own ``submission``.
    """
    from psdn_sonar.benchmark.submission import SubmissionConfig

    # Protocol comes from the model that ran (an adapter in streaming mode
    # measures ttft_s); SONAR_PROTOCOL remains as an explicit override.
    env_protocol = os.getenv("SONAR_PROTOCOL")
    protocol: Literal["batch", "streaming"]
    if env_protocol in ("batch", "streaming"):
        protocol = cast('Literal["batch", "streaming"]', env_protocol)
    else:
        protocol = "streaming" if getattr(model, "streaming", False) else "batch"
    return SubmissionConfig.from_env(
        provider=os.getenv("SONAR_PROVIDER") or getattr(model, "provider", None) or "local",
        model_snapshot=getattr(model, "provider_model_id", None) or model_name,
        region=os.getenv("SONAR_REGION") or None,
        protocol=protocol,
        inference_params={"language_code": language},
    )


def _run_lineage(model, language: str):
    """Build the best-effort ``RunLineage`` block for ``scores.json``.

    Never fails the run: test doubles and third-party adapters may not
    implement ``get_hf_lineage``, and lineage is diagnostic metadata.
    """
    from psdn_sonar.benchmark.scores import RunLineage
    from psdn_sonar.utils.text_processing import wer_normalization_contract

    hf_model_id = None
    hf_revision = None
    try:
        resolved = model.get_hf_lineage()
        if isinstance(resolved, tuple) and len(resolved) == 2:
            candidate_id, candidate_revision = resolved
            hf_model_id = candidate_id if isinstance(candidate_id, str) else None
            hf_revision = candidate_revision if isinstance(candidate_revision, str) else None
    except Exception:
        logger.debug("Could not resolve HF lineage for %s", type(model).__name__, exc_info=True)

    return RunLineage(
        hf_model_id=hf_model_id,
        hf_revision=hf_revision,
        normalization=wer_normalization_contract(language),
    )


def _model_factory(
    name: str,
    kwargs: Optional[dict] = None,
    custom_hf_model: Optional[str] = None,
    language: Optional[str] = None,
    streaming: Optional[bool] = None,
):
    """Create an ASR model by name. Delegates to the centralized ModelRegistry.

    Returns ``None`` only for a name the registry does not know. Any other
    constructor failure — including a missing API key, which adapters raise as
    a plain ``ValueError`` — propagates, so the caller logs the adapter's own
    actionable message instead of "not found in the registry" (issue #168).
    """
    try:
        return create_model(
            name, custom_hf_model=custom_hf_model, language=language, streaming=streaming, **(kwargs or {})
        )
    except UnknownModelError:
        return None


def _resolve_audio_path(
    audio_path: str,
    dataset_root: Path,
    allow_absolute_audio_paths: bool,
) -> str:
    """Resolve a TSV ``audio_path`` against the dataset root.

    A relative path that resolves outside the dataset root (``../`` traversal)
    is always rejected — this used to be gated behind the strict mode, so a
    TSV received from someone else could silently read audio from anywhere on
    disk (issue #127). Absolute paths remain allowed by default because they
    are explicit in the TSV and SONAR's own dataset preparer writes them;
    strict mode (``allow_absolute_audio_paths=False``, ``--strict-audio-paths``
    on the CLI) additionally rejects absolute paths and requires each path to
    be an existing regular file.
    """
    candidate = Path(audio_path)
    if candidate.is_absolute():
        if not allow_absolute_audio_paths:
            raise ValueError(f"audio_path must be relative inside bundle: {audio_path}")
        resolved = candidate.resolve()
    else:
        resolved = (dataset_root / candidate).resolve()
        if not resolved.is_relative_to(dataset_root):
            raise ValueError(
                f"audio_path escapes dataset root: {audio_path} "
                f"(resolves to {resolved}, outside {dataset_root}). "
                f"Use a path inside the TSV's directory, or an explicit absolute path."
            )

    if not allow_absolute_audio_paths:
        if not resolved.exists():
            raise FileNotFoundError(f"audio_path does not exist: {audio_path}")
        if not resolved.is_file():
            raise ValueError(f"audio_path is not a regular file: {audio_path}")

    return str(resolved)


_DEFAULT_AQ_MAX_WORKERS = 4


def _resolve_aq_workers() -> int:
    """Resolve AQ ThreadPoolExecutor max_workers from ``SONAR_AQ_MAX_WORKERS``.

    Each AQ worker holds an audio waveform plus UTMOS / SQUIM / DNSMOS
    intermediate activations for one clip; on long clips (>30 s) this peaks
    around 1.5-2 GB per worker. The historical default of 4 fits comfortably
    on typical hardware but OOMs resource-constrained environments (12 Gi
    RAM). The override lets operators dial it down without a code change.

    Falls back to the historical default on missing or invalid values, and
    clamps below 1 to 1 so the executor always has at least one worker.
    """
    raw = os.getenv("SONAR_AQ_MAX_WORKERS")
    if raw is None:
        return _DEFAULT_AQ_MAX_WORKERS
    try:
        value = int(raw)
    except ValueError:
        return _DEFAULT_AQ_MAX_WORKERS
    return max(1, value)


class SingleSpeakerEvaluator:
    DEFAULT_TSV = "data/eval.tsv"
    DEFAULT_OUTPUT_DIR = "results/asr_evaluation"

    AVAILABLE_MODELS = list_models()
    LANGUAGE_DEFAULT_MODELS = _LANG_DEFAULTS

    @staticmethod
    def load_data(tsv_path: str, allow_absolute_audio_paths: bool = True) -> List[Dict]:
        """Load TSV with audio_path and transcription columns.

        Rows with a missing/blank ``audio_path`` or ``transcription``, or with
        MORE fields than the header (a literal tab inside a field — the
        surplus used to be discarded silently, truncating the reference,
        issue #141), are not silently dropped: they are returned with a
        ``load_error`` key so the evaluator can count them as failed and emit
        an error row, keeping the artifacts honest about how much of the
        input was actually evaluated.

        The file is read as ``utf-8-sig``, so the UTF-8 BOM that Excel
        prepends to exported TSVs is stripped instead of corrupting the first
        column name into an invisible ``\\ufeffaudio_path`` and producing a
        "missing required columns" error for a column that is present
        (issue #141).

        Relative ``audio_path`` values that resolve outside the TSV's
        directory always raise ``ValueError`` (issue #127). With
        ``allow_absolute_audio_paths=False``, absolute paths are rejected too
        and every path must be an existing regular file.
        """
        data = []
        path = Path(tsv_path)
        dataset_root = path.parent.resolve()

        with open(path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f, delimiter="\t")
            fieldnames = reader.fieldnames or []
            missing_columns = [column for column in ("audio_path", "transcription") if column not in fieldnames]
            if missing_columns:
                raise ValueError(f"TSV missing required columns: {', '.join(missing_columns)}")

            # Header is line 1; the first data row is line 2.
            for line_num, row in enumerate(reader, start=2):
                ap = (row.get("audio_path") or "").strip()
                gt = (row.get("transcription") or "").strip()
                # DictReader parks fields beyond the header under the None
                # restkey. Keeping only the aligned columns would silently
                # truncate the reference, so the row is malformed input.
                surplus = row.get(None)
                if surplus:
                    load_error = (
                        f"TSV line {line_num}: {len(fieldnames) + len(surplus)} fields for "
                        f"{len(fieldnames)} header columns (a literal tab inside a field?) — "
                        f"refusing to truncate the transcription to {gt!r}"
                    )
                    logger.warning("%s — row will be counted as failed, not dropped", load_error)
                    data.append({"audio_path": ap, "ground_truth": gt, "load_error": load_error})
                    continue
                if not ap or not gt:
                    blank = "audio_path" if not ap else "transcription"
                    load_error = f"TSV line {line_num}: missing or empty {blank}"
                    logger.warning("%s — row will be counted as failed, not dropped", load_error)
                    data.append({"audio_path": ap, "ground_truth": gt, "load_error": load_error})
                    continue
                data.append(
                    {
                        "audio_path": _resolve_audio_path(
                            ap,
                            dataset_root=dataset_root,
                            allow_absolute_audio_paths=allow_absolute_audio_paths,
                        ),
                        "ground_truth": gt,
                    }
                )
        return data

    @staticmethod
    def _result_row(
        audio_path: str,
        ground_truth: str,
        aq: Dict,
        *,
        prediction: str = "",
        normalized_reference: str = "",
        normalized_hypothesis: str = "",
        wer: Optional[float] = None,
        cer: Optional[float] = None,
        significant_wer: Optional[bool] = None,
        inference_latency_s: Optional[float] = None,
        ttft_s: Optional[float] = None,
        complete_s: Optional[float] = None,
        error: Optional[str] = None,
    ) -> Dict:
        """Build one per-utterance result row.

        Key order is the CSV column order (the writer derives fieldnames from
        the first row), so all rows must come from this single builder.

        ``normalized_reference`` / ``normalized_hypothesis`` are the exact
        strings CER/WER were computed over (issue #143) — without them,
        problems that are invisible in the raw text (e.g. a zero-width space
        surviving normalization in the reference) cannot be diagnosed from
        the artifact. They are empty on rows where scoring never ran.
        """
        return {
            "audio_path": audio_path,
            "ground_truth": ground_truth,
            "prediction": prediction,
            "normalized_reference": normalized_reference,
            "normalized_hypothesis": normalized_hypothesis,
            "wer": wer,
            "cer": cer,
            "semantic_similarity": None,
            "poseidon_score": None,
            "significant_wer": significant_wer,
            "inference_latency_s": inference_latency_s,
            "ttft_s": ttft_s,
            "complete_s": complete_s,
            **aq,
            "error": error,
        }

    @staticmethod
    def _compute_audio_quality(item: Dict) -> tuple:
        """Audio-quality metrics for one sample; empty metrics for missing files."""
        ap = item["audio_path"]
        if ap and os.path.exists(ap):
            return ap, compute_audio_quality_metrics(ap)
        return ap, dict(_EMPTY_AUDIO_QUALITY)

    @classmethod
    def _csv_fieldnames(cls) -> List[str]:
        """Canonical per-utterance CSV column order (from the row builder)."""
        return list(cls._result_row("", "", dict(_EMPTY_AUDIO_QUALITY)).keys())

    @staticmethod
    def _apply_batch_semantics(results: List[Dict], sem_pairs: List[tuple]) -> Optional[str]:
        """Encode all deferred similarity pairs in one batched model.encode() call.

        Fills ``semantic_similarity`` and (when CER/WER are present)
        ``poseidon_score`` in-place on the referenced rows.

        Returns ``None`` on success, or a run-warning string when the metrics
        could not be filled, so the caller can record it in ``scores.json``
        (issue #191): the one-line missing-extra remedy for an absent
        sentence-transformers (a known fix, no traceback), or the failure
        cause for a genuine runtime error (which keeps its traceback in the
        log).
        """
        if not sem_pairs:
            return None
        try:
            from sentence_transformers import util

            from psdn_sonar.utils.metrics import _get_semantic_model

            sem_model = _get_semantic_model()
            all_texts = [t for _, gt, pred in sem_pairs for t in (gt, pred or "")]
            embeds = sem_model.encode(all_texts, convert_to_tensor=False, show_progress_bar=False)
            for i, (result_idx, _, _) in enumerate(sem_pairs):
                e1, e2 = embeds[2 * i], embeds[2 * i + 1]
                # Clamped to [0, 1] at the source so the stored value, its
                # mean, and the POSEIDON input share one range (issue #107).
                sim = clamp_similarity(float(util.cos_sim(e1[None], e2[None])[0][0]))
                r = results[result_idx]
                results[result_idx]["semantic_similarity"] = sim
                if r["cer"] is not None and r["wer"] is not None:
                    results[result_idx]["poseidon_score"] = calculate_poseidon_score(r["cer"], r["wer"], sim)
        except ImportError:
            logger.warning(_SEMANTICS_MISSING_EXTRA_WARNING)
            return _SEMANTICS_MISSING_EXTRA_WARNING
        except Exception as exc:
            logger.warning("Batch semantic similarity failed", exc_info=True)
            return (
                "semantic_similarity and poseidon_score were not computed: "
                f"batch semantic similarity failed ({exc}). WER/CER are unaffected; "
                "the traceback is in the run log."
            )
        return None

    @staticmethod
    def evaluate_one(
        model: Any,
        data: List[Dict],
        model_name: str,
        max_samples: int = 0,
        compute_sem: bool = False,
        language: str = "bn",
        significant_wer_threshold: float = DEFAULT_SIGNIFICANT_WER_THRESHOLD,
    ) -> Dict:
        """Evaluate one model and return results.

        ``significant_wer_threshold`` is the WER value at and above which
        an utterance is flagged as a significant error in the per-row
        ``significant_wer`` column and the run-level
        ``significant_wer_rate`` aggregate.
        """
        if max_samples > 0:
            data = data[:max_samples]

        results = []
        total_wer = 0.0
        total_cer = 0.0
        successful = 0
        failed = 0
        start = time.time()

        logger.info(f"Starting evaluation for model: {model_name}")
        logger.info(f"Total samples to process: {len(data)}")

        # Pairs deferred for a single batched model.encode() call after the loop.
        # Each entry: (result_idx, gt_norm, pred_norm)
        _sem_pairs: List[tuple] = []

        aq_max_workers = min(_resolve_aq_workers(), os.cpu_count() or 1)
        with ThreadPoolExecutor(max_workers=aq_max_workers) as executor:
            _aq_cache = dict(executor.map(SingleSpeakerEvaluator._compute_audio_quality, data))

        from tqdm import tqdm

        for idx, item in enumerate(tqdm(data, desc=f"  {model_name}", unit="sample", leave=True), 1):
            audio_path = item["audio_path"]
            ground_truth = item["ground_truth"]

            aq = _aq_cache.get(audio_path, dict(_EMPTY_AUDIO_QUALITY))

            load_error = item.get("load_error")
            if load_error:
                failed += 1
                logger.warning(f"[{idx}/{len(data)}] Skipping transcription: {load_error}")
                results.append(
                    SingleSpeakerEvaluator._result_row(
                        audio_path,
                        ground_truth,
                        aq,
                        error=load_error,
                    )
                )
                continue

            inference_latency_s = None
            ttft_s = None
            complete_s = None
            try:
                # Clear any stale failure cause so a previous clip's error is
                # never attributed to this one (issue #170).
                if hasattr(model, "last_transcribe_error"):
                    model.last_transcribe_error = None
                if not os.path.exists(audio_path):
                    logger.warning(f"[{idx}/{len(data)}] Audio file not found: {audio_path}")
                    prediction = ""
                else:
                    t0 = time.perf_counter()
                    raw_result = model.transcribe(audio_path)
                    measured = round(time.perf_counter() - t0, 4)
                    inference_latency_s = measured
                    # Latency-aware adapters return (text, LatencyMetrics); legacy /
                    # batch adapters return a bare string. Normalise both, falling
                    # back to the wall-clock measurement for complete_s when the
                    # adapter does not report its own.
                    prediction, latency = unpack_transcription(raw_result, fallback_complete_s=measured)
                    if latency is not None:
                        complete_s = round(latency.complete_s, 4) if latency.complete_s is not None else measured
                        ttft_s = round(latency.ttft_s, 4) if latency.ttft_s is not None else None

                prediction = (prediction or "").strip()
                if not prediction:
                    failed += 1
                    # Prefer the adapter's recorded cause (401, invalid key,
                    # network failure, …) over the downstream symptom, so the
                    # CSV error column and scores JSON the CLI points at are
                    # self-explaining (issue #170).
                    cause = getattr(model, "last_transcribe_error", None)
                    results.append(
                        SingleSpeakerEvaluator._result_row(
                            audio_path,
                            ground_truth,
                            aq,
                            inference_latency_s=inference_latency_s,
                            ttft_s=ttft_s,
                            complete_s=complete_s,
                            error=f"Transcription failed: {cause}" if cause else "Empty prediction",
                        )
                    )
                    continue

                cer, wer, gt_norm, pred_norm = UtteranceEvaluator.score_single_variant(
                    ground_truth,
                    prediction,
                    language=language,
                )
                if cer is None or wer is None:
                    # One project-wide convention for missing metrics (issue
                    # #107): an uncomputable CER/WER is excluded from every
                    # aggregate and the row is surfaced as failed — never
                    # scored as best case (this used to substitute 0.0,
                    # deflating the run averages). Transcription itself
                    # succeeded, so the prediction is preserved on the row.
                    # The normalized pair is preserved too: an empty
                    # normalized_reference IS the diagnosis here.
                    failed += 1
                    scoring_error = (
                        "CER/WER uncomputable (reference normalized to empty, or jiwer "
                        "unavailable) — row excluded from aggregates; transcription "
                        "succeeded, see the prediction column"
                    )
                    logger.warning(f"[{idx}/{len(data)}] {scoring_error}")
                    results.append(
                        SingleSpeakerEvaluator._result_row(
                            audio_path,
                            ground_truth,
                            aq,
                            prediction=prediction,
                            normalized_reference=gt_norm,
                            normalized_hypothesis=pred_norm,
                            inference_latency_s=inference_latency_s,
                            ttft_s=ttft_s,
                            complete_s=complete_s,
                            error=scoring_error,
                        )
                    )
                    continue

                total_wer += wer
                total_cer += cer
                successful += 1

                if compute_sem and gt_norm and gt_norm.strip():
                    _sem_pairs.append((len(results), gt_norm, pred_norm))

                results.append(
                    SingleSpeakerEvaluator._result_row(
                        audio_path,
                        ground_truth,
                        aq,
                        prediction=prediction,
                        normalized_reference=gt_norm,
                        normalized_hypothesis=pred_norm,
                        wer=wer,
                        cer=cer,
                        significant_wer=is_significant_wer(wer, significant_wer_threshold),
                        inference_latency_s=inference_latency_s,
                        ttft_s=ttft_s,
                        complete_s=complete_s,
                    )
                )
            except Exception as e:
                failed += 1
                logger.error(f"[{idx}/{len(data)}] Error: {str(e)}")
                results.append(
                    SingleSpeakerEvaluator._result_row(
                        audio_path,
                        ground_truth,
                        aq,
                        inference_latency_s=inference_latency_s,
                        ttft_s=ttft_s,
                        complete_s=complete_s,
                        error=str(e),
                    )
                )

        semantics_warning = None
        if compute_sem:
            semantics_warning = SingleSpeakerEvaluator._apply_batch_semantics(results, _sem_pairs)

        elapsed = time.time() - start
        # None (null in scores.json), not 0.0: a run with zero successful
        # samples must not report the best possible error rates (issue #102).
        avg_wer = total_wer / successful if successful > 0 else None
        avg_cer = total_cer / successful if successful > 0 else None

        avg_sem = None
        avg_poseidon = None
        if compute_sem:
            sem_values = [r["semantic_similarity"] for r in results if r["semantic_similarity"] is not None]
            poseidon_values = [r["poseidon_score"] for r in results if r["poseidon_score"] is not None]
            if sem_values:
                avg_sem = sum(sem_values) / len(sem_values)
            if poseidon_values:
                avg_poseidon = sum(poseidon_values) / len(poseidon_values)

        latency_values = [r["inference_latency_s"] for r in results if r.get("inference_latency_s") is not None]
        latency_stats = compute_latency_summary(latency_values)
        avg_latency = latency_stats["avg_latency_s"]
        median_latency = latency_stats["median_latency_s"]
        p95_latency = latency_stats["p95_latency_s"]

        # Protocol-aware split: complete-latency percentiles (always populated)
        # plus TTFT percentiles (None unless a streaming adapter reported TTFT).
        complete_values = [r.get("complete_s") for r in results]
        ttft_values = [r.get("ttft_s") for r in results]
        protocol_stats = compute_protocol_latency_summary(complete_values, ttft_values)

        logger.info(f"Evaluation completed in {elapsed:.2f}s")
        logger.info(f"Successful: {successful}, Failed: {failed}")
        if avg_wer is None:
            logger.warning(
                "No samples were successfully evaluated — WER/CER aggregates are undefined (null). "
                "See the per-utterance error column for the failure reasons."
            )
        elif avg_poseidon is not None:
            logger.info(
                f"Average WER: {avg_wer:.4f}, CER: {avg_cer:.4f}, Sem: {avg_sem:.4f}, POSEIDON: {avg_poseidon:.4f}"
            )
        else:
            logger.info(f"Average WER: {avg_wer:.4f}, Average CER: {avg_cer:.4f}")
        if avg_latency is not None:
            logger.info(f"Latency — avg: {avg_latency:.4f}s, median: {median_latency:.4f}s, p95: {p95_latency:.4f}s")
        if protocol_stats["ttft_p50"] is not None:
            logger.info(
                f"TTFT — p50: {protocol_stats['ttft_p50']:.4f}s, p95: {protocol_stats['ttft_p95']:.4f}s "
                f"(complete p50: {protocol_stats['complete_p50']:.4f}s, p95: {protocol_stats['complete_p95']:.4f}s)"
            )

        return {
            "model_name": model_name,
            "results": results,
            # Why semantic_similarity/poseidon_score are null when they are —
            # recorded into the scores.json warnings array (issue #191).
            "semantics_warning": semantics_warning,
            "summary": {
                "total_samples": len(data),
                "successful": successful,
                "failed": failed,
                "avg_wer": avg_wer,
                "avg_cer": avg_cer,
                "elapsed_time": elapsed,
                "avg_latency_s": avg_latency,
                "median_latency_s": median_latency,
                "p95_latency_s": p95_latency,
                "complete_p50_s": protocol_stats["complete_p50"],
                "complete_p95_s": protocol_stats["complete_p95"],
                "ttft_p50_s": protocol_stats["ttft_p50"],
                "ttft_p95_s": protocol_stats["ttft_p95"],
            },
        }

    @classmethod
    def run_evaluation(
        cls,
        tsv_path: Optional[str] = None,
        output_dir: Optional[str] = None,
        models: Optional[List[str]] = None,
        max_samples: int = 0,
        compute_sem: bool = False,
        custom_hf_model: Optional[str] = None,
        language: str = "bn",
        allow_absolute_audio_paths: bool = True,
        submission: Optional["SubmissionConfig"] = None,
        write_scores: bool = True,
        significant_wer_threshold: float = DEFAULT_SIGNIFICANT_WER_THRESHOLD,
        streaming: bool = False,
    ) -> Dict:
        """Run evaluation for specified models.

        ``significant_wer_threshold`` is propagated into each row's
        ``significant_wer`` flag and into the run-level
        ``significant_wer_rate`` aggregate written to ``scores.json``.

        ``streaming`` requests the streaming protocol from adapters that have
        one (``assemblyai_api``, which then records ``ttft_s``); models
        without a streaming mode log a warning and run batch as usual.
        """
        load_env()

        _custom_hf_model = custom_hf_model

        if tsv_path is None:
            tsv_path = cls.DEFAULT_TSV
        if output_dir is None:
            output_dir = cls.DEFAULT_OUTPUT_DIR
        if models is None:
            models = cls.AVAILABLE_MODELS

        data = cls.load_data(tsv_path, allow_absolute_audio_paths=allow_absolute_audio_paths)
        logger.info(f"Loaded {len(data)} samples from {tsv_path}")

        # A supported --language applied to data in a different language used
        # to run silently and produce a healthy-looking scorecard (issue
        # #148). Warn once per run and record it in every scores.json so the
        # artifact itself is distinguishable from a correct run's.
        from psdn_sonar.language.script_check import script_mismatch_warning

        run_warnings: List[str] = []
        mismatch = script_mismatch_warning((row.get("ground_truth", "") for row in data), language)
        if mismatch:
            logger.warning(mismatch)
            run_warnings.append(mismatch)

        # A language scored with the generic fallback normalizer carries the
        # same claim as the script-mismatch warning above — these WER/CER are
        # not comparable to a dedicated-normalizer run's — but was only ever
        # printed at CLI load time, never recorded in the artifact (issue
        # #184). Record it in every scores.json; the shared helper keeps the
        # wording identical to the CLI's log line.
        from psdn_sonar.utils.text_processing import fallback_normalizer_warning

        fallback = fallback_normalizer_warning(language)
        if fallback:
            logger.warning(fallback)
            run_warnings.append(fallback)

        # A core install without [ml] cannot compute the headline metrics.
        # Say so up front — before any transcription time or API spend — in
        # the same one-actionable-line style as the other dependency paths
        # (#169/#177), and record it in every scores.json so a reader of the
        # artifact alone can tell the metrics are absent and why (issue #191).
        if compute_sem and _semantics_dependency_missing():
            logger.warning(_SEMANTICS_MISSING_EXTRA_WARNING)
            run_warnings.append(_SEMANTICS_MISSING_EXTRA_WARNING)

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Pre-warm DNSMOS/UTMOS/SQUIM in background while the first ASR model loads.
        # Quality models are module-level singletons so this is a one-time cost.
        import threading

        def _prewarm_quality_models():
            from psdn_sonar.quality_models import _get_dnsmos, _get_squim, _get_utmos

            _get_dnsmos()
            _get_utmos()
            _get_squim()

        _prewarm_thread = threading.Thread(target=_prewarm_quality_models, daemon=True)
        _prewarm_thread.start()

        all_results = {}
        skipped_models = []
        for model_name in models:
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Evaluating model: {model_name}")
            logger.info(f"{'=' * 60}")

            try:
                model = _model_factory(
                    model_name,
                    custom_hf_model=_custom_hf_model,
                    language=language,
                    streaming=streaming or None,
                )
            except Exception as e:
                # One unconstructible model must not end the multi-model run
                # (issue #108: the bn defaults died at khushids_bengali's
                # missing peft after ~3 GB of downloads, and models already
                # evaluated in the run lost their output). Log the actionable
                # reason (adapters name their missing dependency/extra) and
                # continue; if every model skips, the guard below still fails
                # the run loudly.
                logger.error("Skipping model %s — could not be constructed: %s", model_name, e)
                skipped_models.append(model_name)
                continue
            if model is None:
                logger.error(
                    "Model %s not found in the registry; skipping. Registered ids: %s",
                    model_name,
                    ", ".join(cls.AVAILABLE_MODELS),
                )
                skipped_models.append(model_name)
                continue

            lineage = _run_lineage(model, language)
            if lineage.hf_model_id or lineage.hf_revision:
                logger.info(
                    "Model lineage: %s @ %s",
                    lineage.hf_model_id or "<unknown id>",
                    lineage.hf_revision or "<unknown revision>",
                )

            _prewarm_thread.join()  # ensure quality models are loaded before AQ computation starts
            result = cls.evaluate_one(
                model,
                data,
                model_name,
                max_samples,
                compute_sem,
                language,
                significant_wer_threshold=significant_wer_threshold,
            )

            # Mirror of the reference-side script check above, on this
            # model's own transcripts. A service that returns a different
            # writing system (observed: Devanagari for --language bn) floors
            # every row at WER 1.0 with nothing in the artifact to separate
            # it from a model that transcribed badly (issue #207).
            from psdn_sonar.language.script_check import hypothesis_script_mismatch_warning

            hyp_mismatch = hypothesis_script_mismatch_warning(
                (row.get("prediction") or "" for row in result["results"]),
                language,
                model_name,
            )
            if hyp_mismatch:
                logger.warning(hyp_mismatch)
            result["hypothesis_script_warning"] = hyp_mismatch

            all_results[model_name] = result

            output_file = Path(output_dir) / f"asr_detailed_{model_name}.csv"
            with open(output_file, "w", encoding="utf-8", newline="") as f:
                # Always write the header so consumers reading the CSV (e.g.
                # pandas.read_csv) never hit a zero-byte file on empty runs.
                fieldnames = result["results"][0].keys() if result["results"] else cls._csv_fieldnames()
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(result["results"])

            logger.info(f"Results saved to {output_file}")

            if write_scores:
                from psdn_sonar.benchmark.scores import build_run_scores, scores_json_path, write_scores_json

                run_submission = submission or _default_submission_for_model(
                    model,
                    model_name,
                    language=language,
                )

                # Warnings raised during this model's own evaluation (a
                # semantics failure in the batch encode, or transcripts that
                # came back in the wrong writing system — issue #207) are
                # recorded in this model's artifact; anything already in
                # run_warnings from the preflight is not duplicated.
                per_model_warnings = [
                    w
                    for w in (result.get("semantics_warning"), result.get("hypothesis_script_warning"))
                    if w and w not in run_warnings
                ]
                model_warnings = run_warnings + per_model_warnings if per_model_warnings else run_warnings

                scores_path = scores_json_path(output_dir, model_name)
                artifact = build_run_scores(
                    run_submission,
                    result,
                    compute_sem=compute_sem,
                    significant_wer_threshold=significant_wer_threshold,
                    lineage=lineage,
                    run_warnings=model_warnings,
                )
                write_scores_json(scores_path, artifact)
                logger.info(f"Scores saved to {scores_path}")

        if not all_results:
            # Do not blanket-recommend "--models / --hf-model" here: when the
            # reason is a missing install extra or API key, every model that
            # needs it fails the same way (issue #169).
            raise ValueError(
                f"None of the requested models could be constructed: {', '.join(skipped_models)}. "
                "Fix the per-model reasons in the log lines above first — a missing install "
                "extra or API key affects every model that needs it, so trying another id "
                "fails the same way. For a mistyped id, pass a registered one via --models "
                f"(registered ids: {', '.join(cls.AVAILABLE_MODELS)}) or a HuggingFace repo id "
                "via --hf-model."
            )

        total_successful = sum(r["summary"]["successful"] for r in all_results.values())
        if total_successful == 0:
            raise NoSamplesEvaluatedError(
                f"Evaluation finished with 0 successful samples across {len(all_results)} model(s). "
                f"Per-row failure reasons are in {output_dir}/asr_detailed_<model>.csv "
                "(error column) and scores_<model>.json; WER/CER aggregates are null."
            )

        return all_results
