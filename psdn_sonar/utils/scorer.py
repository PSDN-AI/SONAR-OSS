import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..config import config

logger = logging.getLogger(__name__)


@dataclass
class ScoreResult:
    """Result container for POSEIDON scoring of a single (reference, hypothesis) pair.

    Metrics are ``None`` when unmeasurable — empty reference, or the metric
    backend (jiwer / sentence-transformers) unavailable or failing. This is
    the project-wide missing-value convention (issue #107): a metric that
    cannot be computed is reported as missing, never substituted with a
    best- or worst-case value. ``poseidon_score`` is ``None`` whenever any
    component is ``None``.
    """

    wer: Optional[float]
    cer: Optional[float]
    similarity: Optional[float]
    poseidon_score: Optional[float]


class PoseidonScorer:
    """Configurable scorer that computes WER, CER, semantic similarity, and the composite POSEIDON score.

    Delegates CER/WER to :func:`psdn_sonar.utils.metrics.calculate_cer_wer`
    and the composite to
    :func:`psdn_sonar.utils.metrics.calculate_poseidon_score`, so a
    (reference, hypothesis) pair scores identically here and in the
    evaluation pipelines. Unmeasurable metrics are ``None`` (never
    worst-case substitutes — issue #107; this class used to silently report
    WER/CER 1.0 and similarity 0.0 for pairs it could not score, inflating
    batch averages relative to the pipelines, which exclude missing
    values). Similarity is cosine clamped to ``[0, 1]``, the range every
    artifact reports.

    Note: an empty *hypothesis* against a non-empty reference IS measurable
    (WER/CER are genuinely 1.0 — every word wrong); only an empty
    *reference* or a backend failure makes a metric missing.
    """

    def __init__(
        self,
        wer_weight: Optional[float] = None,
        cer_weight: Optional[float] = None,
        semantic_weight: Optional[float] = None,
        model_name: Optional[str] = None,
    ):
        self.wer_weight = wer_weight if wer_weight is not None else config.wer_weight
        self.cer_weight = cer_weight if cer_weight is not None else config.cer_weight
        self.semantic_weight = semantic_weight if semantic_weight is not None else config.semantic_weight
        self.model_name = model_name or config.similarity_model
        self._model = None

        total_weight = self.wer_weight + self.cer_weight + self.semantic_weight
        if not np.isclose(total_weight, 1.0):
            raise ValueError(f"Weights must sum to 1.0, got {total_weight}")

    @property
    def model(self):
        if self._model is None:
            if self.model_name == config.similarity_model:
                from ..utils.metrics import _get_semantic_model

                self._model = _get_semantic_model()
            else:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self.model_name)
        return self._model

    def calculate_wer(self, reference: str, hypothesis: str) -> Optional[float]:
        """WER via the canonical helper; ``None`` when unmeasurable."""
        from ..utils.metrics import calculate_cer_wer

        return calculate_cer_wer(reference, hypothesis)[1]

    def calculate_cer(self, reference: str, hypothesis: str) -> Optional[float]:
        """CER via the canonical helper; ``None`` when unmeasurable."""
        from ..utils.metrics import calculate_cer_wer

        return calculate_cer_wer(reference, hypothesis)[0]

    def calculate_similarity(self, text1: str, text2: str) -> Optional[float]:
        """Cosine similarity clamped to ``[0, 1]``; ``None`` when unmeasurable.

        Kept local (rather than delegating to
        :func:`~psdn_sonar.utils.metrics.compute_semantic_similarity`) so a
        custom ``model_name`` is honored; the missing-value and clamping
        conventions are identical.
        """
        from ..utils.metrics import clamp_similarity

        if not text1 or not text1.strip():
            return None
        try:
            from sentence_transformers import util

            e1, e2 = self.model.encode([text1, text2 or ""], convert_to_tensor=False, show_progress_bar=False)
            return clamp_similarity(float(util.cos_sim(e1[None], e2[None])[0][0]))
        except Exception:
            logger.warning("Semantic similarity calculation failed", exc_info=True)
            return None

    def calculate_poseidon_score(self, wer: float, cer: float, similarity: float) -> float:
        """Composite score via the canonical helper, using this scorer's weights.

        Raises ``TypeError`` on ``None`` inputs (see
        :func:`psdn_sonar.utils.metrics.calculate_poseidon_score`); use
        :meth:`score`, which returns ``poseidon_score=None`` when any
        component is missing.
        """
        from ..utils.metrics import calculate_poseidon_score

        return calculate_poseidon_score(
            cer,
            wer,
            similarity,
            wer_weight=self.wer_weight,
            cer_weight=self.cer_weight,
            semantic_weight=self.semantic_weight,
        )

    def score(self, reference: str, hypothesis: str) -> ScoreResult:
        from ..utils.metrics import calculate_cer_wer

        cer, wer = calculate_cer_wer(reference, hypothesis)
        similarity = self.calculate_similarity(reference, hypothesis)
        poseidon_score = (
            self.calculate_poseidon_score(wer, cer, similarity)
            if (wer is not None and cer is not None and similarity is not None)
            else None
        )

        def _round(value: Optional[float]) -> Optional[float]:
            return round(value, 4) if value is not None else None

        return ScoreResult(
            wer=_round(wer),
            cer=_round(cer),
            similarity=_round(similarity),
            poseidon_score=_round(poseidon_score),
        )
