from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..config import config


@dataclass
class ScoreResult:
    """Result container for POSEIDON scoring of a single (reference, hypothesis) pair."""

    wer: float
    cer: float
    similarity: float
    poseidon_score: float


class PoseidonScorer:
    """Configurable scorer that computes WER, CER, semantic similarity, and the composite POSEIDON score."""

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

    def calculate_wer(self, reference: str, hypothesis: str) -> float:
        if not reference or not hypothesis:
            return 1.0

        try:
            from jiwer import wer

            return min(wer(reference, hypothesis), 1.0)
        except Exception:
            return 1.0

    def calculate_cer(self, reference: str, hypothesis: str) -> float:
        if not reference or not hypothesis:
            return 1.0

        try:
            from jiwer import cer

            return min(cer(reference, hypothesis), 1.0)
        except Exception:
            return 1.0

    def calculate_similarity(self, text1: str, text2: str) -> float:
        if not text1 or not text2:
            return 0.0

        try:
            from sentence_transformers import util

            e1, e2 = self.model.encode([text1, text2], convert_to_tensor=False, show_progress_bar=False)
            return float(util.cos_sim(e1[None], e2[None])[0][0])
        except Exception:
            return 0.0

    def calculate_poseidon_score(self, wer: float, cer: float, similarity: float) -> float:
        wer_capped = min(max(wer, 0.0), 1.0)
        cer_capped = min(max(cer, 0.0), 1.0)
        similarity_capped = min(max(similarity, 0.0), 1.0)

        score = (
            self.wer_weight * (1 - wer_capped)
            + self.cer_weight * (1 - cer_capped)
            + self.semantic_weight * similarity_capped
        )

        return min(max(score, 0.0), 1.0)

    def score(self, reference: str, hypothesis: str) -> ScoreResult:
        wer = self.calculate_wer(reference, hypothesis)
        cer = self.calculate_cer(reference, hypothesis)
        similarity = self.calculate_similarity(reference, hypothesis)
        poseidon_score = self.calculate_poseidon_score(wer, cer, similarity)

        return ScoreResult(
            wer=round(wer, 4),
            cer=round(cer, 4),
            similarity=round(similarity, 4),
            poseidon_score=round(poseidon_score, 4),
        )
