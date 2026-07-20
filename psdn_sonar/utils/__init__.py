"""Utility subpackage public surface.

We deliberately do NOT eagerly re-export ``llm_metrics`` here. The
LLM-judged metrics live in :mod:`psdn_sonar.utils.llm_metrics` and
should be imported from that submodule directly:

    from psdn_sonar.utils.llm_metrics import evaluate_sample, get_client

Reasons for keeping them off the package-level ``__all__``:

  - The judge implementation depends on ``google-genai``, which is in
    the optional ``[apis]`` extra. Eagerly re-exporting makes the
    base import path silently brittle for users who haven't installed
    that extra (since ``llm_metrics`` itself only imports ``google.genai``
    inside functions, this works today but tightly couples the package's
    public API to an optional dependency that may grow heavier later).
  - Keeping the surface small means we can iterate on the LLM-judged
    metrics module (signatures, return shapes, retry semantics) without
    being committed to a stable public contract from the package root.
"""

from .plotting import ASRResultPlotter
from .scorer import PoseidonScorer, ScoreResult

__all__ = [
    "PoseidonScorer",
    "ScoreResult",
    "ASRResultPlotter",
]
