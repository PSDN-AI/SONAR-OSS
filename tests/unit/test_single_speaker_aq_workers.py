"""Tests for the SONAR_AQ_MAX_WORKERS env override on the single-speaker
evaluator's audio-quality ThreadPoolExecutor pool size.
"""

from unittest.mock import MagicMock

import pytest

from psdn_sonar.evaluators import single_speaker as single_speaker_module
from psdn_sonar.evaluators.single_speaker import (
    _DEFAULT_AQ_MAX_WORKERS,
    SingleSpeakerEvaluator,
    _resolve_aq_workers,
)


def test_default_when_env_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SONAR_AQ_MAX_WORKERS", raising=False)
    assert _resolve_aq_workers() == _DEFAULT_AQ_MAX_WORKERS


def test_env_override_lowers_workers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SONAR_AQ_MAX_WORKERS", "1")
    assert _resolve_aq_workers() == 1


def test_env_override_raises_workers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SONAR_AQ_MAX_WORKERS", "8")
    assert _resolve_aq_workers() == 8


def test_invalid_value_falls_back_to_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SONAR_AQ_MAX_WORKERS", "not-a-number")
    assert _resolve_aq_workers() == _DEFAULT_AQ_MAX_WORKERS


def test_zero_clamped_to_one(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SONAR_AQ_MAX_WORKERS", "0")
    assert _resolve_aq_workers() == 1


def test_negative_clamped_to_one(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SONAR_AQ_MAX_WORKERS", "-3")
    assert _resolve_aq_workers() == 1


def test_evaluate_one_passes_resolved_workers_to_thread_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Wiring test: confirms ``evaluate_one`` actually plumbs
    ``_resolve_aq_workers()`` into ``ThreadPoolExecutor(max_workers=...)``.

    Captures the constructor argument by monkeypatching the
    ``ThreadPoolExecutor`` symbol imported into the evaluator module. Uses an
    empty ``data`` list so the AQ pool runs over nothing, the downstream ASR
    loop is skipped, and we never touch the model — keeps the test fast and
    free of HuggingFace / GPU dependencies.
    """
    captured_workers: list[int] = []

    class CapturingExecutor:
        def __init__(self, max_workers: int) -> None:
            captured_workers.append(max_workers)

        def __enter__(self) -> "CapturingExecutor":
            return self

        def __exit__(self, *_: object) -> None:
            return None

        def map(self, _fn: object, _items: object) -> object:
            return iter([])

    monkeypatch.setattr(single_speaker_module, "ThreadPoolExecutor", CapturingExecutor)
    monkeypatch.setenv("SONAR_AQ_MAX_WORKERS", "1")

    SingleSpeakerEvaluator.evaluate_one(
        model=MagicMock(),
        data=[],
        model_name="wiring-probe",
    )

    # The resolved value was 1; cpu_count cap can only lower it (never raise),
    # so a captured 1 means the resolver value flowed through unchanged.
    assert captured_workers == [1]
