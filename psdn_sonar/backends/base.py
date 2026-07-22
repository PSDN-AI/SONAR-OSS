"""Abstract ASR backend contract.

Backends are the config-driven counterpart to the per-model adapters in
:mod:`psdn_sonar.models`: instead of naming a specific checkpoint, a run
config selects a backend (``conf/backend/*.yaml``) and the backend resolves
the model for the run's language. Implementations register themselves with
``psdn_sonar.registry.register_asr`` and are resolved by name via
``get_asr_backend``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class ASRBackend(ABC):
    """Lifecycle contract for a config-driven ASR backend.

    ``setup`` receives the full run config and must leave the backend ready
    to transcribe; ``teardown`` releases whatever ``setup``/``transcribe``
    acquired (model handles, sessions) so a backend can be reused across
    runs.
    """

    @abstractmethod
    def setup(self, config: Any) -> None:
        pass

    @abstractmethod
    def transcribe(self, audio_path: Path, language: str) -> str:
        pass

    @abstractmethod
    def teardown(self) -> None:
        pass

    def supports_language(self, language: str) -> bool:
        return True
