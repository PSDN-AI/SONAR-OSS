"""Pydantic models for the v1 benchmark submission contract."""

from __future__ import annotations

import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator

from psdn_sonar import __version__

KNOWN_INFERENCE_PARAM_KEYS = frozenset(
    {
        "beam_size",
        "temperature",
        "language_code",
        "best_of",
        "patience",
        "length_penalty",
    }
)

Protocol = Literal["batch", "streaming"]


def _resolve_git_sha() -> str:
    """SHA of the checkout the *package* runs from, never the caller's cwd.

    ``SONAR_GIT_SHA`` overrides (CI sets it when it evaluates a wheel built
    from a known commit). Otherwise the SHA is resolved with ``git -C`` on
    this file's own directory, and recorded only if this very file is
    tracked by that repository — a venv nested inside an unrelated repo
    would otherwise attribute the run to that repo's HEAD (issue #110: the
    old code ran ``git rev-parse HEAD`` in the caller's working directory,
    so a run started from any other repo recorded that repo's commit).
    Wheel/pip installs have no checkout and record ``"unknown"``;
    ``package_version`` identifies the code there.
    """
    env_sha = os.getenv("SONAR_GIT_SHA", "").strip()
    if env_sha:
        return env_sha
    own_file = Path(__file__).resolve()
    git_c = ["git", "-C", str(own_file.parent)]
    try:
        subprocess.check_output(
            [*git_c, "ls-files", "--error-unmatch", str(own_file)],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return subprocess.check_output(
            [*git_c, "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _resolve_poseidon_weights() -> dict[str, float]:
    """The POSEIDON weights in effect, including any ``POSEIDON_*_WEIGHT`` env overrides."""
    from psdn_sonar.config import config

    return {
        "wer": config.wer_weight,
        "cer": config.cer_weight,
        "semantic": config.semantic_weight,
    }


def _resolve_similarity_model() -> str:
    """The semantic-similarity model id in effect, including any ``SIMILARITY_MODEL`` override."""
    from psdn_sonar.config import config

    return config.similarity_model


def _resolve_device() -> Optional[str]:
    """Best-effort compute device ("cuda"/"mps"/"cpu"), or None without torch."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
        return "cpu"
    except Exception:
        return None


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


class SubmissionConfig(BaseModel):
    """Machine-readable metadata describing the conditions of one ASR eval run."""

    provider: str = Field(
        ...,
        min_length=1,
        description=(
            "Id of the service that actually served inference (e.g. openai, "
            "elevenlabs, assemblyai); 'local' for in-process inference."
        ),
    )
    model_snapshot: str = Field(
        ...,
        min_length=1,
        description="Pinned model id or snapshot string (e.g. whisper-1@2024-06-01).",
    )
    region: Optional[str] = Field(
        default=None,
        description=(
            "Inference region (e.g. us-east-1, ap-south-1) when known. Null "
            "otherwise: hosted providers do not disclose one and local runs "
            "have none, so a value only appears when the caller supplies it "
            "(issue #184: this used to default to the meaningless 'local')."
        ),
    )
    protocol: Protocol
    inference_params: dict[str, Any] = Field(default_factory=dict)
    sample_rate_hz: Optional[int] = Field(default=None, ge=1)
    seed: int
    judge_model: Optional[str] = None
    prompt_version: Optional[str] = Field(
        default=None,
        description="8-char PROMPT_VERSION hash when LLM-judge metrics are included.",
    )
    git_sha: str
    package_version: str
    timestamp_utc: str = Field(..., description="ISO-8601 UTC timestamp for run start (Z suffix).")
    # Score-changing inputs and environment (issue #110). Optional with None
    # defaults so pre-existing artifacts and direct constructions still
    # validate; from_env() always fills them.
    poseidon_weights: Optional[dict[str, float]] = Field(
        default=None,
        description="POSEIDON component weights in effect (wer/cer/semantic), including POSEIDON_*_WEIGHT env overrides.",
    )
    similarity_model: Optional[str] = Field(
        default=None,
        description="Semantic-similarity model id in effect, including the SIMILARITY_MODEL env override.",
    )
    os_platform: Optional[str] = Field(default=None, description="OS identification (platform.platform()).")
    python_version: Optional[str] = Field(default=None, description="Python version the run executed under.")
    device: Optional[str] = Field(
        default=None,
        description="Compute device used by local models (cuda/mps/cpu); null when torch is unavailable.",
    )

    @field_validator("inference_params")
    @classmethod
    def _validate_inference_params(cls, value: dict[str, Any]) -> dict[str, Any]:
        unknown = set(value) - KNOWN_INFERENCE_PARAM_KEYS
        if unknown:
            raise ValueError(
                f"Unknown inference_params keys: {sorted(unknown)}; allowed: {sorted(KNOWN_INFERENCE_PARAM_KEYS)}"
            )
        return value

    @classmethod
    def from_env(
        cls,
        *,
        provider: str,
        model_snapshot: str,
        region: Optional[str] = None,
        protocol: Protocol = "batch",
        inference_params: Optional[dict[str, Any]] = None,
        sample_rate_hz: Optional[int] = None,
        seed: Optional[int] = None,
        judge_model: Optional[str] = None,
        prompt_version: Optional[str] = None,
    ) -> SubmissionConfig:
        """Build a config with git SHA, package version, timestamp, seed, and
        the score-changing inputs (POSEIDON weights, similarity model,
        OS/Python/device) resolved automatically."""
        from psdn_sonar.config_loader import get_run_seed

        return cls(
            provider=provider,
            model_snapshot=model_snapshot,
            region=region,
            protocol=protocol,
            inference_params=inference_params or {},
            sample_rate_hz=sample_rate_hz,
            seed=get_run_seed() if seed is None else seed,
            judge_model=judge_model,
            prompt_version=prompt_version,
            git_sha=_resolve_git_sha(),
            package_version=__version__,
            timestamp_utc=_utc_now_iso(),
            poseidon_weights=_resolve_poseidon_weights(),
            similarity_model=_resolve_similarity_model(),
            os_platform=platform.platform(),
            python_version=platform.python_version(),
            device=_resolve_device(),
        )
