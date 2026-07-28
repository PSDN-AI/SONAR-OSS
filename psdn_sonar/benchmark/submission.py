"""Pydantic models for the v1 benchmark submission contract."""

from __future__ import annotations

import os
import subprocess
from datetime import datetime, timezone
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
    env_sha = os.getenv("SONAR_GIT_SHA", "").strip()
    if env_sha:
        return env_sha
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


class SubmissionConfig(BaseModel):
    """Machine-readable metadata describing the conditions of one ASR eval run."""

    provider: str = Field(..., min_length=1, description="Provider id (e.g. openai, assemblyai, huggingface).")
    model_snapshot: str = Field(
        ...,
        min_length=1,
        description="Pinned model id or snapshot string (e.g. whisper-1@2024-06-01).",
    )
    region: str = Field(..., min_length=1, description="Inference region (e.g. us-east-1, ap-south-1).")
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
        region: str,
        protocol: Protocol = "batch",
        inference_params: Optional[dict[str, Any]] = None,
        sample_rate_hz: Optional[int] = None,
        seed: Optional[int] = None,
        judge_model: Optional[str] = None,
        prompt_version: Optional[str] = None,
    ) -> SubmissionConfig:
        """Build a config with git SHA, package version, timestamp, and seed resolved automatically."""
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
        )
