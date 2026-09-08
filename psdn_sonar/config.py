import logging
import os
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


def load_env() -> None:
    """Load .env file if python-dotenv is available. Call explicitly before using API keys.

    Variables already present in the process environment win over .env
    entries (python-dotenv's own default). ``override=True`` made the .env
    value take effect even when the same name was exported in the shell, so
    a per-run ``env ELEVENLABS_API_KEY=... psdn-sonar ...`` silently ran
    with the checkout's .env credential instead (issue #212).
    """
    try:
        from dotenv import load_dotenv as _load_dotenv
    except ImportError:
        return

    candidates = [
        os.path.join(os.getcwd(), ".env"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"),
    ]

    for path in candidates:
        if os.path.isfile(path):
            _load_dotenv(path, override=False)
            logger.debug("Loaded .env from: %s", path)
            return

    _load_dotenv()


def _safe_float(env_var: str, default: str) -> float:
    raw = os.getenv(env_var, default)
    try:
        return float(raw)
    except (ValueError, TypeError):
        logger.warning("Invalid value for %s=%r, using default %s", env_var, raw, default)
        return float(default)


def validate_poseidon_weights(wer_weight: float, cer_weight: float, semantic_weight: float) -> None:
    """Reject a malformed POSEIDON weight set: negative entries or a sum away from 1.0.

    The single validator behind every site that accepts weights (the env-var
    ``Config``, the per-call override of ``calculate_poseidon_score``, and
    ``PoseidonScorer``), so they accept and reject identically. The sign
    check exists because the sum check alone let a negative weight through
    silently — e.g. ``-0.5/0.75/0.75`` sums to 1.0 and inverts the
    composite, ranking a worse transcript higher (issue #238).
    """
    weights = {
        "wer_weight": wer_weight,
        "cer_weight": cer_weight,
        "semantic_weight": semantic_weight,
    }
    negative = {name: value for name, value in weights.items() if value < 0}
    if negative:
        named = ", ".join(f"{name}={value}" for name, value in negative.items())
        raise ValueError(
            f"POSEIDON weights must be non-negative, got {named}. A negative weight "
            "inverts the composite score, ranking worse transcripts higher."
        )
    total = wer_weight + cer_weight + semantic_weight
    if abs(total - 1.0) > 0.001:
        raise ValueError(f"POSEIDON weights must sum to 1.0, got {total}")


@dataclass
class Config:
    """POSEIDON scoring configuration.

    Field defaults are read from environment variables at construction time
    (``POSEIDON_*_WEIGHT``, ``SIMILARITY_MODEL``); pass explicit values to
    override. Weights must be non-negative and sum to 1.0.
    """

    wer_weight: float = field(default_factory=lambda: _safe_float("POSEIDON_WER_WEIGHT", "0.35"))
    cer_weight: float = field(default_factory=lambda: _safe_float("POSEIDON_CER_WEIGHT", "0.20"))
    semantic_weight: float = field(default_factory=lambda: _safe_float("POSEIDON_SEMANTIC_WEIGHT", "0.45"))
    similarity_model: str = field(
        default_factory=lambda: os.getenv(
            "SIMILARITY_MODEL",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        )
    )

    def __post_init__(self):
        self.validate()

    def validate(self) -> None:
        validate_poseidon_weights(self.wer_weight, self.cer_weight, self.semantic_weight)


_config: Config | None = None


def get_config() -> Config:
    """Return the singleton Config, creating it on first access (lazy)."""
    global _config
    if _config is None:
        _config = Config()
    return _config


class _ConfigProxy:
    """Transparent proxy that defers Config creation until first attribute access."""

    def __getattr__(self, name: str):
        return getattr(get_config(), name)


config = _ConfigProxy()
