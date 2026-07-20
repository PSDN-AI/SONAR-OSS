import logging
import os
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


def load_env() -> None:
    """Load .env file if python-dotenv is available. Call explicitly before using API keys."""
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
            _load_dotenv(path, override=True)
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


@dataclass
class Config:
    """POSEIDON scoring configuration.

    Field defaults are read from environment variables at construction time
    (``POSEIDON_*_WEIGHT``, ``SIMILARITY_MODEL``); pass explicit values to
    override. Weights must sum to 1.0.
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
        weight_sum = self.wer_weight + self.cer_weight + self.semantic_weight
        if abs(weight_sum - 1.0) > 0.001:
            raise ValueError(f"Poseidon weights must sum to 1.0, got {weight_sum}")


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
