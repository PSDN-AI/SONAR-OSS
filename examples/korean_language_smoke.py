"""Korean language processor and backend smoke check.

Demonstrates the config-driven registry: loads the Korean profile, runs
normalization/tokenization samples, and configures the HuggingFace ASR
backend (model download permitting).
"""

import logging

import psdn_sonar.backends  # noqa: F401 — triggers @register_asr decorators
import psdn_sonar.language  # noqa: F401 — triggers @register_language decorators
from psdn_sonar.config_loader import load_config
from psdn_sonar.registry import get_asr_backend, get_language_processor

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def check_korean_processor():
    logger.info("Korean language processor")
    config = load_config(language="ko", backend="huggingface")

    logger.info("Language: %s (%s)", config.language.name, config.language.code)
    logger.info("Tokenizer: %s", config.language.tokenizer)
    logger.info("Default model: %s", config.backend.model.default_ko)

    processor_cls = get_language_processor("ko")
    processor = processor_cls(config)

    test_cases = [
        "안녕하세요",
        "한국어 테스트입니다",
        "숫자 123을 읽어보세요",
        "서울은 대한민국의 수도입니다",
    ]

    for text in test_cases:
        normalized = processor.normalize(text)
        tokens = processor.tokenize(normalized)
        logger.info("Original:   %s", text)
        logger.info("Normalized: %s", normalized)
        logger.info("Tokens (%d): %s", len(tokens), tokens[:5])


def check_korean_backend():
    logger.info("Korean ASR backend configuration")
    config = load_config(language="ko", backend="huggingface")

    backend_cls = get_asr_backend("huggingface")
    backend = backend_cls()

    try:
        backend.setup(config)
        logger.info("Backend configured: model=%s device=%s", backend.model_name, backend.device)
        logger.info("Supports Korean: %s", backend.supports_language("ko"))
    except Exception as e:
        logger.warning("Backend setup failed (expected if model not downloaded): %s", e)


if __name__ == "__main__":
    check_korean_processor()
    check_korean_backend()
