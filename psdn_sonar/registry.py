import logging
from typing import Callable, Dict, Type

logger = logging.getLogger(__name__)

_ASR_BACKENDS: Dict[str, Type] = {}
_LANGUAGE_PROCESSORS: Dict[str, Type] = {}


def register_asr(name: str) -> Callable:
    def decorator(cls: Type) -> Type:
        _ASR_BACKENDS[name] = cls
        logger.debug(f"Registered ASR backend: {name}")
        return cls

    return decorator


def register_language(code: str) -> Callable:
    def decorator(cls: Type) -> Type:
        _LANGUAGE_PROCESSORS[code] = cls
        logger.debug(f"Registered language processor: {code}")
        return cls

    return decorator


def get_asr_backend(name: str) -> Type:
    if name not in _ASR_BACKENDS:
        available = list(_ASR_BACKENDS.keys())
        raise ValueError(f"Unknown ASR backend: {name}. Available: {available}")
    return _ASR_BACKENDS[name]


def get_language_processor(code: str) -> Type:
    if code not in _LANGUAGE_PROCESSORS:
        available = list(_LANGUAGE_PROCESSORS.keys())
        raise ValueError(f"Unknown language: {code}. Available: {available}")
    return _LANGUAGE_PROCESSORS[code]


def list_asr_backends() -> list:
    return list(_ASR_BACKENDS.keys())


def list_language_processors() -> list:
    return list(_LANGUAGE_PROCESSORS.keys())
