"""Loanword normalization — replaces Latin-script tokens with native-script
transliterations using a precomputed JSON cache.

The cache contents are part of the scoring contract: reference and hypothesis
text must be normalized with the same cache for WER numbers to be comparable.
"""

from .normalizer import (
    extract_latin_tokens,
    get_cache_path,
    is_latin,
    load_cache,
    replace_latin_tokens,
)
from .validator import validate_cache, validate_cache_file

__all__ = [
    "extract_latin_tokens",
    "get_cache_path",
    "is_latin",
    "load_cache",
    "replace_latin_tokens",
    "validate_cache",
    "validate_cache_file",
]
