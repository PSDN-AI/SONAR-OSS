"""Loanword normalizer: replaces Latin-script (English) tokens with native-script
transliterations using a precomputed JSON cache.

Applying the same replacement to both reference and hypothesis ensures that a
correctly recognized loanword is never counted as a WER error just because one
side spells it in Latin script and the other in native script.

The cache maps lowercase English words to their Bengali-script transliterations
(e.g. "customer" -> "কাস্টমার", "insurance" -> "ইনস্যুরেন্স").

Flow:
  1. Load cache from config/language/<lang>/loanword_cache.json
  2. For each text: detect Latin tokens, replace with cached transliterations
  3. Mixed-script tokens (e.g. "helpline-এ") are handled by replacing Latin runs
"""

from __future__ import annotations

import json
import logging
import re
from importlib.resources import files
from os import PathLike
from pathlib import Path
from typing import cast

try:
    from importlib.abc import Traversable
except ImportError:
    from importlib.resources.abc import Traversable

logger = logging.getLogger(__name__)

_LATIN_RE = re.compile(r"^[a-zA-Z0-9\-']+$")
_LATIN_RUN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9]*")


def is_latin(token: str) -> bool:
    """Check if a token is Latin-script (English word, number, or hyphenated)."""
    return bool(_LATIN_RE.match(token))


def extract_latin_tokens(texts: list[str]) -> set[str]:
    """Extract all unique Latin-script tokens from a list of texts.

    Handles both standalone Latin tokens and Latin portions of hyphenated
    mixed-script tokens. Pure-digit tokens are excluded.
    """
    tokens: set[str] = set()
    for text in texts:
        for word in text.split():
            if is_latin(word):
                lower = word.lower()
                if not lower.isdigit():
                    tokens.add(lower)
            else:
                for match in _LATIN_RUN_RE.finditer(word):
                    tokens.add(match.group().lower())
    return tokens


def load_cache(cache_path: PathLike[str] | str | Traversable) -> dict[str, str]:
    """Load the loanword cache from disk."""
    if isinstance(cache_path, (str, PathLike)):
        fs_cache_path = Path(cast(str | PathLike[str], cache_path))
        if fs_cache_path.exists():
            with open(fs_cache_path, encoding="utf-8") as f:
                return json.load(f)
        return {}

    if cache_path.is_file():
        with cache_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def get_cache_path(language_code: str = "bn") -> Path | Traversable:
    """Get the cache file path for a given language.

    Prefer the repo-root cache during source checkout, then fall back to the
    packaged resource included in wheels and sdists.
    """
    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    repo_cache = repo_root / "config" / "language" / language_code / "loanword_cache.json"
    if repo_cache.exists():
        return repo_cache
    return (
        files("psdn_sonar")
        .joinpath("resources")
        .joinpath("language")
        .joinpath(language_code)
        .joinpath("loanword_cache.json")
    )


def replace_latin_tokens(text: str, cache: dict[str, str]) -> tuple[str, int, int]:
    """Replace Latin-script tokens in text with cached native-script transliterations.

    Handles standalone Latin tokens ("insurance" -> "ইনস্যুরেন্স") and Latin
    portions of hyphenated mixed-script tokens ("helpline-এ" -> "হেল্পলাইনএ").
    Pure-digit tokens are skipped (handled by number word normalization).

    Returns:
        Tuple of (transliterated text, tokens_replaced count, tokens_uncached count).
    """
    tokens = text.split()
    result = []
    replaced = 0
    uncached = 0

    for token in tokens:
        if token.isdigit():
            result.append(token)
        elif is_latin(token):
            key = token.lower()
            if key in cache:
                result.append(cache[key])
                replaced += 1
            else:
                result.append(token)
                uncached += 1
        elif _LATIN_RUN_RE.search(token):
            token_replaced = False

            def _replace_run(m: re.Match) -> str:
                nonlocal replaced, uncached, token_replaced
                key = m.group().lower()
                if key in cache:
                    replaced += 1
                    token_replaced = True
                    return cache[key]
                else:
                    uncached += 1
                    return m.group()

            new_token = _LATIN_RUN_RE.sub(_replace_run, token)
            if token_replaced:
                new_token = new_token.replace("-", "")
            result.append(new_token)
        else:
            result.append(token)

    return " ".join(result), replaced, uncached
