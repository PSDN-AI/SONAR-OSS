"""Validation tooling for loanword caches.

The hand-curated JSON files at ``config/language/<lang>/loanword_cache.json``
ship in the wheel and are loaded at WER/CER evaluation time. They are large
(Bengali ~48 KB, Hindi ~12 KB, Korean ~5 KB) and easy to break with a typo
or a casing slip — and a broken entry causes silent normalization drift,
not a crash.

This module exposes ``validate_cache(cache, language=None)`` which returns
a list of human-readable issue strings (empty list = clean cache). It is
intended for both:

  * the test suite (``tests/test_loanword_cache_integrity.py``) — hard
    assertion that every shipped cache passes validation
  * a future CLI / pre-commit hook — stops a bad cache reaching ``main``

Checks performed:

  1. Keys are non-empty strings
  2. Keys are pure lowercase ASCII (the loader does ``.lower()`` on lookup,
     so any uppercase/Unicode keys would never match)
  3. Keys are unique after lowercasing (case-insensitive collisions would
     make lookup non-deterministic)
  4. Values are non-empty strings
  5. Values contain at least one non-ASCII character (the cache exists to
     map a Latin/ASCII token to a native-script transliteration; a value
     that's still pure ASCII means the transliteration was forgotten)
  6. Values do not contain leading/trailing whitespace (would survive into
     the normalized output and inflate token-level metrics)
"""

from __future__ import annotations


def validate_cache(cache: dict[str, str], language: str | None = None) -> list[str]:
    """Return a list of issue strings describing problems in ``cache``.

    An empty return value means the cache is clean. ``language`` is purely
    informational and shows up in the issue strings to make multi-cache
    test failures readable.
    """
    issues: list[str] = []
    seen_lower: dict[str, str] = {}
    tag = f"[{language}] " if language else ""

    for key, value in cache.items():
        if not isinstance(key, str) or not key:
            issues.append(f"{tag}empty / non-string key: {key!r}")
            continue

        if not key.isascii():
            issues.append(f"{tag}key {key!r} is not pure ASCII (cache lookup lowercases ASCII only)")
        elif key != key.lower():
            issues.append(f"{tag}key {key!r} is not lowercase (will never be matched after lookup-time .lower())")

        lower = key.lower()
        if lower in seen_lower and seen_lower[lower] != key:
            issues.append(
                f"{tag}case-insensitive duplicate key: {key!r} collides with {seen_lower[lower]!r} "
                "(cache lookup is case-insensitive, so one of these is unreachable)"
            )
        seen_lower[lower] = key

        if not isinstance(value, str) or not value:
            issues.append(f"{tag}empty / non-string value for key {key!r}: {value!r}")
            continue

        if value != value.strip():
            issues.append(
                f"{tag}value for key {key!r} has leading/trailing whitespace: {value!r} "
                "(would survive normalization and inflate token-level metrics)"
            )

        if value.isascii():
            issues.append(
                f"{tag}value for key {key!r} is still pure ASCII: {value!r} "
                "(expected a non-ASCII transliteration into the target script)"
            )

    return issues


def validate_cache_file(path, language: str | None = None) -> list[str]:
    """Load ``path`` and run ``validate_cache`` on it.

    Returns a single-issue list if the file is missing or malformed JSON,
    otherwise the same shape as ``validate_cache``.
    """
    import json
    from pathlib import Path

    p = Path(path)
    if not p.is_file():
        return [f"cache file does not exist: {p}"]
    try:
        with p.open(encoding="utf-8") as f:
            cache = json.load(f)
    except json.JSONDecodeError as e:
        return [f"cache file {p} is not valid JSON: {e}"]
    if not isinstance(cache, dict):
        return [f"cache file {p} top-level value must be a JSON object, got {type(cache).__name__}"]
    return validate_cache(cache, language=language)
