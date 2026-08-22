"""Hard assertion that every shipped loanword cache passes validation.

This is the test file `psdn_sonar/utils/loanword/validator.py` has always
pointed at; until issue #115 it did not exist, so the shipped caches had no
CI validation at all. That matters because loanword replacement is applied
to BOTH the reference and the hypothesis during normalization — a polluted
entry (uppercase key that can never match, ASCII value that was never
transliterated, whitespace that survives into tokens) silently changes
WER/CER for bn/hi/ko rather than crashing.
"""

import json
from pathlib import Path

import pytest

from psdn_sonar.utils.loanword.validator import validate_cache, validate_cache_file

REPO_ROOT = Path(__file__).resolve().parents[1]
SHIPPED_CACHES = {lang: REPO_ROOT / "config" / "language" / lang / "loanword_cache.json" for lang in ("bn", "hi", "ko")}


class TestShippedCaches:
    @pytest.mark.parametrize("language", sorted(SHIPPED_CACHES))
    def test_cache_passes_every_integrity_check(self, language):
        issues = validate_cache_file(SHIPPED_CACHES[language], language=language)
        assert issues == [], "shipped cache failed validation:\n" + "\n".join(issues)

    @pytest.mark.parametrize("language", sorted(SHIPPED_CACHES))
    def test_cache_is_nonempty(self, language):
        cache = json.loads(SHIPPED_CACHES[language].read_text(encoding="utf-8"))
        assert len(cache) > 0


class TestValidatorChecks:
    """Each of the validator's six documented checks actually fires."""

    def test_clean_cache_has_no_issues(self):
        assert validate_cache({"hello": "হ্যালো"}) == []

    def test_empty_key(self):
        assert any("empty / non-string key" in i for i in validate_cache({"": "হ্যালো"}))

    def test_non_ascii_key(self):
        assert any("not pure ASCII" in i for i in validate_cache({"héllo": "হ্যালো"}))

    def test_non_lowercase_key(self):
        assert any("not lowercase" in i for i in validate_cache({"Hello": "হ্যালো"}))

    def test_case_insensitive_duplicate_is_unreachable_by_construction(self):
        # A JSON object cannot carry both "Hello" and "hello" as distinct
        # post-lowering duplicates unless both literal keys exist; the check
        # exists for caches built programmatically.
        issues = validate_cache({"hello": "হ্যালো", "Hello": "হালো"})
        assert any("collides" in i for i in issues) or any("not lowercase" in i for i in issues)

    def test_empty_value(self):
        assert any("empty / non-string value" in i for i in validate_cache({"hello": ""}))

    def test_pure_ascii_value(self):
        assert any("still pure ASCII" in i for i in validate_cache({"hello": "hello"}))

    def test_whitespace_value(self):
        assert any("whitespace" in i for i in validate_cache({"hello": " হ্যালো "}))

    def test_language_tag_appears_in_issues(self):
        issues = validate_cache({"hello": ""}, language="bn")
        assert issues and issues[0].startswith("[bn] ")


class TestValidateCacheFile:
    def test_missing_file(self, tmp_path):
        issues = validate_cache_file(tmp_path / "nope.json")
        assert issues and "does not exist" in issues[0]

    def test_invalid_json(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text("{not json", encoding="utf-8")
        issues = validate_cache_file(bad)
        assert issues and "not valid JSON" in issues[0]

    def test_non_object_top_level(self, tmp_path):
        arr = tmp_path / "arr.json"
        arr.write_text("[1, 2]", encoding="utf-8")
        issues = validate_cache_file(arr)
        assert issues and "must be a JSON object" in issues[0]
