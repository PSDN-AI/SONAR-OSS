"""Unit tests for ``psdn_sonar.utils.llm_metrics``.

These tests use a minimal stub Gemini client (no network) so the
careful ``None`` semantics around missing data, the entity-empty
exclusion rule, the prompt-version cache key, and the retry policy
can be regression-protected without an API key.

The tests are deliberately hermetic: the optional ``google-genai``
dependency (which lives in the ``[apis]`` extra) is NOT required to
run them. If the real SDK isn't installed we inject a minimal stub
into ``sys.modules`` BEFORE importing ``llm_metrics`` so that
``_call_llm`` (which does ``from google.genai import types`` inside
the function body) can resolve a fake ``GenerateContentConfig`` and
exception classes. This mirrors what CI sees when the workflow
installs ``[dev]`` but skips ``[apis]``.
"""

from __future__ import annotations

import importlib
import json
import os
import sys
import types as _types
from dataclasses import dataclass
from pathlib import Path

import pytest


def _install_google_genai_stub() -> None:
    """Install a minimal ``google.genai`` stub if the real SDK is missing.

    Provides just enough surface area for our wrapper functions:
      * ``google.genai.types.GenerateContentConfig`` — accepts kwargs
        and stashes them as attributes; ``_call_llm`` builds one of
        these and passes it through to ``client.models.generate_content``.
      * ``google.genai.errors.{APIError, ClientError, ServerError}`` —
        used by ``_is_retryable`` to classify exceptions. Tests don't
        raise these directly (we test with ``ConnectionError`` /
        ``ValueError`` / ``RuntimeError``), but the import must succeed.
      * ``google.genai.Client`` — present so callers that try to build
        a real client get something that doesn't immediately explode,
        though no test path in this file actually constructs one.
    """
    try:
        importlib.import_module("google.genai.types")
        importlib.import_module("google.genai.errors")
        return
    except Exception:
        pass

    class _GenerateContentConfig:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    class _APIError(Exception):
        pass

    class _ClientError(_APIError):
        pass

    class _ServerError(_APIError):
        pass

    class _Client:
        def __init__(self, api_key=None):
            self.api_key = api_key

    google_pkg = sys.modules.setdefault("google", _types.ModuleType("google"))
    if not hasattr(google_pkg, "__path__"):
        google_pkg.__path__ = []  # type: ignore[attr-defined]

    genai_pkg = _types.ModuleType("google.genai")
    genai_types = _types.ModuleType("google.genai.types")
    genai_errors = _types.ModuleType("google.genai.errors")

    genai_types.GenerateContentConfig = _GenerateContentConfig
    genai_errors.APIError = _APIError
    genai_errors.ClientError = _ClientError
    genai_errors.ServerError = _ServerError
    genai_pkg.Client = _Client
    genai_pkg.types = genai_types
    genai_pkg.errors = genai_errors

    sys.modules["google.genai"] = genai_pkg
    sys.modules["google.genai.types"] = genai_types
    sys.modules["google.genai.errors"] = genai_errors
    google_pkg.genai = genai_pkg  # type: ignore[attr-defined]


_install_google_genai_stub()

from psdn_sonar.utils import llm_metrics  # noqa: E402  — import after stub install

# ---------------------------------------------------------------------------
# Stub Gemini client
# ---------------------------------------------------------------------------


@dataclass
class _StubCandidate:
    finish_reason: str = "STOP"


class _StubResponse:
    def __init__(self, text=None, finish_reason="STOP"):
        self.text = text
        self.candidates = [_StubCandidate(finish_reason=finish_reason)]


class _StubModels:
    def __init__(self, return_value=None, raise_seq=None, finish_reason="STOP"):
        self._return_value = return_value
        self._raise_seq = list(raise_seq or [])
        self._finish_reason = finish_reason
        self.calls: list[dict] = []

    def generate_content(self, *, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        if self._raise_seq:
            exc = self._raise_seq.pop(0)
            if exc is not None:
                raise exc
        text = self._return_value
        if isinstance(text, dict):
            text = json.dumps(text)
        return _StubResponse(text=text, finish_reason=self._finish_reason)


class StubClient:
    def __init__(self, return_value=None, raise_seq=None, finish_reason="STOP"):
        self.models = _StubModels(return_value=return_value, raise_seq=raise_seq, finish_reason=finish_reason)


# ---------------------------------------------------------------------------
# _parse_json
# ---------------------------------------------------------------------------


class TestParseJson:
    def test_parses_raw_json(self):
        assert llm_metrics._parse_json('{"a": 1}') == {"a": 1}

    def test_strips_markdown_fence(self):
        assert llm_metrics._parse_json('```json\n{"a": 1}\n```') == {"a": 1}

    def test_strips_bare_fence(self):
        assert llm_metrics._parse_json('```\n{"a": 1}\n```') == {"a": 1}

    def test_malformed_json_raises(self):
        with pytest.raises(json.JSONDecodeError):
            llm_metrics._parse_json("not json at all")


# ---------------------------------------------------------------------------
# _is_retryable
# ---------------------------------------------------------------------------


class TestIsRetryable:
    def test_value_error_not_retryable(self):
        assert llm_metrics._is_retryable(ValueError("nope")) is False

    def test_runtime_error_not_retryable(self):
        assert llm_metrics._is_retryable(RuntimeError("nope")) is False

    def test_connection_error_retryable(self):
        assert llm_metrics._is_retryable(ConnectionError("tcp reset")) is True

    def test_httpx_timeout_retryable_if_installed(self):
        try:
            import httpx
        except ImportError:
            pytest.skip("httpx not installed")
        assert llm_metrics._is_retryable(httpx.ReadTimeout("read timeout")) is True
        assert llm_metrics._is_retryable(httpx.ConnectError("connect")) is True


# ---------------------------------------------------------------------------
# _score_from_entities — entity-empty exclusion (#3, #4)
# ---------------------------------------------------------------------------


class TestScoreFromEntities:
    def test_zero_entities_returns_none(self):
        score, total, correct = llm_metrics._score_from_entities([])
        assert score is None
        assert total == 0
        assert correct == 0

    def test_all_correct_returns_one(self):
        entities = [{"correct": True}, {"correct": True}]
        score, total, correct = llm_metrics._score_from_entities(entities)
        assert score == 1.0
        assert total == 2
        assert correct == 2

    def test_partial_correct(self):
        entities = [{"correct": True}, {"correct": False}, {"correct": True}]
        score, total, correct = llm_metrics._score_from_entities(entities)
        assert score == pytest.approx(2 / 3)
        assert total == 3
        assert correct == 2

    def test_missing_correct_key_treated_as_false(self):
        entities = [{"text": "x"}, {"correct": True}]
        score, total, correct = llm_metrics._score_from_entities(entities)
        assert score == 0.5
        assert total == 2
        assert correct == 1

    def test_non_list_returns_none(self):
        score, total, correct = llm_metrics._score_from_entities("not a list")
        assert score is None and total == 0 and correct == 0

    def test_python_computed_so_llm_arithmetic_is_irrelevant(self):
        """Even if the LLM had returned ``score=1.0`` in the JSON, the
        Python recompute is the source of truth — verifying the fix
        for #4 (LLMs are unreliable arithmetic agents)."""
        entities = [{"correct": False}, {"correct": False}]
        score, _, _ = llm_metrics._score_from_entities(entities)
        assert score == 0.0


# ---------------------------------------------------------------------------
# evaluate_entity_preservation — #3 zero-entity exclusion via stub
# ---------------------------------------------------------------------------


class TestEvaluateEntityPreservation:
    def test_zero_entities_returns_none_score(self):
        client = StubClient(return_value={"entities": []})
        out = llm_metrics.evaluate_entity_preservation("ref", "pred", client=client)
        assert out["entity_score"] is None, "entity-empty utterances must be excluded from aggregation"
        assert out["total_entities"] == 0
        assert out["correct_entities"] == 0

    def test_score_recomputed_in_python(self):
        """Even if the LLM returns an arithmetically-wrong ``score`` field,
        the recomputed ``entity_score`` reflects the entities list."""
        client = StubClient(
            return_value={
                "entities": [{"correct": True}, {"correct": False}],
                "score": 0.99,  # ignored
            }
        )
        out = llm_metrics.evaluate_entity_preservation("ref", "pred", client=client)
        assert out["entity_score"] == 0.5

    def test_failure_returns_none(self):
        client = StubClient(raise_seq=[RuntimeError("boom")])
        out = llm_metrics.evaluate_entity_preservation("ref", "pred", client=client)
        assert out["entity_score"] is None
        assert "error" in out

    def test_malformed_json_returns_none(self):
        client = StubClient(return_value="not json")
        out = llm_metrics.evaluate_entity_preservation("ref", "pred", client=client)
        assert out["entity_score"] is None


# ---------------------------------------------------------------------------
# evaluate_intent_preservation
# ---------------------------------------------------------------------------


class TestEvaluateIntentPreservation:
    def test_value_one_returns_one(self):
        client = StubClient(return_value={"intent_preserved": 1, "reasoning": "ok"})
        out = llm_metrics.evaluate_intent_preservation("ref", "pred", client=client)
        assert out["intent_preserved"] == 1
        assert out["reasoning"] == "ok"

    def test_value_zero_returns_zero(self):
        client = StubClient(return_value={"intent_preserved": 0, "reasoning": "no"})
        out = llm_metrics.evaluate_intent_preservation("ref", "pred", client=client)
        assert out["intent_preserved"] == 0

    def test_missing_key_returns_none(self):
        client = StubClient(return_value={"reasoning": "no intent key"})
        out = llm_metrics.evaluate_intent_preservation("ref", "pred", client=client)
        assert out["intent_preserved"] is None

    def test_non_binary_value_returns_none(self):
        client = StubClient(return_value={"intent_preserved": 2})
        out = llm_metrics.evaluate_intent_preservation("ref", "pred", client=client)
        assert out["intent_preserved"] is None

    def test_failure_returns_none(self):
        client = StubClient(raise_seq=[ValueError("boom")])
        out = llm_metrics.evaluate_intent_preservation("ref", "pred", client=client)
        assert out["intent_preserved"] is None
        assert "error" in out


# ---------------------------------------------------------------------------
# evaluate_sample — combined call (#10)
# ---------------------------------------------------------------------------


class TestEvaluateSampleCombined:
    def test_single_call_returns_both_metrics(self):
        client = StubClient(
            return_value={
                "entities": [{"correct": True}, {"correct": True}, {"correct": False}],
                "intent_preserved": 1,
                "intent_reasoning": "core meaning preserved",
            }
        )
        out = llm_metrics.evaluate_sample("ref", "pred", client=client)
        assert out.entity_score == pytest.approx(2 / 3)
        assert out.intent_preserved == 1
        assert out.intent_reasoning == "core meaning preserved"
        assert len(client.models.calls) == 1, "combined prompt must use a single round-trip"

    def test_zero_entities_with_intent_one(self):
        client = StubClient(
            return_value={
                "entities": [],
                "intent_preserved": 1,
                "intent_reasoning": "fine",
            }
        )
        out = llm_metrics.evaluate_sample("ref", "pred", client=client)
        assert out.entity_score is None, "no entities -> excluded"
        assert out.intent_preserved == 1

    def test_failure_returns_none_for_both(self):
        client = StubClient(raise_seq=[RuntimeError("boom")])
        out = llm_metrics.evaluate_sample("ref", "pred", client=client)
        assert out.entity_score is None
        assert out.intent_preserved is None
        assert out.error is not None


# ---------------------------------------------------------------------------
# _call_llm — #5 (None text + MAX_TOKENS handling)
# ---------------------------------------------------------------------------


class TestCallLLM:
    def test_none_text_raises_runtime_error(self):
        client = StubClient(return_value=None, finish_reason="SAFETY")
        with pytest.raises(RuntimeError, match="Empty Gemini response.*finish_reason=SAFETY"):
            llm_metrics._call_llm(client, "system", "user")

    def test_max_tokens_finish_reason_raises(self):
        # Even with non-None text, MAX_TOKENS means JSON is truncated.
        client = StubClient(return_value='{"partial":', finish_reason="MAX_TOKENS")
        with pytest.raises(RuntimeError, match="MAX_TOKENS"):
            llm_metrics._call_llm(client, "system", "user")

    def test_retries_on_connection_error_then_succeeds(self):
        client = StubClient(
            return_value={"intent_preserved": 1, "reasoning": "x"},
            raise_seq=[ConnectionError("flake")],
        )
        # initial_backoff_sec=0 to skip sleeps in tests
        out = llm_metrics._call_llm(
            client, "system", "user", max_retries=2, initial_backoff_sec=0, backoff_multiplier=1
        )
        assert "intent_preserved" in out

    def test_does_not_retry_value_error(self):
        client = StubClient(raise_seq=[ValueError("nope")])
        with pytest.raises(ValueError):
            llm_metrics._call_llm(client, "system", "user", max_retries=3, initial_backoff_sec=0, backoff_multiplier=1)
        assert len(client.models.calls) == 1, "non-retryable exception must NOT trigger retries"


# ---------------------------------------------------------------------------
# make_cache_key — colocated with PROMPT_VERSION
# ---------------------------------------------------------------------------


class TestMakeCacheKey:
    def test_includes_all_four_components(self):
        key = llm_metrics.make_cache_key("audio/foo.wav", "Whisper", "gemini-3.1-pro-preview")
        assert "audio/foo.wav" in key
        assert "Whisper" in key
        assert "gemini-3.1-pro-preview" in key
        assert llm_metrics.PROMPT_VERSION in key

    def test_different_judge_yields_different_key(self):
        a = llm_metrics.make_cache_key("a.wav", "Whisper", "gemini-3.1-pro-preview")
        b = llm_metrics.make_cache_key("a.wav", "Whisper", "gemini-2.5-pro")
        assert a != b

    def test_repeat_index_zero_is_legacy_key(self):
        # The variance harness must not change the key for a single-shot
        # run: repeat_index=0 has to be byte-identical to the no-arg call so
        # pre-harness caches keep hitting.
        legacy = llm_metrics.make_cache_key("a.wav", "Whisper", "gemini-2.5-pro")
        with_zero = llm_metrics.make_cache_key("a.wav", "Whisper", "gemini-2.5-pro", repeat_index=0)
        assert legacy == with_zero

    def test_repeat_index_extends_key_and_avoids_collisions(self):
        base = llm_metrics.make_cache_key("a.wav", "Whisper", "gemini-2.5-pro", repeat_index=0)
        r1 = llm_metrics.make_cache_key("a.wav", "Whisper", "gemini-2.5-pro", repeat_index=1)
        r2 = llm_metrics.make_cache_key("a.wav", "Whisper", "gemini-2.5-pro", repeat_index=2)
        # Each repeat is distinct from repeat 0 and from each other.
        assert len({base, r1, r2}) == 3
        assert r1.endswith("||r1")
        assert r2.endswith("||r2")

    def test_negative_repeat_index_rejected(self):
        with pytest.raises(ValueError, match="repeat_index must be >= 0"):
            llm_metrics.make_cache_key("a.wav", "Whisper", "gemini-2.5-pro", repeat_index=-1)


# ---------------------------------------------------------------------------
# PROMPT_VERSION cache invalidation (#1)
# ---------------------------------------------------------------------------


class TestPromptVersion:
    def test_prompt_version_is_stable_8_char_hex(self):
        v = llm_metrics.PROMPT_VERSION
        assert isinstance(v, str)
        assert len(v) == 8
        assert all(c in "0123456789abcdef" for c in v)

    def test_prompt_version_changes_when_prompt_changes(self, monkeypatch):
        original = llm_metrics._prompt_version()
        monkeypatch.setattr(llm_metrics, "_COMBINED_SYSTEM_PROMPT", "edited prompt")
        new = llm_metrics._prompt_version()
        assert original != new, "editing any system prompt must change PROMPT_VERSION"


# ---------------------------------------------------------------------------
# get_client — #17 fail-fast on missing API key, #188 .env credential contract
# ---------------------------------------------------------------------------


class TestGetClient:
    @pytest.fixture(autouse=True)
    def _no_ambient_keys(self, monkeypatch):
        """Strip ambient credentials so each test controls its own world.

        ``load_env`` is stubbed to a no-op by default: a developer's real
        repository ``.env`` (which may well contain a Gemini key — that's
        the whole point of #188) must not leak into these tests. Tests
        that exercise the .env path replace the stub with a fake loader.
        """
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.setattr(llm_metrics, "load_env", lambda: None)

    def test_missing_api_key_raises(self):
        with pytest.raises(RuntimeError, match="Gemini API key"):
            llm_metrics.get_client()

    def test_missing_key_error_names_dotenv_mechanism(self):
        """#188: the old message said only "in the environment", while the
        ElevenLabs adapter in the same codebase said ".env or as env var" —
        two adapters giving opposite instructions for the same contract.
        The message must name both mechanisms and point at .env.example."""
        with pytest.raises(RuntimeError) as exc_info:
            llm_metrics.get_client()
        message = str(exc_info.value)
        assert ".env" in message
        assert ".env.example" in message
        assert "environment variable" in message

    def test_dotenv_key_is_loaded_before_the_check(self, monkeypatch):
        """#188 regression: a key that only exists in .env (nothing exported
        in the shell) must be visible to get_client(). Simulated by a fake
        load_env that injects the key the way python-dotenv would."""
        monkeypatch.setattr(llm_metrics, "load_env", lambda: monkeypatch.setenv("GEMINI_API_KEY", "key-from-dotenv"))
        client = llm_metrics.get_client()
        assert client is not None

    def test_google_api_key_fallback_also_loaded_from_dotenv(self, monkeypatch):
        monkeypatch.setattr(
            llm_metrics, "load_env", lambda: monkeypatch.setenv("GOOGLE_API_KEY", "fallback-from-dotenv")
        )
        client = llm_metrics.get_client()
        assert client is not None


class TestCredentialContractDocs:
    """#188: the four descriptions of the Gemini credential contract used to
    disagree — README silent, .env.example missing the key, the error message
    saying "in the environment" only, and the TestLiveSmoke docstring claiming
    the README documents the preferred env name. Pin the repo documents so
    they can't silently drift apart again. (The error-message leg is pinned
    by TestGetClient.test_missing_key_error_names_dotenv_mechanism.)"""

    @staticmethod
    def _repo_root():
        from pathlib import Path

        return Path(llm_metrics.__file__).resolve().parents[2]

    def test_readme_documents_the_env_names(self):
        readme = (self._repo_root() / "README.md").read_text(encoding="utf-8")
        assert "GEMINI_API_KEY" in readme, "TestLiveSmoke's docstring promises the README documents this name"
        assert "GOOGLE_API_KEY" in readme
        assert "llm_metrics" in readme

    def test_env_example_lists_the_gemini_keys(self):
        env_example = (self._repo_root() / ".env.example").read_text(encoding="utf-8")
        assert "GEMINI_API_KEY" in env_example
        assert "GOOGLE_API_KEY" in env_example


# ---------------------------------------------------------------------------
# _coerce_intent_value — C2 regression (asymmetric data loss on stringified ints)
# ---------------------------------------------------------------------------


class TestCoerceIntentValue:
    """The previous gate ``intent not in (0, 1, True, False)`` silently
    dropped JSON-mode degraded responses like ``"intent_preserved": "1"``
    as ``None``, producing asymmetric data loss invisible in aggregates.
    These tests pin the new tolerant coercion contract."""

    @pytest.mark.parametrize(
        "value,expected",
        [
            (0, 0),
            (1, 1),
            (True, 1),
            (False, 0),
            ("0", 0),
            ("1", 1),
            ("  1 ", 1),
            ("\n0\n", 0),
        ],
    )
    def test_accepts_valid_forms(self, value, expected):
        assert llm_metrics._coerce_intent_value(value) == expected

    @pytest.mark.parametrize(
        "value",
        [None, "", "yes", "no", "true", "false", "01", "2", 2, -1, 0.5, 1.0, [], {}, "1.0"],
    )
    def test_rejects_ambiguous_forms(self, value):
        assert llm_metrics._coerce_intent_value(value) is None

    def test_evaluate_intent_preservation_recovers_string_one(self):
        client = StubClient(return_value={"intent_preserved": "1", "reasoning": "ok"})
        result = llm_metrics.evaluate_intent_preservation("ref", "pred", client=client)
        assert result["intent_preserved"] == 1, "stringified '1' must NOT be silently dropped"

    def test_evaluate_intent_preservation_recovers_string_zero(self):
        client = StubClient(return_value={"intent_preserved": "0", "reasoning": "ok"})
        result = llm_metrics.evaluate_intent_preservation("ref", "pred", client=client)
        assert result["intent_preserved"] == 0

    def test_evaluate_sample_recovers_string_intent(self):
        client = StubClient(
            return_value={
                "entities": [{"correct": True}],
                "intent_preserved": "1",
                "intent_reasoning": "ok",
            }
        )
        out = llm_metrics.evaluate_sample("ref", "pred", client=client)
        assert out.intent_preserved == 1


# ---------------------------------------------------------------------------
# DEFAULT_MODEL — N2 regression
# ---------------------------------------------------------------------------


class TestDefaultModelIsStable:
    """The published numbers must be reproducible long after preview-tier
    model aliases get retired or renamed. This guards against an
    accidental flip back to a ``-preview`` / ``-exp-`` default.

    Issue #187 showed the tier heuristic is only half the story — Google
    retired the *stable* 2.5 generation for new users while the previews
    stayed up — so these static checks are necessary but not sufficient;
    the live check is ``TestLiveSmoke``, run by the scheduled
    ``live-gemini`` workflow (see ``TestLiveSmokeIsWired``)."""

    def test_default_model_is_not_preview(self):
        m = llm_metrics.DEFAULT_MODEL
        lowered = m.lower()
        assert "preview" not in lowered, f"DEFAULT_MODEL must be a stable alias, got {m!r}"
        assert "-exp-" not in lowered, f"DEFAULT_MODEL must not be an experimental alias, got {m!r}"
        assert lowered.startswith("gemini-"), f"DEFAULT_MODEL must be a Gemini family model, got {m!r}"

    def test_default_model_is_not_in_the_retired_25_generation(self):
        """#187: the whole stable Gemini 2.5 generation is retired for new
        users (both ``gemini-2.5-pro`` and ``gemini-2.5-flash`` 404 with
        "no longer available to new users"), and both names the module
        hardcoded were in it. Pin that no name from that generation comes
        back as the default."""
        lowered = llm_metrics.DEFAULT_MODEL.lower()
        assert not lowered.startswith("gemini-2."), (
            f"DEFAULT_MODEL {llm_metrics.DEFAULT_MODEL!r} is in the Gemini 2.x generation, "
            "which Google retired for new users (issue #187) — the judge would 404 on its own default"
        )


class TestLiveSmokeIsWired:
    """#187, second half: ``TestLiveSmoke`` was written to catch a retired
    ``DEFAULT_MODEL`` — the exact failure that then happened — but nothing
    triggered it: zero occurrences of ``RUN_LIVE_GEMINI_TESTS`` anywhere
    under ``.github/workflows/``. Pin the wiring so the workflow can't be
    deleted or defanged without this suite noticing."""

    @staticmethod
    def _workflow_text() -> str:
        repo_root = Path(llm_metrics.__file__).resolve().parents[2]
        workflow = repo_root / ".github" / "workflows" / "live-gemini.yml"
        assert workflow.is_file(), (
            "The scheduled live-gemini workflow is gone; without it the live "
            "DEFAULT_MODEL check runs nowhere (issue #187)"
        )
        return workflow.read_text(encoding="utf-8")

    def test_workflow_runs_on_a_schedule(self):
        text = self._workflow_text()
        assert "schedule:" in text and "cron:" in text, (
            "live-gemini.yml must run on a schedule — a manual-only trigger "
            "recreates the 'wired to nothing' state issue #187 reported"
        )

    def test_workflow_sets_the_opt_in_and_selects_the_marker(self):
        text = self._workflow_text()
        assert "RUN_LIVE_GEMINI_TESTS" in text
        assert "-m live_gemini" in text
        assert "secrets.GEMINI_API_KEY" in text


# ---------------------------------------------------------------------------
# Live-API smoke test — N3 (skip-if-no-key)
# ---------------------------------------------------------------------------


def _live_gemini_skip_reason() -> str | None:
    """Opt-in gate so CI never hits the network even if a key is present."""
    if os.getenv("RUN_LIVE_GEMINI_TESTS") != "1":
        return "Set RUN_LIVE_GEMINI_TESTS=1 to enable live Gemini API tests (network)"
    if not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
        return "Live test requires GEMINI_API_KEY or GOOGLE_API_KEY"
    return None


_LIVE_GEMINI_SKIP = _live_gemini_skip_reason()


@pytest.mark.live_gemini
@pytest.mark.skipif(_LIVE_GEMINI_SKIP is not None, reason=_LIVE_GEMINI_SKIP or "")
class TestLiveSmoke:
    """Validate that the literal ``DEFAULT_MODEL`` string is a name the
    live Gemini API currently accepts. Unit tests use a stub client and
    cannot catch the failure mode where Google rotates a preview alias
    out from under us. This test fires exactly one tiny request and
    asserts we get a parseable JSON intent answer back.

    **Not part of PR CI**: requires ``RUN_LIVE_GEMINI_TESTS=1``
    *and* ``GEMINI_API_KEY`` or ``GOOGLE_API_KEY`` — same credential
    contract as ``get_client()`` (preferred env name documented in README).
    The scheduled ``live-gemini`` workflow runs it weekly (issue #187:
    every hardcoded judge model name had been retired for new users and
    this test, the one thing written to notice, was wired to nothing).
    """

    def test_default_model_string_is_live(self):
        try:
            from google import genai  # noqa: F401
        except ImportError:
            pytest.skip("google-genai SDK not installed (install with the [apis] extra)")

        client = llm_metrics.get_client()
        result = llm_metrics.evaluate_intent_preservation(
            reference="The quick brown fox jumps over the lazy dog.",
            prediction="The quick brown fox jumps over the lazy dog.",
            client=client,
            model=llm_metrics.DEFAULT_MODEL,
        )
        # We do NOT assert intent_preserved == 1 (single live request, no
        # retries here, judges can occasionally be quirky on tautologies).
        # We DO assert we got SOMETHING back — i.e. the model string is
        # currently a valid Gemini alias and the SDK round-trip works.
        assert "intent_preserved" in result, f"live request didn't return expected shape: {result!r}"
        assert result.get("error") is None, f"live request errored: {result.get('error')!r}"
