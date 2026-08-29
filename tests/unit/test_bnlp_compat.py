"""Tests for the guarded bnlp import (issue #204).

bnlp's module body checks NLTK for ``tokenizers/punkt`` but downloads
``punkt_tab`` — mismatched names — so on an environment holding only
``punkt_tab`` the check fails forever and ``nltk.download`` reaches the
network on every import, with no timeout. ``text_processing`` imported bnlp
at module scope, putting that round trip (and one observed indefinite hang)
on every evaluation in any install with the ``[bengali]`` extra, regardless
of language.

These tests run without bnlp installed: a fake package reproducing the real
module body verbatim stands in for it, so the guard is exercised against the
actual defect mechanics in every CI environment.
"""

import socket
import subprocess
import sys
import textwrap

import pytest

from psdn_sonar.utils import bnlp_compat, text_processing

_REAL_BNLP_BODY = textwrap.dedent(
    '''
    """Fake bnlp reproducing the real package's import-time NLTK check."""

    import nltk

    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        print("punkt not found. downloading...")
        nltk.download("punkt_tab")


    class BasicTokenizer:
        def tokenize(self, text):
            return ["bnlp-token", *text.split()]
    '''
)


@pytest.fixture
def isolated_bnlp(monkeypatch):
    """No bnlp module cached, tokenizer cache reset; cleans up test imports."""
    saved = {name: sys.modules.pop(name) for name in list(sys.modules) if name.split(".")[0] == "bnlp"}
    monkeypatch.setattr(text_processing, "_bengali_tokenizer", text_processing._BNLP_UNSET)
    yield
    for name in list(sys.modules):
        if name.split(".")[0] == "bnlp":
            del sys.modules[name]
    sys.modules.update(saved)


@pytest.fixture
def fake_bnlp(tmp_path, monkeypatch, isolated_bnlp):
    import importlib

    pkg = tmp_path / "bnlp"
    pkg.mkdir()
    (pkg / "__init__.py").write_text(_REAL_BNLP_BODY, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    return pkg


def _find_with_punkt_tab_only(resource, *args, **kwargs):
    """NLTK data state from the issue: punkt_tab present, legacy punkt absent."""
    if resource == "tokenizers/punkt_tab":
        return "/fake/nltk_data/tokenizers/punkt_tab"
    raise LookupError(resource)


def _find_nothing(resource, *args, **kwargs):
    raise LookupError(resource)


class TestModuleImportIsLazy:
    def test_importing_text_processing_does_not_import_bnlp(self):
        """The import every evaluation performs must not pull in bnlp — that
        is what put the network call on English/Hindi/Korean runs too."""
        code = (
            "import sys; import psdn_sonar.utils.text_processing; "
            "sys.exit('bnlp was imported at module scope' if 'bnlp' in sys.modules else 0)"
        )
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr


class TestGuardedImport:
    def test_offline_when_punkt_tab_is_local(self, fake_bnlp, monkeypatch, capsys):
        """The issue's exact environment: punkt_tab local, punkt absent.
        The import must succeed with no download call and no 'punkt not
        found' noise."""
        import nltk

        monkeypatch.setattr(nltk.data, "find", _find_with_punkt_tab_only)

        def _network_reached(*args, **kwargs):
            raise AssertionError("nltk.download reached the network despite punkt_tab being local")

        monkeypatch.setattr(nltk, "download", _network_reached)

        module = bnlp_compat.import_bnlp()

        assert module is not None
        assert module.BasicTokenizer().tokenize("a b") == ["bnlp-token", "a", "b"]
        assert "punkt not found" not in capsys.readouterr().out
        # Patches restored: the originals (our stubs) are back in place.
        assert nltk.data.find is _find_with_punkt_tab_only
        assert nltk.download is _network_reached

    def test_download_is_bounded_when_punkt_tab_is_missing(self, fake_bnlp, monkeypatch):
        """A genuinely required first-time download must run under a socket
        timeout, so a dead connection fails instead of hanging forever."""
        import nltk

        monkeypatch.setattr(nltk.data, "find", _find_nothing)
        observed = {}

        def _fake_download(resource, *args, **kwargs):
            observed["resource"] = resource
            observed["timeout"] = socket.getdefaulttimeout()
            return True

        monkeypatch.setattr(nltk, "download", _fake_download)
        assert socket.getdefaulttimeout() is None  # the unbounded default

        module = bnlp_compat.import_bnlp()

        assert module is not None
        assert observed["resource"] == "punkt_tab"
        assert observed["timeout"] == bnlp_compat.NLTK_DOWNLOAD_TIMEOUT_S
        assert socket.getdefaulttimeout() is None  # restored

    def test_returns_none_when_bnlp_not_installed(self, isolated_bnlp, monkeypatch):
        monkeypatch.setattr(bnlp_compat, "bnlp_installed", lambda: False)
        assert bnlp_compat.import_bnlp() is None

    def test_already_imported_module_is_returned_as_is(self, isolated_bnlp, monkeypatch):
        marker = object()
        monkeypatch.setitem(sys.modules, "bnlp", marker)
        assert bnlp_compat.import_bnlp() is marker

    def test_patches_restored_when_the_import_itself_fails(self, tmp_path, monkeypatch, isolated_bnlp):
        import importlib

        import nltk

        pkg = tmp_path / "bnlp"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("raise RuntimeError('broken bnlp')", encoding="utf-8")
        monkeypatch.syspath_prepend(str(tmp_path))
        importlib.invalidate_caches()

        monkeypatch.setattr(nltk.data, "find", _find_with_punkt_tab_only)
        sentinel_download = lambda *a, **k: True  # noqa: E731
        monkeypatch.setattr(nltk, "download", sentinel_download)

        with pytest.raises(RuntimeError, match="broken bnlp"):
            bnlp_compat.import_bnlp()

        assert nltk.data.find is _find_with_punkt_tab_only
        assert nltk.download is sentinel_download
        assert socket.getdefaulttimeout() is None


class TestLazyTokenizerAndContract:
    def test_tokenize_falls_back_to_whitespace_without_bnlp(self, monkeypatch):
        monkeypatch.setattr(text_processing, "_bengali_tokenizer", None)
        assert text_processing._tokenize_bengali("এক দুই") == ["এক", "দুই"]

    def test_loader_loads_through_the_guard_and_caches(self, fake_bnlp, monkeypatch):
        import nltk

        monkeypatch.setattr(nltk.data, "find", _find_with_punkt_tab_only)
        monkeypatch.setattr(nltk, "download", lambda *a, **k: (_ for _ in ()).throw(AssertionError("network reached")))

        first = text_processing._load_bengali_tokenizer()
        assert first is not None
        assert text_processing._tokenize_bengali("a b") == ["bnlp-token", "a", "b"]
        assert text_processing._load_bengali_tokenizer() is first  # cached

    def test_contract_marks_availability_from_the_actual_load(self, fake_bnlp, monkeypatch):
        import nltk

        monkeypatch.setattr(nltk.data, "find", _find_with_punkt_tab_only)
        assert text_processing.wer_normalization_contract("bn") == "bn:v4+bnlp"

    def test_contract_marks_minus_bnlp_when_not_installed(self, isolated_bnlp, monkeypatch):
        monkeypatch.setattr(bnlp_compat, "bnlp_installed", lambda: False)
        assert text_processing.wer_normalization_contract("bn") == "bn:v4-bnlp"

    def test_broken_bnlp_degrades_to_whitespace_and_minus_bnlp(self, monkeypatch, caplog):
        """A bnlp that raises on load must not take the run down: whitespace
        fallback, a warning, and the contract records -bnlp (it marks the
        tokenizer in force, not what is installed)."""
        monkeypatch.setattr(text_processing, "_bengali_tokenizer", text_processing._BNLP_UNSET)

        def _broken_import():
            raise RuntimeError("broken bnlp")

        monkeypatch.setattr(bnlp_compat, "import_bnlp", _broken_import)
        with caplog.at_level("WARNING"):
            assert text_processing._load_bengali_tokenizer() is None
        assert "falls back to whitespace splitting" in caplog.text
        assert text_processing.wer_normalization_contract("bn") == "bn:v4-bnlp"
        assert text_processing._tokenize_bengali("এক দুই") == ["এক", "দুই"]
