"""Line-ending tolerance of the installed-package gate (issue #250).

The gate compared the wheel's YAML/JSON against the checkout's with raw
``read_bytes()`` and the repository declared no line-ending attributes, so
on a host where git checks files out CRLF the gate failed a clean tree over
content that matches. The comparison now normalises CRLF to LF, and
``.gitattributes`` pins text files to LF.
"""

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

_spec = importlib.util.spec_from_file_location(
    "check_installed_package", REPO_ROOT / "scripts" / "check_installed_package.py"
)
check_installed_package = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check_installed_package)

_content_bytes = check_installed_package._content_bytes


class TestContentComparisonIsLineEndingTolerant:
    def test_crlf_checkout_matches_lf_wheel(self, tmp_path):
        """The issue's measured case: identical YAML, 12 lines, one \\r per
        line of difference — must compare equal."""
        lf = tmp_path / "wheel.yaml"
        crlf = tmp_path / "checkout.yaml"
        content = b"backend: elevenlabs\nmodel: scribe_v1\ntimeout: 30\n"
        lf.write_bytes(content)
        crlf.write_bytes(content.replace(b"\n", b"\r\n"))

        assert _content_bytes(lf) == _content_bytes(crlf)

    def test_real_content_differences_still_detected(self, tmp_path):
        a = tmp_path / "a.yaml"
        b = tmp_path / "b.yaml"
        a.write_bytes(b"timeout: 30\n")
        b.write_bytes(b"timeout: 31\r\n")

        assert _content_bytes(a) != _content_bytes(b)

    def test_json_caches_covered_the_same_way(self, tmp_path):
        # The second comparison site guards the loanword caches (JSON) —
        # the same exposure, per the issue.
        lf = tmp_path / "cache.json"
        crlf = tmp_path / "cache_crlf.json"
        content = b'{\n  "word": "translit"\n}\n'
        lf.write_bytes(content)
        crlf.write_bytes(content.replace(b"\n", b"\r\n"))

        assert _content_bytes(lf) == _content_bytes(crlf)


class TestGitattributesPinsLineEndings:
    def test_text_files_pinned_to_lf(self):
        gitattributes = REPO_ROOT / ".gitattributes"
        if not gitattributes.is_file():
            pytest.skip("running outside a source checkout")
        text = gitattributes.read_text(encoding="utf-8")
        assert "eol=lf" in text
