"""data_downloader without the [cloud] extra (issue #240).

The module imported boto3 at top level, so without the extra it could not be
imported at all: ``scripts/download_data.py --help`` died on a boto3
traceback into package internals that never named the extra — and --help is
the command a user runs to find out what they need. The import is now
guarded, and the requirement is enforced in ``DataDownloader.__init__`` with
the same install-message shape the other extras use.

These tests deliberately do NOT ``importorskip("boto3")`` — they cover the
boto3-less path, simulated whether or not boto3 is installed here.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from psdn_sonar.utils import data_downloader

REPO_ROOT = Path(__file__).resolve().parents[2]


class TestModuleImportsWithoutBoto3:
    def test_downloader_raises_actionable_import_error(self, monkeypatch):
        monkeypatch.setattr(data_downloader, "boto3", None)
        with pytest.raises(ImportError, match=r"psdn-sonar\[cloud\]"):
            data_downloader.DataDownloader()

    def test_help_prints_usage_without_boto3(self, tmp_path):
        """The issue's exact reproduction: --help must print usage and exit 0
        even when importing boto3 fails."""
        # A fake boto3 whose import fails exactly like an absent package.
        (tmp_path / "boto3.py").write_text("raise ImportError(\"No module named 'boto3'\")\n", encoding="utf-8")
        env = dict(os.environ)
        env["PYTHONPATH"] = str(tmp_path) + os.pathsep + env.get("PYTHONPATH", "")

        result = subprocess.run(
            [sys.executable, str(REPO_ROOT / "scripts" / "download_data.py"), "--help"],
            capture_output=True,
            text=True,
            env=env,
            cwd=str(REPO_ROOT),
        )

        assert result.returncode == 0, result.stderr
        assert "usage" in result.stdout.lower()
        assert "Traceback" not in result.stderr

    def test_module_import_survives_missing_boto3(self, tmp_path):
        """Direct import of the module (the issue's second reproduction)."""
        (tmp_path / "boto3.py").write_text("raise ImportError(\"No module named 'boto3'\")\n", encoding="utf-8")
        env = dict(os.environ)
        env["PYTHONPATH"] = str(tmp_path) + os.pathsep + env.get("PYTHONPATH", "")

        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "from psdn_sonar.utils import data_downloader; "
                "assert data_downloader.boto3 is None; "
                "print('import ok')",
            ],
            capture_output=True,
            text=True,
            env=env,
            cwd=str(REPO_ROOT),
        )

        assert result.returncode == 0, result.stderr
        assert "import ok" in result.stdout
