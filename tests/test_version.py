"""Smoke test for the package skeleton: importable, versioned."""

from importlib.metadata import version

import psdn_sonar


def test_version_is_exposed():
    assert psdn_sonar.__version__ == version("psdn-sonar")
