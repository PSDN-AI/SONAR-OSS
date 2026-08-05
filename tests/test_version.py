"""Smoke test for the package skeleton: importable, versioned."""

import psdn_sonar


def test_version_is_exposed():
    assert psdn_sonar.__version__ == "0.1.0.dev1"
