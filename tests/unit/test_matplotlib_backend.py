"""Issue #280: the suite pins a headless matplotlib backend.

Nothing in the repository set one, so plot tests used whatever the host
resolved to; on a host with an interactive default one plot test per full
run failed on the display connection (a different one each time), and each
passed alone. conftest.py now pins Agg for the process and exports
MPLBACKEND for subprocesses, so a full run does not depend on whether the
host can open a display.
"""

import os

import pytest


class TestHeadlessBackendPinned:
    def test_suite_runs_on_a_headless_backend(self):
        matplotlib = pytest.importorskip("matplotlib")
        assert matplotlib.get_backend().lower() == "agg"

    def test_backend_env_var_is_exported_for_subprocesses(self):
        assert os.environ.get("MPLBACKEND")
