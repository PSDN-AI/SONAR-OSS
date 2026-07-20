"""Tests for run.seed wiring via config_loader.get_run_seed."""

from psdn_sonar.config_loader import get_run_seed, load_config


def test_get_run_seed_reads_conf_default():
    cfg = load_config()
    assert get_run_seed(cfg) == 42


def test_get_run_seed_honors_override():
    cfg = load_config(overrides={"run": {"seed": 7}})
    assert get_run_seed(cfg) == 7
