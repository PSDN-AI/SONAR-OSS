"""Tests for ``psdn_sonar.aggregators.locale_macro``.

These cover the three core behaviours of the macro mean:

  (a) equal-sized locales reproduce simple-mean,
  (b) one large + several small locales differ from a naive concat-then-mean,
  (c) a model that's missing a locale is excluded from that locale's mean
      (with a logged warning) rather than treated as 0.
"""

from __future__ import annotations

import logging

import pytest

from psdn_sonar.aggregators import (
    macro_mean_across_locales,
    macro_mean_per_model,
)


def test_equal_sized_locales_match_simple_mean():
    """AC (a): when each locale carries equal weight, macro == arithmetic mean."""
    per_locale = {
        "fleurs": {"wer": 0.30, "cer": 0.10},
        "openslr_bd": {"wer": 0.40, "cer": 0.20},
        "commonvoice": {"wer": 0.50, "cer": 0.30},
    }
    result = macro_mean_across_locales(per_locale)

    assert result["headline"]["wer"] == pytest.approx(0.40)
    assert result["headline"]["cer"] == pytest.approx(0.20)
    assert result["n_locales"] == 3
    assert result["locales"] == ["commonvoice", "fleurs", "openslr_bd"]
    assert result["per_locale"]["fleurs"]["wer"] == pytest.approx(0.30)


def test_one_large_locale_differs_from_naive_concat_mean():
    """AC (b): macro mean weights every locale equally regardless of size.

    Naive concat-then-mean would weight by sample count, which here drags the
    headline toward the large locale's value. The macro mean must instead
    average the per-locale aggregates (one vote per locale).
    """
    # Caller passes the per-locale aggregates (each locale already collapsed
    # to one number per metric); we just check we produce the unweighted mean.
    per_locale = {
        "large_dataset": {"wer": 0.20},
        "fleurs": {"wer": 0.50},
        "openslr_bd": {"wer": 0.55},
        "commonvoice": {"wer": 0.45},
    }
    result = macro_mean_across_locales(per_locale)

    expected_macro = (0.20 + 0.50 + 0.55 + 0.45) / 4
    assert result["headline"]["wer"] == pytest.approx(expected_macro)

    # Sanity: a size-weighted mean (using 1500 / 100 / 100 / 100 utterances)
    # would give a meaningfully different number, demonstrating the macro
    # mean's purpose.
    sizes = {"large_dataset": 1500, "fleurs": 100, "openslr_bd": 100, "commonvoice": 100}
    weighted = sum(per_locale[k]["wer"] * sizes[k] for k in per_locale) / sum(sizes.values())
    # Naive concat-then-mean lands at 0.25 (the big locale dominates); the
    # macro mean is 0.425. Asserting both spellings prevents future drift
    # from accidentally collapsing macro back onto the size-weighted answer.
    assert weighted == pytest.approx(0.25, abs=1e-4)
    assert result["headline"]["wer"] != pytest.approx(weighted, abs=1e-3)


def test_missing_metric_in_one_locale_is_dropped_not_zero():
    """A metric absent from one locale must not be counted as 0."""
    per_locale = {
        "fleurs": {"wer": 0.30, "semantic_similarity": 0.95},
        "openslr_bd": {"wer": 0.40},
    }
    result = macro_mean_across_locales(per_locale)

    assert result["headline"]["wer"] == pytest.approx(0.35)
    assert result["headline"]["semantic_similarity"] == pytest.approx(0.95)


def test_non_finite_and_none_values_are_treated_as_missing():
    per_locale = {
        "fleurs": {"wer": 0.30, "cer": float("nan")},
        "openslr_bd": {"wer": 0.40, "cer": None},
        "commonvoice": {"wer": 0.50, "cer": 0.20},
    }
    result = macro_mean_across_locales(per_locale)

    assert result["headline"]["wer"] == pytest.approx(0.40)
    assert result["headline"]["cer"] == pytest.approx(0.20)
    assert result["per_locale"]["fleurs"] == {"wer": 0.30}


def test_empty_input_returns_empty_result():
    result = macro_mean_across_locales({})
    assert result == {
        "headline": {},
        "per_locale": {},
        "locales": [],
        "n_locales": 0,
    }


def test_locale_with_all_missing_metrics_is_dropped_from_n_locales():
    """``n_locales`` counts contributing locales, not supplied locales.

    A locale whose every metric is ``None`` / ``NaN`` / empty drops out of
    ``per_locale``, ``locales`` and ``n_locales`` together. Callers who
    need the supplied count must track it themselves -- this is documented
    on :class:`LocaleMacroResult` so we pin it as a contract test here.
    """
    per_locale = {
        "fleurs": {"wer": 0.30, "cer": 0.10},
        "openslr_bd": {"wer": float("nan"), "cer": None},
        "commonvoice": {},
        "large_dataset": {"wer": 0.50, "cer": 0.20},
    }
    result = macro_mean_across_locales(per_locale)

    assert result["n_locales"] == 2
    assert result["locales"] == ["fleurs", "large_dataset"]
    assert "openslr_bd" not in result["per_locale"]
    assert "commonvoice" not in result["per_locale"]
    assert result["headline"]["wer"] == pytest.approx((0.30 + 0.50) / 2)
    assert result["headline"]["cer"] == pytest.approx((0.10 + 0.20) / 2)


def test_locales_list_is_sorted_for_stable_output():
    per_locale = {
        "z_locale": {"wer": 0.1},
        "a_locale": {"wer": 0.2},
        "m_locale": {"wer": 0.3},
    }
    result = macro_mean_across_locales(per_locale)
    assert result["locales"] == ["a_locale", "m_locale", "z_locale"]


def test_per_model_inverts_locale_first_to_model_first():
    per_locale_per_model = {
        "fleurs": {
            "whisper-1": {"wer": 0.30, "cer": 0.10},
            "assemblyai": {"wer": 0.35, "cer": 0.12},
        },
        "openslr_bd": {
            "whisper-1": {"wer": 0.40, "cer": 0.20},
            "assemblyai": {"wer": 0.42, "cer": 0.22},
        },
    }
    result = macro_mean_per_model(per_locale_per_model)

    assert set(result.keys()) == {"whisper-1", "assemblyai"}
    assert result["whisper-1"]["macro"]["wer"] == pytest.approx(0.35)
    assert result["whisper-1"]["macro"]["cer"] == pytest.approx(0.15)
    assert result["assemblyai"]["macro"]["wer"] == pytest.approx(0.385)
    assert result["whisper-1"]["per_locale"]["fleurs"]["wer"] == pytest.approx(0.30)
    assert result["whisper-1"]["n_locales"] == 2
    assert result["whisper-1"]["locales"] == ["fleurs", "openslr_bd"]


def test_per_model_excludes_missing_locale_and_warns(caplog):
    """AC (c): a model present in only some locales is averaged across those it has,
    not silently treated as 0 in the others. A warning identifies what was dropped."""
    per_locale_per_model = {
        "fleurs": {
            "whisper-1": {"wer": 0.30},
            "assemblyai": {"wer": 0.35},
        },
        "openslr_bd": {
            "whisper-1": {"wer": 0.40},
            # assemblyai missing here on purpose
        },
        "commonvoice": {
            "whisper-1": {"wer": 0.50},
            "assemblyai": {"wer": 0.45},
        },
    }

    with caplog.at_level(logging.WARNING, logger="psdn_sonar.aggregators.locale_macro"):
        result = macro_mean_per_model(per_locale_per_model)

    # whisper-1 has all three locales -> simple average of 0.30/0.40/0.50.
    assert result["whisper-1"]["macro"]["wer"] == pytest.approx(0.40)
    assert result["whisper-1"]["n_locales"] == 3

    # assemblyai is missing openslr_bd -> macro must be (0.35+0.45)/2 = 0.40,
    # NOT (0.35+0.45+0.0)/3 = 0.2666.
    assert result["assemblyai"]["macro"]["wer"] == pytest.approx(0.40)
    assert result["assemblyai"]["n_locales"] == 2
    assert "openslr_bd" not in result["assemblyai"]["per_locale"]

    warning_messages = [rec.message for rec in caplog.records if rec.levelno >= logging.WARNING]
    assert any("assemblyai" in msg and "openslr_bd" in msg for msg in warning_messages), warning_messages


def test_per_model_respects_expected_locales_for_warning_universe(caplog):
    """``expected_locales`` allows callers to assert a stricter universe than what's seen."""
    per_locale_per_model = {
        "fleurs": {"whisper-1": {"wer": 0.3}},
        "openslr_bd": {"whisper-1": {"wer": 0.4}},
    }

    with caplog.at_level(logging.WARNING, logger="psdn_sonar.aggregators.locale_macro"):
        macro_mean_per_model(
            per_locale_per_model,
            expected_locales=["fleurs", "openslr_bd", "commonvoice"],
        )

    messages = [rec.message for rec in caplog.records if rec.levelno >= logging.WARNING]
    assert any("commonvoice" in msg for msg in messages), messages


def test_per_model_no_warning_when_all_models_complete(caplog):
    per_locale_per_model = {
        "fleurs": {"whisper-1": {"wer": 0.3}, "assemblyai": {"wer": 0.35}},
        "openslr_bd": {"whisper-1": {"wer": 0.4}, "assemblyai": {"wer": 0.42}},
    }
    with caplog.at_level(logging.WARNING, logger="psdn_sonar.aggregators.locale_macro"):
        macro_mean_per_model(per_locale_per_model)
    assert caplog.records == []
