"""Unweighted macro mean of per-locale metrics.

Benchmark datasets often differ in size by an order of magnitude (e.g. one
locale with ~1500 utterances alongside several with ~100 each). A naive
concat-then-mean over all utterances gives the largest locale almost all the
weight in the headline number.

The macro mean -- the average of per-locale aggregates with one vote per
locale -- is the standard way to report cross-locale benchmark scores for
multilingual / multi-region speech models. This module is the single source
of truth for that aggregation.

The functions only consume already-aggregated per-locale metric values
(ie. the same `avg_wer`, `avg_cer`, ... numbers that the per-dataset
evaluators already write); they do not re-read raw evaluator CSVs.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Mapping, TypedDict

logger = logging.getLogger(__name__)


class LocaleMacroResult(TypedDict):
    """Return shape of :func:`macro_mean_across_locales`.

    Field semantics:

    * ``headline`` -- one entry per metric that had at least one finite
      contribution, mapped to its unweighted mean across the locales that
      contributed.
    * ``per_locale`` -- input values verbatim after ``float`` coercion and
      after dropping non-finite / ``None`` entries. A locale appears here
      only if it contributed at least one finite metric (a locale whose
      every metric is ``None`` / ``NaN`` / ``inf`` is dropped entirely).
    * ``locales`` -- sorted list of the keys of ``per_locale``; never
      contains locales that were dropped above.
    * ``n_locales`` -- ``len(locales)``. This is the number of *contributing*
      locales (those that survived the drop), not the number of locales the
      caller passed in. Callers that need the input count should track it
      themselves.
    """

    headline: dict[str, float]
    per_locale: dict[str, dict[str, float]]
    locales: list[str]
    n_locales: int


class MacroPerModelResult(TypedDict):
    """Return shape of :func:`macro_mean_per_model` for a single model.

    Field semantics mirror :class:`LocaleMacroResult` but scoped to one
    model: ``macro`` is the per-metric headline, ``per_locale`` /
    ``locales`` / ``n_locales`` describe the locales that contributed at
    least one finite metric for *this* model.
    """

    macro: dict[str, float]
    per_locale: dict[str, dict[str, float]]
    locales: list[str]
    n_locales: int


def _coerce_metric_value(value: Any) -> float | None:
    """Return ``value`` as a finite ``float`` or ``None`` if it cannot be used.

    ``None`` / ``NaN`` / ``inf`` are treated as missing rather than as zero so
    that a model with no measurement for a metric does not silently drag the
    macro mean toward 0.
    """
    if value is None:
        return None
    try:
        coerced = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(coerced):
        return None
    return coerced


def macro_mean_across_locales(
    per_locale: Mapping[str, Mapping[str, Any]],
) -> LocaleMacroResult:
    """Compute an unweighted macro mean across locales for one model.

    Each locale contributes equally, regardless of how many utterances it
    has. Each metric is averaged only over the locales that have a finite
    value for it -- locales missing only *some* metrics still contribute
    the metrics they do have.

    A locale whose entries are *all* missing (every metric is ``None`` /
    ``NaN`` / ``inf``, or the per-locale dict is empty) is dropped from
    the result entirely: it does not appear in ``per_locale``, ``locales``,
    or ``n_locales``. ``n_locales`` is therefore the count of *contributing*
    locales, not the number of locales the caller supplied; see
    :class:`LocaleMacroResult` for the full field contract.

    Args:
        per_locale: Mapping of ``{locale_name: {metric_name: value}}``. The
            inner ``value`` may be any object that ``float()`` accepts;
            ``None`` and non-finite floats are treated as missing.

    Returns:
        A :class:`LocaleMacroResult` dict. ``locales`` is sorted for stable
        diff-friendly output. ``headline`` only includes metrics that have
        at least one finite contribution.
    """
    cleaned: dict[str, dict[str, float]] = {}
    for locale, metrics in per_locale.items():
        if metrics is None:
            continue
        locale_metrics: dict[str, float] = {}
        for metric, raw in metrics.items():
            coerced = _coerce_metric_value(raw)
            if coerced is None:
                continue
            locale_metrics[metric] = coerced
        if locale_metrics:
            cleaned[locale] = locale_metrics

    locales = sorted(cleaned)

    # Pivot: metric -> list of per-locale values, in `locales` order.
    per_metric_values: dict[str, list[float]] = {}
    for locale in locales:
        for metric, value in cleaned[locale].items():
            per_metric_values.setdefault(metric, []).append(value)

    headline: dict[str, float] = {
        metric: sum(values) / len(values) for metric, values in per_metric_values.items() if values
    }

    return LocaleMacroResult(
        headline=headline,
        per_locale={locale: dict(cleaned[locale]) for locale in locales},
        locales=locales,
        n_locales=len(locales),
    )


def macro_mean_per_model(
    per_locale_per_model: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    expected_locales: list[str] | None = None,
) -> dict[str, MacroPerModelResult]:
    """Compute :func:`macro_mean_across_locales` for every model.

    The input is the ``{locale: {model: metrics}}`` shape produced naturally
    by the cross-dataset orchestrator (one evaluator per locale, each one
    writing per-model rows). The output is the inverted ``{model: {macro,
    per_locale, ...}}`` shape that the leaderboard / reporting layer consumes.

    Args:
        per_locale_per_model: Mapping of
            ``{locale_name: {model_name: {metric_name: value}}}``. Metric
            values follow the same coercion rules as
            :func:`macro_mean_across_locales`.
        expected_locales: Optional list of locale names that *should* be
            present for every model. Models missing a locale from this set
            get a single warning per missing locale. The macro mean for
            those models is still computed across the locales they do have
            (missing == excluded, never 0). Defaults to the union of locales
            seen in ``per_locale_per_model``.

    Returns:
        Mapping of ``{model_name: {macro, per_locale, locales, n_locales}}``.
        Models are not sorted; preserve insertion order from the first
        locale they appear in for caller-friendly downstream rendering.
    """
    by_model: dict[str, dict[str, dict[str, Any]]] = {}
    for locale, models in per_locale_per_model.items():
        if models is None:
            continue
        for model_name, metrics in models.items():
            if metrics is None:
                continue
            by_model.setdefault(model_name, {})[locale] = dict(metrics)

    seen_locales = sorted(locale for locale, models in per_locale_per_model.items() if models)
    locale_universe = list(expected_locales) if expected_locales is not None else seen_locales

    out: dict[str, MacroPerModelResult] = {}
    for model_name, locales_for_model in by_model.items():
        missing = [locale for locale in locale_universe if locale not in locales_for_model]
        if missing:
            logger.warning(
                "macro_mean_per_model: model %r is missing locales %s; excluding from macro mean (not counted as 0).",
                model_name,
                missing,
            )

        result = macro_mean_across_locales(locales_for_model)
        out[model_name] = MacroPerModelResult(
            macro=result["headline"],
            per_locale=result["per_locale"],
            locales=result["locales"],
            n_locales=result["n_locales"],
        )
    return out
