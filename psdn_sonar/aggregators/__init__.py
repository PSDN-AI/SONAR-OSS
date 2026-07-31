"""Cross-dataset / cross-locale aggregation primitives.

These helpers consume already-aggregated per-locale metric values (the same
per-locale means that the per-dataset evaluators produce) and combine them
into headline numbers that are robust to locale size imbalance.
"""

from psdn_sonar.aggregators.locale_macro import (
    LocaleMacroResult,
    MacroPerModelResult,
    macro_mean_across_locales,
    macro_mean_per_model,
)

__all__ = [
    "LocaleMacroResult",
    "MacroPerModelResult",
    "macro_mean_across_locales",
    "macro_mean_per_model",
]
