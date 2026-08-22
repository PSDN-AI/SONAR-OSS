"""Render a leaderboard from ``scores_*.json`` run artifacts.

Every number shown comes from an artifact written by a real evaluation run
(:func:`psdn_sonar.benchmark.scores.write_scores_json`). Nothing here derives,
back-solves, or extrapolates a metric: a metric a run did not measure is
displayed as missing (issue PSDN-AI/SONAR-OSS#117, where a published
leaderboard back-solved WER/CER from POSEIDON and rendered generated
distributions as measurements).
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional

from .scores import RunScoresArtifact

logger = logging.getLogger(__name__)

SCORES_GLOB = "scores_*.json"

# Column key -> (header, higher_is_better). Sorting places rows missing the
# chosen metric last regardless of direction.
SORTABLE_METRICS = {
    "poseidon": ("POSEIDON", True),
    "semantic": ("Semantic", True),
    "wer": ("WER", False),
    "cer": ("CER", False),
}

_MISSING = "—"


@dataclass
class LoadedRun:
    """One successfully parsed scores artifact and where it came from."""

    path: Path
    artifact: RunScoresArtifact


@dataclass
class LeaderboardRow:
    """Aggregated view of all runs for one (model, language) pair."""

    model_name: str
    language: Optional[str]
    runs: int
    wer: Optional[float]
    cer: Optional[float]
    semantic: Optional[float]
    poseidon: Optional[float]
    successful: int
    failed: int
    # True when any contributing run recorded run-level warnings (e.g. a
    # reference-script/--language mismatch); such numbers are suspect.
    has_warnings: bool


def run_language(artifact: RunScoresArtifact) -> Optional[str]:
    """Language code the run was configured with, if recorded."""
    value = artifact.submission.inference_params.get("language_code")
    return str(value) if value else None


def collect_scores(roots: Iterable[Path | str]) -> tuple[list[LoadedRun], list[str]]:
    """Recursively load every ``scores_*.json`` under ``roots``.

    Returns the parsed runs plus one human-readable message per file that
    could not be parsed (unreadable, invalid JSON, or not a scores artifact).
    Unparseable files are skipped, never guessed at.
    """
    loaded: list[LoadedRun] = []
    skipped: list[str] = []
    for root in roots:
        for path in sorted(Path(root).rglob(SCORES_GLOB)):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                artifact = RunScoresArtifact.model_validate(payload)
            except (OSError, ValueError) as e:
                # pydantic.ValidationError subclasses ValueError.
                skipped.append(f"Skipping {path}: not a readable scores artifact ({e})")
                continue
            loaded.append(LoadedRun(path=path, artifact=artifact))
    return loaded, skipped


def _mean(values: list[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def build_leaderboard(
    runs: Iterable[LoadedRun],
    language: Optional[str] = None,
    sort: str = "poseidon",
) -> list[LeaderboardRow]:
    """Group runs by (model, language) and aggregate their measured metrics.

    Each metric is the mean of the per-run means that actually measured it;
    a metric measured by no run stays ``None`` (rendered as missing) rather
    than being substituted or derived from other columns.
    """
    if sort not in SORTABLE_METRICS:
        raise ValueError(f"Unknown sort key '{sort}'. Choose one of: {', '.join(SORTABLE_METRICS)}.")

    groups: dict[tuple[str, Optional[str]], list[RunScoresArtifact]] = {}
    for run in runs:
        lang = run_language(run.artifact)
        if language is not None and lang != language:
            continue
        groups.setdefault((run.artifact.model_name, lang), []).append(run.artifact)

    rows: list[LeaderboardRow] = []
    for (model_name, lang), artifacts in groups.items():
        aggregates = [a.aggregate for a in artifacts]
        rows.append(
            LeaderboardRow(
                model_name=model_name,
                language=lang,
                runs=len(artifacts),
                wer=_mean([a.wer_mean for a in aggregates if a.wer_mean is not None]),
                cer=_mean([a.cer_mean for a in aggregates if a.cer_mean is not None]),
                semantic=_mean(
                    [a.semantic_similarity_mean for a in aggregates if a.semantic_similarity_mean is not None]
                ),
                poseidon=_mean([a.poseidon_score_mean for a in aggregates if a.poseidon_score_mean is not None]),
                successful=sum(a.successful for a in aggregates),
                failed=sum(a.failed for a in aggregates),
                has_warnings=any(a.warnings for a in artifacts),
            )
        )

    _, higher_is_better = SORTABLE_METRICS[sort]

    def sort_key(row: LeaderboardRow):
        value = getattr(row, sort)
        missing = value is None
        if missing:
            value = 0.0
        ordered = -value if higher_is_better else value
        return (missing, ordered, row.model_name, row.language or "")

    return sorted(rows, key=sort_key)


def _fmt(value: Optional[float]) -> str:
    return _MISSING if value is None else f"{value:.3f}"


def render_leaderboard(rows: list[LeaderboardRow], sort: str = "poseidon") -> str:
    """Fixed-width text table of measured results.

    Missing metrics render as an em dash; they are never inferred from the
    other columns. Rows whose runs carried warnings are marked with ``!``.
    """
    header = ["Model", "Lang", "Runs", "WER", "CER", "Semantic", "POSEIDON", "OK", "Failed"]
    body: list[list[str]] = []
    for row in rows:
        marker = " !" if row.has_warnings else ""
        body.append(
            [
                f"{row.model_name}{marker}",
                row.language or _MISSING,
                str(row.runs),
                _fmt(row.wer),
                _fmt(row.cer),
                _fmt(row.semantic),
                _fmt(row.poseidon),
                str(row.successful),
                str(row.failed),
            ]
        )

    widths = [max(len(header[i]), *(len(r[i]) for r in body)) for i in range(len(header))]
    lines = [
        "  ".join(h.ljust(widths[i]) for i, h in enumerate(header)).rstrip(),
        "  ".join("-" * widths[i] for i in range(len(header))),
    ]
    lines.extend("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(cells)).rstrip() for cells in body)

    sort_header, higher = SORTABLE_METRICS[sort]
    lines.append("")
    lines.append(f"Sorted by {sort_header} ({'desc' if higher else 'asc'}); rows without {sort_header} sort last.")
    lines.append(f"{_MISSING} = not measured by any contributing run; values are never derived from other columns.")
    if any(row.has_warnings for row in rows):
        lines.append(
            "! = at least one contributing run recorded configuration warnings in scores.json; "
            "treat that row's numbers as suspect."
        )
    return "\n".join(lines)


def rows_as_json(rows: list[LeaderboardRow]) -> str:
    """Machine-readable rendering of the same aggregation."""
    return json.dumps([asdict(row) for row in rows], indent=2, ensure_ascii=False)
