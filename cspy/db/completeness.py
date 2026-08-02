"""Estimate CSP sampling completeness from a clustered mol-CSPy database."""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import sqlite3
import sys
from collections import Counter
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from statistics import NormalDist
from typing import Iterable, Sequence, TextIO

import numpy as np


DEFAULT_WINDOWS = (5.0, 10.0, 15.0)


@dataclass(frozen=True)
class ClusteredObservation:
    """One valid final structure and its cluster representative."""

    structure_id: str
    representative_id: str
    spacegroup: str
    representative_spacegroup: str
    energy: float
    representative_energy: float
    trial_number: int | None
    multiplicity: int


@dataclass(frozen=True)
class Estimate:
    """Good--Turing sample coverage and the Chao lower bound."""

    N: int
    S_obs: int
    f1: int
    f2: int
    good_turing_coverage: float | None
    chao_S: float
    chao_unseen: float


@dataclass(frozen=True)
class SummaryRow:
    """One completeness result for a scope and optional energy window."""

    scope: str
    spacegroup: str
    energy_window_kj_mol: float | None
    energy_cutoff: float | None
    global_min_energy: float
    bootstrap_samples: int
    confidence_level: float
    random_seed: int
    N: int
    S_obs: int
    f1: int
    f2: int
    good_turing_coverage: float | None
    good_turing_percent: float | None
    good_turing_ci_low: float | None
    good_turing_ci_high: float | None
    good_turing_percent_ci_low: float | None
    good_turing_percent_ci_high: float | None
    chao_S: float
    chao_unseen: float
    chao_S_ci_low: float | None
    chao_S_ci_high: float | None


@dataclass(frozen=True)
class ConvergenceRow:
    """Completeness at one sampling checkpoint."""

    sample_size: int
    scope: str
    spacegroup: str
    energy_window_kj_mol: float | None
    energy_cutoff: float | None
    N: int
    S_obs: int
    f1: int
    f2: int
    good_turing_coverage: float | None
    good_turing_percent: float | None
    chao_S: float
    chao_unseen: float


def _normalize_spacegroup(value: object) -> str:
    try:
        parsed = float(value)
        if math.isfinite(parsed) and parsed.is_integer():
            return str(int(parsed))
    except (TypeError, ValueError):
        pass
    return str(value)


def _spacegroup_sort_key(value: str) -> tuple[int, float | str]:
    try:
        return (0, float(value))
    except ValueError:
        return (1, value)


def _table_columns(
    connection: sqlite3.Connection, table_name: str
) -> set[str]:
    escaped = table_name.replace('"', '""')
    return {
        str(row[1]).lower()
        for row in connection.execute(f'PRAGMA table_info("{escaped}")')
    }


def _validate_database_schema(connection: sqlite3.Connection) -> None:
    tables = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )
    }
    required_tables = {"crystal", "trial_structure", "equivalent_to"}
    missing_tables = required_tables - tables
    if missing_tables:
        raise ValueError(
            "Database is missing required table(s): "
            + ", ".join(sorted(missing_tables))
        )

    required_columns = {
        "crystal": {"id", "spacegroup", "energy"},
        "trial_structure": {
            "id",
            "minimization_step",
            "trial_number",
            "valid",
        },
        "equivalent_to": {"unique_id", "equivalent_id"},
    }
    for table, expected in required_columns.items():
        missing = expected - _table_columns(connection, table)
        if missing:
            raise ValueError(
                f"Table {table!r} is missing column(s): "
                + ", ".join(sorted(missing))
            )


def _cluster_parents(
    connection: sqlite3.Connection,
    valid_ids: set[str],
    *,
    strict_assignments: bool = True,
) -> dict[str, str]:
    rows = list(
        connection.execute(
            "SELECT unique_id, equivalent_id FROM equivalent_to"
        )
    )
    if not rows:
        raise ValueError(
            "Database is not clustered: equivalent_to contains no assignments"
        )

    parents: dict[str, str] = {}
    assigned_ids: set[str] = set()
    for unique_raw, equivalent_raw in rows:
        if unique_raw is None:
            continue
        unique_id = str(unique_raw).strip()
        if unique_id in valid_ids:
            assigned_ids.add(unique_id)
        if equivalent_raw is None:
            continue
        equivalent_id = str(equivalent_raw).strip()
        if equivalent_id in valid_ids:
            assigned_ids.add(equivalent_id)
        if not unique_id or not equivalent_id or unique_id == equivalent_id:
            continue
        if equivalent_id not in valid_ids:
            continue
        previous = parents.get(equivalent_id)
        if previous is not None and previous != unique_id:
            raise ValueError(
                f"Structure {equivalent_id!r} is assigned to both "
                f"{previous!r} and {unique_id!r}"
            )
        parents[equivalent_id] = unique_id
    if strict_assignments:
        missing = sorted(valid_ids - assigned_ids)
        if missing:
            preview = ", ".join(missing[:5])
            suffix = " ..." if len(missing) > 5 else ""
            raise ValueError(
                "equivalent_to has no assignment for valid final structure(s): "
                f"{preview}{suffix}"
            )
    return parents


def _resolve_representative(
    structure_id: str, parents: dict[str, str], valid_ids: set[str]
) -> str:
    current = structure_id
    seen: set[str] = set()
    while current in parents:
        if current in seen:
            raise ValueError(
                f"Cycle in equivalent_to involving structure {structure_id!r}"
            )
        seen.add(current)
        current = parents[current]
    if current not in valid_ids:
        raise ValueError(
            f"Representative {current!r} for structure {structure_id!r} "
            "is not a valid final minimization"
        )
    return current


def load_database_observations(
    filename: str | Path,
    *,
    strict_assignments: bool = True,
) -> list[ClusteredObservation]:
    """Load one observation per valid structure at the final minimization step.

    ``equivalent_to`` layouts that encode representatives with either a NULL
    equivalent ID or a self-reference are both accepted. Duplicate
    ``trial_structure`` rows are collapsed by structure ID.
    """

    path = Path(filename)
    uri = f"{path.resolve().as_uri()}?mode=ro"
    try:
        connection = sqlite3.connect(uri, uri=True)
    except sqlite3.Error as exc:
        raise ValueError(f"Unable to open SQLite database {path}: {exc}") from exc

    try:
        _validate_database_schema(connection)
        final_step = connection.execute(
            "SELECT MAX(minimization_step) FROM trial_structure"
        ).fetchone()[0]
        if final_step is None:
            raise ValueError("Database has no minimization records")

        records = list(
            connection.execute(
                """
                WITH selected AS (
                    SELECT id, MAX(rowid) AS selected_rowid
                    FROM trial_structure
                    WHERE valid = 1 AND minimization_step = ?
                    GROUP BY id
                )
                SELECT c.id, c.spacegroup, c.energy, t.trial_number
                FROM selected AS s
                INNER JOIN trial_structure AS t ON t.rowid = s.selected_rowid
                INNER JOIN crystal AS c ON c.id = s.id
                ORDER BY c.id
                """,
                (final_step,),
            )
        )
        if not records:
            raise ValueError("Database has no valid final minimizations")

        by_id: dict[str, tuple[str, float, int | None]] = {}
        for structure_raw, spacegroup_raw, energy_raw, trial_number in records:
            structure_id = str(structure_raw).strip()
            try:
                energy = float(energy_raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Structure {structure_id!r} has invalid energy {energy_raw!r}"
                ) from exc
            if not math.isfinite(energy):
                raise ValueError(
                    f"Structure {structure_id!r} has non-finite energy"
                )
            by_id[structure_id] = (
                _normalize_spacegroup(spacegroup_raw),
                energy,
                trial_number,
            )

        valid_ids = set(by_id)
        parents = _cluster_parents(
            connection,
            valid_ids,
            strict_assignments=strict_assignments,
        )
    except sqlite3.Error as exc:
        raise ValueError(f"Unable to read {path}: {exc}") from exc
    finally:
        connection.close()

    representatives = {
        structure_id: _resolve_representative(structure_id, parents, valid_ids)
        for structure_id in valid_ids
    }
    multiplicities = Counter(representatives.values())

    observations = []
    for structure_id in sorted(valid_ids):
        spacegroup, energy, trial_number = by_id[structure_id]
        representative_id = representatives[structure_id]
        representative_spacegroup, representative_energy, _ = by_id[
            representative_id
        ]
        observations.append(
            ClusteredObservation(
                structure_id=structure_id,
                representative_id=representative_id,
                spacegroup=spacegroup,
                representative_spacegroup=representative_spacegroup,
                energy=energy,
                representative_energy=representative_energy,
                trial_number=trial_number,
                multiplicity=multiplicities[representative_id],
            )
        )
    return observations


def combine_database_observations(
    filenames: Sequence[str | Path],
    *,
    global_clusters: str | Path | None = None,
    strict_assignments: bool = True,
) -> list[ClusteredObservation]:
    """Load observations from one or more clustered source databases.

    When several databases are clustered together, each source database stores
    its within-database equivalences while the reduced output database stores
    equivalences between their local representatives.  ``global_clusters`` is
    that reduced output database; supplying it composes both levels so that a
    rediscovery in another input database is counted in the same cluster.
    """

    paths = list(filenames)
    if not paths:
        raise ValueError("At least one source database is required")

    observations = [
        observation
        for filename in paths
        for observation in load_database_observations(
            filename, strict_assignments=strict_assignments
        )
    ]
    structure_ids = [item.structure_id for item in observations]
    duplicate_ids = sorted(
        structure_id
        for structure_id, count in Counter(structure_ids).items()
        if count > 1
    )
    if duplicate_ids:
        preview = ", ".join(duplicate_ids[:5])
        suffix = " ..." if len(duplicate_ids) > 5 else ""
        raise ValueError(
            "Structure IDs must be unique across source databases; repeated: "
            f"{preview}{suffix}"
        )

    if global_clusters is None:
        if len(paths) > 1:
            raise ValueError(
                "Multiple source databases require --global-clusters with the "
                "reduced database produced by clustering them together"
            )
        return observations

    global_observations = load_database_observations(
        global_clusters, strict_assignments=strict_assignments
    )
    global_by_id = {
        item.structure_id: item for item in global_observations
    }
    local_representatives = {item.representative_id for item in observations}
    missing = sorted(local_representatives - global_by_id.keys())
    if missing:
        preview = ", ".join(missing[:5])
        suffix = " ..." if len(missing) > 5 else ""
        raise ValueError(
            "Global cluster database does not contain local representative(s): "
            f"{preview}{suffix}"
        )

    combined = []
    for item in observations:
        global_item = global_by_id[item.representative_id]
        combined.append(
            replace(
                item,
                representative_id=global_item.representative_id,
                representative_spacegroup=global_item.representative_spacegroup,
                representative_energy=global_item.representative_energy,
                multiplicity=0,
            )
        )

    multiplicities = Counter(item.representative_id for item in combined)
    return [
        replace(item, multiplicity=multiplicities[item.representative_id])
        for item in combined
    ]


def estimate_completeness(multiplicities: Iterable[int]) -> Estimate:
    """Calculate Good--Turing coverage and the Chao lower bound."""

    counts = list(multiplicities)
    if any(isinstance(n, bool) or not isinstance(n, int) or n <= 0 for n in counts):
        raise ValueError("All multiplicities must be positive integers")

    N = sum(counts)
    S_obs = len(counts)
    f1 = sum(n == 1 for n in counts)
    f2 = sum(n == 2 for n in counts)
    coverage = None if N == 0 else 1.0 - (f1 / N)
    if f2 > 0:
        chao_unseen = (f1 * f1) / (2.0 * f2)
    else:
        chao_unseen = f1 * (f1 - 1) / 2.0

    return Estimate(
        N=N,
        S_obs=S_obs,
        f1=f1,
        f2=f2,
        good_turing_coverage=coverage,
        chao_S=S_obs + chao_unseen,
        chao_unseen=chao_unseen,
    )


def _bootstrap_intervals(
    representative_ids: Sequence[str],
    *,
    samples: int,
    confidence_level: float,
    rng: np.random.Generator,
) -> tuple[float | None, float | None, float | None, float | None]:
    """Parametric bootstrap intervals including estimated unseen mass.

    A plain empirical bootstrap turns sampled singletons into artificial
    rediscoveries and is severely biased when coverage is low. This bootstrap
    reserves the Good--Turing unseen mass and distributes it over the Chao
    estimate of unseen species before drawing each replicate.
    """

    if not representative_ids or samples == 0:
        return (None, None, None, None)

    observed_counts = np.asarray(
        list(Counter(representative_ids).values()), dtype=np.int64
    )
    point = estimate_completeness(int(value) for value in observed_counts)
    unseen_mass = point.f1 / point.N
    unseen_species = (
        max(1, int(round(point.chao_unseen))) if unseen_mass > 0 else 0
    )
    observed_probabilities = observed_counts / observed_counts.sum()
    coverage_values = np.empty(samples, dtype=float)
    chao_values = np.empty(samples, dtype=float)
    for index in range(samples):
        unseen_draws = int(rng.binomial(point.N, unseen_mass))
        observed_draws = point.N - unseen_draws
        if observed_draws:
            resampled_observed = rng.multinomial(
                observed_draws, observed_probabilities
            )
        else:
            resampled_observed = np.zeros_like(observed_counts)
        if unseen_draws:
            _labels, resampled_unseen = np.unique(
                rng.integers(0, unseen_species, size=unseen_draws),
                return_counts=True,
            )
        else:
            resampled_unseen = np.asarray([], dtype=np.int64)
        counts = np.concatenate(
            [
                resampled_observed[resampled_observed > 0],
                resampled_unseen,
            ]
        )
        estimate = estimate_completeness(
            int(value) for value in counts if value > 0
        )
        coverage_values[index] = estimate.good_turing_coverage
        chao_values[index] = estimate.chao_S

    degrees = 1 if samples > 1 else 0
    z_value = NormalDist().inv_cdf(0.5 + confidence_level / 2.0)
    coverage_standard_error = float(np.std(coverage_values, ddof=degrees))
    coverage_low = max(
        0.0, point.good_turing_coverage - z_value * coverage_standard_error
    )
    coverage_high = min(
        1.0, point.good_turing_coverage + z_value * coverage_standard_error
    )
    positive_chao = chao_values[chao_values > 0]
    if point.chao_S > 0 and len(positive_chao):
        log_standard_error = float(
            np.std(np.log(positive_chao), ddof=degrees)
        )
        chao_low = max(
            float(point.S_obs),
            math.exp(math.log(point.chao_S) - z_value * log_standard_error),
        )
        chao_high = math.exp(math.log(point.chao_S) + z_value * log_standard_error)
    else:
        chao_low = chao_high = point.chao_S
    return (
        float(coverage_low),
        float(coverage_high),
        float(chao_low),
        float(chao_high),
    )


def summarize(
    observations: Sequence[ClusteredObservation],
    windows: Sequence[float] = DEFAULT_WINDOWS,
    *,
    by_spacegroup: bool = True,
    bootstrap_samples: int = 1000,
    confidence_level: float = 0.95,
    random_seed: int = 0,
) -> list[SummaryRow]:
    """Summarize global and per-space-group sample coverage.

    Global results combine rediscoveries across space groups. Per-space-group
    results count the observations originating from that space group while
    retaining the global representative and global energy cutoff.
    """

    if not observations:
        raise ValueError("No clustered observations were read")
    if any(window < 0 or not math.isfinite(window) for window in windows):
        raise ValueError("Energy windows must be finite and non-negative")
    if (
        isinstance(bootstrap_samples, bool)
        or not isinstance(bootstrap_samples, int)
        or bootstrap_samples < 0
    ):
        raise ValueError("bootstrap_samples must be a non-negative integer")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between zero and one")

    global_min = min(item.representative_energy for item in observations)
    spacegroups = sorted(
        {item.spacegroup for item in observations}, key=_spacegroup_sort_key
    )
    rows: list[SummaryRow] = []

    def append_scope(
        subset: Sequence[ClusteredObservation],
        scope: str,
        spacegroup: str,
        window: float | None,
        cutoff: float | None,
    ) -> None:
        multiplicities = Counter(item.representative_id for item in subset)
        estimate = estimate_completeness(multiplicities.values())
        coverage = estimate.good_turing_coverage
        seed_material = (
            f"{random_seed}|{scope}|{spacegroup}|{window}|{cutoff}"
        ).encode()
        row_seed = int.from_bytes(
            hashlib.sha256(seed_material).digest()[:8], "little"
        )
        coverage_low, coverage_high, chao_low, chao_high = _bootstrap_intervals(
            [item.representative_id for item in subset],
            samples=bootstrap_samples,
            confidence_level=confidence_level,
            rng=np.random.default_rng(row_seed),
        )
        rows.append(
            SummaryRow(
                scope=scope,
                spacegroup=spacegroup,
                energy_window_kj_mol=window,
                energy_cutoff=cutoff,
                global_min_energy=global_min,
                bootstrap_samples=bootstrap_samples,
                confidence_level=confidence_level,
                random_seed=random_seed,
                **asdict(estimate),
                good_turing_percent=None if coverage is None else 100.0 * coverage,
                good_turing_ci_low=coverage_low,
                good_turing_ci_high=coverage_high,
                good_turing_percent_ci_low=(
                    None if coverage_low is None else 100.0 * coverage_low
                ),
                good_turing_percent_ci_high=(
                    None if coverage_high is None else 100.0 * coverage_high
                ),
                chao_S_ci_low=chao_low,
                chao_S_ci_high=chao_high,
            )
        )

    append_scope(observations, "all", "ALL", None, None)
    if by_spacegroup:
        for spacegroup in spacegroups:
            append_scope(
                [item for item in observations if item.spacegroup == spacegroup],
                "all",
                spacegroup,
                None,
                None,
            )

    for window in windows:
        cutoff = global_min + window
        selected = [
            item for item in observations if item.representative_energy <= cutoff
        ]
        append_scope(selected, "window", "ALL", window, cutoff)
        if by_spacegroup:
            for spacegroup in spacegroups:
                append_scope(
                    [item for item in selected if item.spacegroup == spacegroup],
                    "window",
                    spacegroup,
                    window,
                    cutoff,
                )
    return rows


def _trial_sort_key(item: ClusteredObservation) -> tuple:
    if item.trial_number is None:
        return (1, 0, 0.0, item.structure_id)
    try:
        return (0, 0, float(item.trial_number), item.structure_id)
    except (TypeError, ValueError):
        return (0, 1, str(item.trial_number), item.structure_id)


def convergence(
    observations: Sequence[ClusteredObservation],
    windows: Sequence[float] = DEFAULT_WINDOWS,
    *,
    points: int = 20,
    by_spacegroup: bool = True,
) -> list[ConvergenceRow]:
    """Build a coverage curve from prefixes ordered by trial number."""

    if not observations:
        raise ValueError("No clustered observations were read")
    if isinstance(points, bool) or not isinstance(points, int) or points < 1:
        raise ValueError("convergence points must be a positive integer")
    if any(window < 0 or not math.isfinite(window) for window in windows):
        raise ValueError("Energy windows must be finite and non-negative")

    global_min = min(item.representative_energy for item in observations)
    scopes = [("all", None, None)] + [
        ("window", window, global_min + window) for window in windows
    ]
    rows = []
    groups = [("ALL", list(observations))]
    if by_spacegroup:
        groups.extend(
            (
                spacegroup,
                [item for item in observations if item.spacegroup == spacegroup],
            )
            for spacegroup in sorted(
                {item.spacegroup for item in observations},
                key=_spacegroup_sort_key,
            )
        )

    for spacegroup, group in groups:
        ordered = sorted(group, key=_trial_sort_key)
        size = len(ordered)
        checkpoints = sorted(
            {
                max(1, math.ceil(index * size / min(points, size)))
                for index in range(1, min(points, size) + 1)
            }
        )
        for sample_size in checkpoints:
            prefix = ordered[:sample_size]
            for scope, window, cutoff in scopes:
                selected = (
                    prefix
                    if cutoff is None
                    else [
                        item
                        for item in prefix
                        if item.representative_energy <= cutoff
                    ]
                )
                multiplicities = Counter(
                    item.representative_id for item in selected
                )
                estimate = estimate_completeness(multiplicities.values())
                coverage_value = estimate.good_turing_coverage
                rows.append(
                    ConvergenceRow(
                        sample_size=sample_size,
                        scope=scope,
                        spacegroup=spacegroup,
                        energy_window_kj_mol=window,
                        energy_cutoff=cutoff,
                        **asdict(estimate),
                        good_turing_percent=(
                            None
                            if coverage_value is None
                            else 100.0 * coverage_value
                        ),
                    )
                )
    return rows


OUTPUT_FIELDS = (
    "scope",
    "spacegroup",
    "energy_window_kj_mol",
    "energy_cutoff",
    "global_min_energy",
    "bootstrap_samples",
    "confidence_level",
    "random_seed",
    "N",
    "S_obs",
    "f1",
    "f2",
    "good_turing_coverage",
    "good_turing_percent",
    "good_turing_ci_low",
    "good_turing_ci_high",
    "good_turing_percent_ci_low",
    "good_turing_percent_ci_high",
    "chao_S",
    "chao_unseen",
    "chao_S_ci_low",
    "chao_S_ci_high",
)

CONVERGENCE_FIELDS = (
    "sample_size",
    "scope",
    "spacegroup",
    "energy_window_kj_mol",
    "energy_cutoff",
    "N",
    "S_obs",
    "f1",
    "f2",
    "good_turing_coverage",
    "good_turing_percent",
    "chao_S",
    "chao_unseen",
)


def _format_output_value(value: object) -> object:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.10g}"
    return value


def write_summary(rows: Sequence[SummaryRow], handle: TextIO) -> None:
    writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                key: _format_output_value(value)
                for key, value in asdict(row).items()
            }
        )


def write_convergence(
    rows: Sequence[ConvergenceRow], handle: TextIO
) -> None:
    writer = csv.DictWriter(handle, fieldnames=CONVERGENCE_FIELDS)
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                key: _format_output_value(value)
                for key, value in asdict(row).items()
            }
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate Good--Turing coverage and the Chao lower-bound "
            "estimator from a clustered mol-CSPy database."
        )
    )
    parser.add_argument(
        "databases",
        nargs="+",
        type=Path,
        help="clustered source mol-CSPy database(s)",
    )
    parser.add_argument(
        "--global-clusters",
        type=Path,
        help=(
            "reduced output database produced by clustering multiple source "
            "databases together"
        ),
    )
    parser.add_argument(
        "--windows",
        nargs="*",
        type=float,
        default=DEFAULT_WINDOWS,
        metavar="KJ_MOL",
        help="windows above the global minimum (default: 5 10 15)",
    )
    parser.add_argument(
        "--total-only",
        action="store_true",
        help="omit the additional per-space-group rows",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=1000,
        help="bootstrap resamples for confidence intervals (default: 1000; 0 disables)",
    )
    parser.add_argument(
        "--confidence-level",
        type=float,
        default=0.95,
        help="bootstrap confidence level (default: 0.95)",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=0,
        help="random seed for reproducible intervals (default: 0)",
    )
    parser.add_argument(
        "--allow-incomplete-clusters",
        action="store_true",
        help="treat valid structures absent from equivalent_to as singletons",
    )
    parser.add_argument(
        "--convergence-output",
        type=Path,
        help="write a trial-ordered completeness curve to this CSV",
    )
    parser.add_argument(
        "--convergence-points",
        type=int,
        default=20,
        help="number of checkpoints in the convergence curve (default: 20)",
    )
    parser.add_argument("-o", "--output", type=Path, help="write CSV here")
    return parser


def main(sys_args: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(sys_args)
    try:
        observations = combine_database_observations(
            args.databases,
            global_clusters=args.global_clusters,
            strict_assignments=not args.allow_incomplete_clusters,
        )
        rows = summarize(
            observations,
            windows=args.windows,
            by_spacegroup=not args.total_only,
            bootstrap_samples=args.bootstrap_samples,
            confidence_level=args.confidence_level,
            random_seed=args.random_seed,
        )
        if args.output is None:
            write_summary(rows, sys.stdout)
        else:
            with args.output.open("w", newline="", encoding="utf-8") as handle:
                write_summary(rows, handle)
        if args.convergence_output is not None:
            curve = convergence(
                observations,
                windows=args.windows,
                points=args.convergence_points,
                by_spacegroup=not args.total_only,
            )
            with args.convergence_output.open(
                "w", newline="", encoding="utf-8"
            ) as handle:
                write_convergence(curve, handle)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
