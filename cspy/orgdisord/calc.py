import json
import logging
import argparse
import numpy as np
import pandas as pd

from scipy.special import logsumexp
from math import lgamma
from pathlib import Path
from typing import Literal, Any
from collections.abc import Iterable, Mapping, Sequence

from cspy.db.datastore import CspDataStore
from cspy.util.constants import KB
from .disord_utils import query_rows

LOG = logging.getLogger(__name__)


def load_optimised_rows(input_db):
    db = CspDataStore(str(input_db))

    try:
        rows = query_rows(
            db,
            """
            select C.id, C.energy, C.density, T.metadata
            from crystal C
            join trial_structure T on T.id == C.id
            where T.valid == 1
              and C.energy is not null
            order by C.id
            """,
        )
    finally:
        db.close()

    structures = []

    for structure_id, energy, density, metadata_text in rows:
        metadata = json.loads(metadata_text) if metadata_text else {}

        components = metadata.get("components")
        ratio = metadata.get("ratio")
        multiplicity = metadata.get("multiplicity", 1)

        if not components:
            raise ValueError(
                f"Structure {structure_id} has no metadata['components']; "
                "cannot group by composition."
            )

        components = {
            str(component): int(count)
            for component, count in components.items()
        }

        n_sites = sum(components.values())

        if ratio is None:
            ratio = {
                component: count / n_sites
                for component, count in components.items()
            }

        ratio = {
            str(component): float(fraction)
            for component, fraction in ratio.items()
        }

        structures.append(
            {
                "id": structure_id,
                "energy": float(energy),
                "density": (
                    float(density)
                    if density is not None
                    else np.nan
                ),
                "components": components,
                "ratio": ratio,
                "multiplicity": int(multiplicity),
                "metadata": metadata,
            }
        )

    return structures


def parse_orientation_degeneracies(
    orientation_args: Sequence[tuple[str, str | float]] | None,
) -> dict[str, float]:
    """
    Parse repeated CLI args like:

        --orientation-degeneracy A1 2
        --orientation-degeneracy A2 1

    The factor is raised to the number of molecules/components of that type.
    """
    factors = {}

    if not orientation_args:
        return factors

    for component, factor in orientation_args:
        factor = float(factor)

        if factor <= 0:
            raise ValueError("Orientation degeneracy factors must be positive.")

        factors[str(component)] = factor

    return factors

def log_multinomial_from_counts(
    counts: Iterable[int],
) -> float:
    total = int(sum(counts))

    return lgamma(total + 1) - sum(lgamma(int(c) + 1) for c in counts)


def log_W_theoretical_from_components(
    components: Mapping[str, int],
    orientation_degeneracies: Mapping[str, float] | None = None,
) -> float:
    """
    Generic theoretical W for a composition.

    Positional part:
        N! / prod_i n_i!

    Optional orientation part:
        prod_i orientation_factor_i ** n_i
    """
    orientation_degeneracies = orientation_degeneracies or {}

    counts = [int(v) for v in components.values()]
    log_W = log_multinomial_from_counts(counts)

    for component, count in components.items():
        factor = orientation_degeneracies.get(component, 1.0)
        log_W += int(count) * np.log(float(factor))

    return float(log_W)

def log_W_ideal_mixing_from_components(
    components: Mapping[str, int],
    orientation_degeneracies: Mapping[str, float] | None = None,
) -> float:
    """
    Ideal-mixing target log multiplicity for one composition.

    Compositional part:
        ln W_ideal = -N * sum_i x_i ln(x_i)

    Optional orientational part:
        sum_i n_i ln(g_i)

    where g_i is the orientation degeneracy for component i.
    """
    orientation_degeneracies = orientation_degeneracies or {}

    counts = {
        str(component): int(count)
        for component, count in components.items()
    }

    n_sites = int(sum(counts.values()))

    if n_sites <= 0:
        raise ValueError("Cannot calculate ideal mixing entropy for zero sites.")

    log_W = 0.0

    # Ideal compositional mixing entropy.
    for component, count in counts.items():
        if count <= 0:
            continue

        x = count / n_sites
        log_W -= n_sites * x * np.log(x)

    # Optional orientation degeneracy.
    for component, count in counts.items():
        factor = orientation_degeneracies.get(component, 1.0)

        if factor <= 0:
            raise ValueError("Orientation degeneracy factors must be positive.")

        if count > 0 and factor != 1.0:
            log_W += count * np.log(float(factor))

    return float(log_W)


def calculate_ensemble_for_group(
    group: Sequence[Mapping[str, Any]],
    temperature: float,
    energy_input: str = "per_site",
    orientation_degeneracies: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """
    Calculate ensemble free energy for one composition.

    W is the theoretical number of possible configurations for the composition.

    W_sampled is the sum of orgdisord/merge multiplicities in the sampled set.
    """
    RT = KB * temperature
    beta = 1.0 / RT

    energies = np.array([row["energy"] for row in group], dtype=float)
    org_multiplicities = np.array(
        [row.get("multiplicity", 1) for row in group],
        dtype=float,
    )

    components = group[0]["components"]
    n_sites = sum(components.values())

    if energy_input == "per_site":
        energies_total = energies * n_sites
    elif energy_input == "total":
        energies_total = energies
    else:
        raise ValueError(f"Unknown energy_input: {energy_input}")

    e_min_total = np.min(energies_total)

    log_W_theoretical = log_W_theoretical_from_components(
        components,
        orientation_degeneracies=orientation_degeneracies,
    )
    
    log_W_ideal = log_W_ideal_mixing_from_components(
        components,
        orientation_degeneracies=orientation_degeneracies,
    )

    W_theoretical = float(np.exp((log_W_theoretical)))
    W_ideal = float(np.exp((log_W_ideal)))
    W_sampled = float(np.sum(org_multiplicities))

    if W_sampled <= 0:
        raise ValueError("W_sampled must be positive.")

    # Sampled/merged relative partition function.
    ln_Z_sample_relative = logsumexp(
        np.log(org_multiplicities) - beta * (energies_total - e_min_total)
    )

    # W_theoretical used previously
    # ln_scaling = log_W_theoretical - np.log(W_sampled)
    
    # Scale sampled/merged configurations to the ideal-mixing target W.
    ln_scaling = log_W_ideal - np.log(W_sampled)

    ln_Z_relative = ln_Z_sample_relative + ln_scaling
    Z_relative = float(np.exp((ln_Z_relative)))

    G_total = e_min_total - RT * ln_Z_relative

    if energy_input == "per_site":
        G = G_total / n_sites
        e_min = e_min_total / n_sites
    else:
        G = G_total
        e_min = e_min_total

    return {
        "G": float(G),
        "energy_min": float(e_min),
        "energy_mean": float(np.mean(energies)),
        "energy_median": float(np.median(energies)),
        "W": W_theoretical,
        "ln_W": log_W_theoretical,
        "W_ideal": W_ideal,
        "ln_W_ideal": float(log_W_ideal),
        "W_sampled": W_sampled,
        "sample_fraction": W_sampled / W_theoretical,
        "sample_fraction_ideal": W_sampled / W_ideal,
        "n_samples": int(len(group)),
        "Z_relative": Z_relative,
        "ln_Z_relative": float(ln_Z_relative),
        "ln_Z_sample_relative": float(ln_Z_sample_relative),
        "ln_scaling": float(ln_scaling),
        "energies": energies.tolist(),
        "orgdisord_multiplicities": org_multiplicities.astype(int).tolist(),
    }


def bootstrap_ensemble_ci(
    group: Sequence[Mapping[str, Any]],
    temperature: float,
    energy_input: str = "per_site",
    orientation_degeneracies: Mapping[str, float] | None = None,
    n_bootstrap: int = 1000,
    ci: float = 98,
    seed: int | None = 42,
) -> tuple[float, float]:
    if len(group) < 2 or n_bootstrap <= 0:
        return np.nan, np.nan

    rng = np.random.default_rng(seed)
    values = []

    n = len(group)

    for _ in range(n_bootstrap):
        indices = rng.choice(np.arange(n), size=n, replace=True)
        boot_group = [group[i] for i in indices]

        result = calculate_ensemble_for_group(
            boot_group,
            temperature=temperature,
            energy_input=energy_input,
            orientation_degeneracies=orientation_degeneracies,
        )
        values.append(result["G"])

    alpha = (100.0 - ci) / 2.0

    return (
        float(np.percentile(values, alpha)),
        float(np.percentile(values, 100.0 - alpha)),
    )


def parse_reference_energies(
    reference_energy_args: Sequence[tuple[str, str | float]] | None,
) -> dict[str, float]:

    refs = {}

    if not reference_energy_args:
        return refs

    for component, energy in reference_energy_args:
        refs[str(component)] = float(energy)

    return refs


def infer_reference_energies(
    results_df: pd.DataFrame,
    component_names: Sequence[str],
) -> dict[str, float]:
    """
    Infer pure-component reference energies from rows where one component ratio is 1.
    Uses G_ensemble for the pure row.
    """
    refs = {}

    for component in component_names:
        ratio_col = f"x_{component}"

        if ratio_col not in results_df.columns:
            continue

        mask = np.isclose(results_df[ratio_col].values.astype(float), 1.0, atol=1e-8)

        pure_rows = results_df.loc[mask]

        if pure_rows.empty:
            continue

        # If there are multiple pure rows, take the lowest ensemble free energy.
        refs[component] = float(pure_rows["G_ensemble"].min())

    return refs


def weighted_reference_sum(
    ratios: Mapping[str, float],
    reference_energies: Mapping[str, float],
) -> float:
    total = 0.0

    for component, x in ratios.items():
        if component not in reference_energies:
            return np.nan

        total += float(x) * float(reference_energies[component])

    return float(total)

def orientation_degeneracies_from_metadata(
    structures: Sequence[Mapping[str, Any]],
) -> dict[str, float]:
    found = []

    for structure in structures:
        metadata = structure.get("metadata", {})
        factors = metadata.get("orientation_degeneracies")

        if not factors:
            continue

        cleaned = {
            str(component): float(factor)
            for component, factor in factors.items()
        }

        found.append(cleaned)

    if not found:
        return {}

    first = found[0]

    for other in found[1:]:
        if other != first:
            raise ValueError(
                "Inconsistent orientation_degeneracies found in DB metadata: "
                f"{first} vs {other}"
            )

    return first


def calculate_ensemble_table(
    database: str | Path,
    output_csv: str | Path | None = None,
    temperature: float = 298.15,
    energy_input: Literal["per_site", "total"] = "per_site",
    reference_energies: Mapping[str, float] | None = None,
    orientation_degeneracies: Mapping[str, float] | None = None,
    n_bootstrap: int = 1000,
    ci: float = 98.0,
    seed: int | None = None,
) -> pd.DataFrame:
    """Calculate ensemble free energies grouped by composition.

    Parameters
    ----------
    database
        CSPy database containing optimised structures.
    output_csv
        Output CSV path. When omitted, ``<database>.ensemble.csv`` is used.
    temperature
        Temperature in kelvin.
    energy_input
        Whether stored energies are per site or total energies.
    reference_energies
        Optional pure-component reference energies.
    orientation_degeneracies
        Optional orientational degeneracy factors by component.
    n_bootstrap
        Number of bootstrap resamples.
    ci
        Bootstrap confidence interval percentage.
    seed
        Random seed used for bootstrap resampling.

    Returns
    -------
    pandas.DataFrame
        One row per composition with ensemble and reference free energies.
    """
    input_db = Path(database)

    reference_energies = dict(reference_energies or {})
    orientation_degeneracies = dict(
        orientation_degeneracies or {}
    )

    structures = load_optimised_rows(input_db)

    if not structures:
        raise ValueError(
            f"No valid optimised structures found in {input_db}"
        )

    component_names = sorted(
        {
            component
            for structure in structures
            for component in structure["components"]
        }
    )

    grouped: dict[
        tuple[tuple[str, int], ...],
        list[dict[str, Any]],
    ] = {}

    for structure in structures:
        composition_key = tuple(
            sorted(
                (
                    str(component),
                    int(count),
                )
                for component, count
                in structure["components"].items()
            )
        )

        grouped.setdefault(
            composition_key,
            [],
        ).append(structure)

    if not orientation_degeneracies:
        orientation_degeneracies = (
            orientation_degeneracies_from_metadata(
                structures
            )
        )

    LOG.info(
        "Orientation degeneracies used: %s",
        orientation_degeneracies,
    )

    records: list[dict[str, Any]] = []

    for composition_key, group in grouped.items():
        components = {
            component: int(count)
            for component, count in composition_key
        }

        n_sites = sum(components.values())

        if n_sites <= 0:
            raise ValueError(
                f"Invalid zero-site composition: {components}"
            )

        ratios = {
            component: (
                components.get(component, 0)
                / float(n_sites)
            )
            for component in component_names
        }

        ensemble = calculate_ensemble_for_group(
            group,
            temperature=temperature,
            energy_input=energy_input,
            orientation_degeneracies=(
                orientation_degeneracies
            ),
        )

        ci_low, ci_high = bootstrap_ensemble_ci(
            group,
            temperature=temperature,
            energy_input=energy_input,
            orientation_degeneracies=(
                orientation_degeneracies
            ),
            n_bootstrap=n_bootstrap,
            ci=ci,
            seed=seed,
        )

        record: dict[str, Any] = {
            "n_sites": n_sites,
            "W": ensemble["W"],
            "ln_W": ensemble["ln_W"],
            "W_ideal": ensemble["W_ideal"],
            "ln_W_ideal": ensemble["ln_W_ideal"],
            "W_sampled": ensemble["W_sampled"],
            "Z_relative": ensemble["Z_relative"],
            "ln_Z_relative": ensemble[
                "ln_Z_relative"
            ],
            "ln_Z_sample_relative": ensemble[
                "ln_Z_sample_relative"
            ],
            "ln_scaling": ensemble["ln_scaling"],
            "sample_fraction": ensemble[
                "sample_fraction"
            ],
            "sample_fraction_ideal": ensemble[
                "sample_fraction_ideal"
            ],
            "n_samples": ensemble["n_samples"],
            "energy_min": ensemble["energy_min"],
            "energy_mean": ensemble["energy_mean"],
            "energy_median": ensemble[
                "energy_median"
            ],
            "G_ensemble": ensemble["G"],
            "G_bootstrap_low": ci_low,
            "G_bootstrap_high": ci_high,
        }

        for component in component_names:
            record[f"n_{component}"] = (
                components.get(component, 0)
            )
            record[f"x_{component}"] = (
                ratios[component]
            )

        records.append(record)

    dataframe = pd.DataFrame(records)

    sort_columns = [
        f"x_{component}"
        for component in component_names
        if f"x_{component}" in dataframe.columns
    ]

    if sort_columns:
        dataframe = dataframe.sort_values(
            sort_columns
        ).reset_index(drop=True)

    if reference_energies:
        reference_source = "user"
    else:
        reference_energies = infer_reference_energies(
            dataframe,
            component_names,
        )
        reference_source = "auto"

    LOG.info(
        "Reference energies used: %s",
        reference_energies,
    )

    weighted_references: list[float] = []
    delta_values: list[float] = []
    delta_lows: list[float] = []
    delta_highs: list[float] = []

    for _, row in dataframe.iterrows():
        ratios = {
            component: float(
                row[f"x_{component}"]
            )
            for component in component_names
        }

        reference = weighted_reference_sum(
            ratios,
            reference_energies,
        )

        weighted_references.append(reference)

        if np.isnan(reference):
            delta_values.append(np.nan)
            delta_lows.append(np.nan)
            delta_highs.append(np.nan)
            continue

        delta_values.append(
            float(row["G_ensemble"]) - reference
        )
        delta_lows.append(
            float(row["G_bootstrap_low"]) - reference
        )
        delta_highs.append(
            float(row["G_bootstrap_high"]) - reference
        )

    dataframe["reference_source"] = (
        reference_source
    )
    dataframe["reference_weighted_sum"] = (
        weighted_references
    )
    dataframe["delta_G"] = delta_values
    dataframe["delta_G_bootstrap_low"] = (
        delta_lows
    )
    dataframe["delta_G_bootstrap_high"] = (
        delta_highs
    )
    dataframe["reference_energies_json"] = (
        json.dumps(
            reference_energies,
            sort_keys=True,
        )
    )

    if output_csv is None:
        output_path = input_db.with_name(
            f"{input_db.stem}.ensemble.csv"
        )
    else:
        output_path = Path(output_csv)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_csv(
        output_path,
        index=False,
    )

    LOG.info(
        "Wrote ensemble table to %s",
        output_path,
    )

    return dataframe


def run_cspy_disord_calc(
    args: argparse.Namespace,
) -> pd.DataFrame:
    """Run the ensemble calculation from parsed CLI arguments."""
    return calculate_ensemble_table(
        database=args.database,
        output_csv=args.output_csv,
        temperature=args.temperature,
        energy_input=args.energy_input,
        reference_energies=parse_reference_energies(
            args.reference_energy
        ),
        orientation_degeneracies=parse_orientation_degeneracies(
            args.orientation_degeneracy
        ),
        n_bootstrap=args.bootstrap,
        ci=args.ci,
        seed=args.seed,
    )

