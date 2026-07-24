import json
import logging
import math
import re
import tempfile
import time

from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from ase import Atoms
from ase.cell import Cell
from ase.io import write as ase_write

from orgdisord.enumerate import (
    OrderedfromDisordered,
    align_components_to_reference,
    component_centroids,
    connected_components_indices,
    make_whole_molecule,
)
from orgdisord.merge import merge_structures
from orgdisord.parse_cif_file import CifParser

from .database import write_cifs_to_cspy_db
from .disord_utils import (
    cif_tokens,
    clean_cif_value,
    find_atom_site_loop_bounds,
    find_cif_scalar_value,
    normalise_disorder_label,
    parse_cif_number,
    read_group_map_from_cif,
    read_orientation_degeneracies_from_cif,
    read_site_maps_from_cif,
)


LOG = logging.getLogger(__name__)


def _component_sort_key(component):
    component = str(component)
    match = re.fullmatch(r"A(\d+)", component)

    if match:
        return (0, int(match.group(1)))

    return (1, component)


def complete_counts(counts, total_sites, component_order):

    total_sites = int(total_sites)

    counts = {
        str(component): int(value)
        for component, value in (counts or {}).items()
    }

    unknown = sorted(set(counts).difference(component_order))

    if unknown:
        raise ValueError(
            f"Unknown components in composition: {unknown}. "
            f"Available components are {list(component_order)}."
        )

    missing = [
        component
        for component in component_order
        if component not in counts
    ]

    if missing:
        if len(component_order) == 2 and len(missing) == 1:
            counts[missing[0]] = total_sites - sum(counts.values())
        else:
            raise ValueError(
                f"Missing counts for {missing}. "
                f"Provide all counts for components {list(component_order)}."
            )

    counts = {
        component: int(counts.get(component, 0))
        for component in component_order
    }

    if sum(counts.values()) != total_sites:
        raise ValueError(
            f"Component counts must sum to total_sites={total_sites}. "
            f"Got {counts}, sum={sum(counts.values())}."
        )

    return counts


def counts_from_percentages(percentages, total_sites, component_order):
    total_sites = int(total_sites)

    percentages = {
        str(component): float(value)
        for component, value in (percentages or {}).items()
    }

    unknown = sorted(set(percentages).difference(component_order))

    if unknown:
        raise ValueError(
            f"Unknown components in percentage composition: {unknown}. "
            f"Available components are {list(component_order)}."
        )

    missing = [
        component
        for component in component_order
        if component not in percentages
    ]

    if missing:
        if len(component_order) == 2 and len(missing) == 1:
            supplied = next(iter(percentages))
            percentages[missing[0]] = 1.0 - percentages[supplied]
        else:
            raise ValueError(
                f"Missing percentage values for {missing}. "
                f"Provide all percentages for components "
                f"{list(component_order)}."
            )

    values = np.asarray(
        [percentages[component] for component in component_order],
        dtype=float,
    )

    if not np.isclose(values.sum(), 1.0, atol=0.01):
        raise ValueError(
            "Percentages must sum to 100%. "
            f"Got {100.0 * values.sum():.8g}% from {percentages}."
        )

    raw_counts = values * total_sites
    integer_counts = np.floor(raw_counts).astype(int)
    remaining = total_sites - int(integer_counts.sum())

    if remaining:
        remainders = raw_counts - integer_counts
        order = np.argsort(-remainders, kind="stable")
        integer_counts[order[:remaining]] += 1

    return {
        component: int(count)
        for component, count in zip(component_order, integer_counts)
    }


def read_ratio_counts_file(path):
    """
    Read a CSV composition schedule.

    Exact-count format::

        label,n_A1,n_A2,n_A3,n_samples
        equimolar,11,11,10,200
        binary_A1_A2,16,16,0,200

    Percentage format::

        label,pct_A1,pct_A2,pct_A3,n_samples
        equimolar,33.3,33.3,33.3,200
        binary_A1_A2,50,50,0,200

    In a binary system one component may be blank and is inferred as the
    remainder. For ternary systems every component must be supplied.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Composition schedule was not found: {path}"
        )

    dataframe = pd.read_csv(path)

    sample_names = (
        "n_samples",
        "samples",
        "maxiters",
        "max_configs",
    )
    sample_column = next(
        (name for name in sample_names if name in dataframe.columns),
        None,
    )

    label_column = "label" if "label" in dataframe.columns else None

    count_columns = [
        column
        for column in dataframe.columns
        if str(column).startswith("n_")
        and str(column) not in sample_names
    ]
    percentage_columns = [
        column
        for column in dataframe.columns
        if str(column).startswith("pct_")
    ]

    if not count_columns and not percentage_columns:
        raise ValueError(
            "The composition CSV has no composition columns. "
            "Use n_A1,n_A2,... or pct_A1,pct_A2,... columns."
        )

    records = []

    for row_index, row in dataframe.iterrows():
        has_counts = any(
            pd.notna(row[column])
            for column in count_columns
        )
        has_percentages = any(
            pd.notna(row[column])
            for column in percentage_columns
        )

        if has_counts and has_percentages:
            raise ValueError(
                f"CSV row {row_index} mixes exact counts and percentages. "
                "Use only one representation per row."
            )

        if not has_counts and not has_percentages:
            continue

        record = {
            "counts": None,
            "percentages": None,
            "n_samples": None,
            "label": None,
        }

        if has_counts:
            counts = {}

            for column in count_columns:
                if pd.isna(row[column]):
                    continue

                component = str(column).removeprefix("n_")
                value = int(row[column])

                if value < 0:
                    raise ValueError(
                        f"Negative count in row {row_index}: "
                        f"{column}={value}."
                    )

                counts[component] = value

            record["counts"] = counts

        else:
            percentages = {}

            for column in percentage_columns:
                if pd.isna(row[column]):
                    continue

                component = str(column).removeprefix("pct_")
                value = float(row[column])

                if value < 0.0:
                    raise ValueError(
                        f"Negative percentage in row {row_index}: "
                        f"{column}={value}."
                    )

                percentages[component] = value / 100.0

            record["percentages"] = percentages

        if sample_column is not None and pd.notna(row[sample_column]):
            n_samples = int(row[sample_column])

            if n_samples <= 0:
                raise ValueError(
                    f"n_samples must be positive in row {row_index}."
                )

            record["n_samples"] = n_samples

        if label_column is not None and pd.notna(row[label_column]):
            record["label"] = str(row[label_column]).strip()

        records.append(record)

    if not records:
        raise ValueError(
            f"No composition rows were read from {path}."
        )

    return records


def parse_orientation_degeneracies(args):

    factors = {}

    for component, factor in (
        getattr(args, "orientation_degeneracy", None) or []
    ):
        factor = float(factor)


        factors[str(component)] = factor

    return factors


def lnW_from_counts(counts, orientation_degeneracies=None):
    """
    Return the theoretical natural logarithm of configurational degeneracy.
    Positional contribution::
        W_pos = N! / product_i(n_i!)
    Orientational contribution::
        W_orient = product_i(g_i ** n_i)
    """
    orientation_degeneracies = orientation_degeneracies or {}

    total = int(sum(int(value) for value in counts.values()))

    ln_w = math.lgamma(total + 1)
    ln_w -= sum(
        math.lgamma(int(value) + 1)
        for value in counts.values()
    )

    for component, count in counts.items():
        factor = float(orientation_degeneracies.get(component, 1.0))

        if factor <= 0.0:
            raise ValueError(f"Orientation degeneracy for {component} must be positive.")

        ln_w += int(count) * math.log(factor)

    log10_w = ln_w / math.log(10.0)

    return float(ln_w), float(log10_w)


def choose_n_samples(
    record,
    args,
    ln_w,
    log10_w,
):
    """Choose the requested number of samples for one composition."""
    mode = getattr(
        args,
        "samples_from",
        "10lnW",
    )
    scale = float(
        getattr(
            args,
            "sample_scale",
            1.0,
        )
    )

    if scale <= 0.0:
        raise ValueError(
            "--sample-scale must be positive."
        )

    if mode == "csv":
        n_samples = record.get("n_samples")

        if n_samples is None:
            raise ValueError(
                "--samples-from csv requires n_samples "
                "for every composition row."
            )

        return int(n_samples)

    if mode == "lnW":
        value = scale * ln_w
    elif mode == "10lnW":
        value = 10.0 * scale * ln_w
    elif mode in {"log10W", "logW"}:
        value = scale * log10_w
    elif mode in {"10log10W", "10logW"}:
        value = 10.0 * scale * log10_w
    else:
        raise ValueError(
            f"Unknown samples-from mode: {mode!r}."
        )

    return max(
        1,
        int(math.ceil(value)),
    )


def build_composition_schedule(
    args,
    component_order,
    total_sites,
    orientation_degeneracies,
):

    ratio_counts_file = getattr(args, "ratio_counts_file", None)

    if ratio_counts_file is None:
        if len(component_order) != 1:
            raise ValueError(
                "Mixed-crystal enumeration requires --ratio-counts-file. "
                "Specify compositions using n_A1,n_A2,... or "
                "pct_A1,pct_A2,... columns."
            )

        component = component_order[0]
        records = [
            {
                "counts": {component: int(total_sites)},
                "percentages": None,
                "n_samples": int(getattr(args, "maxiters", 512)),
                "label": f"all_{component}",
            }
        ]
        LOG.info(
            "No composition CSV supplied for the single-component "
            "disorder model; sampling the all-%s composition.",
            component,
        )
    else:
        records = read_ratio_counts_file(ratio_counts_file)
    converted = []

    for record in records:
        if record["percentages"] is not None:
            counts = counts_from_percentages(
                record["percentages"],
                total_sites=total_sites,
                component_order=component_order,
            )
        else:
            counts = complete_counts(
                record["counts"],
                total_sites=total_sites,
                component_order=component_order,
            )

        converted.append(
            {
                "counts": counts,
                "n_samples": record.get("n_samples"),
                "label": record.get("label"),
            }
        )

    if getattr(args, "include_pure_components", True):
        for component in component_order:
            counts = {
                name: 0
                for name in component_order
            }
            counts[component] = total_sites

            converted.append(
                {
                    "counts": counts,
                    "n_samples": 1,
                    "label": f"pure_{component}",
                }
            )

    unique_records = []
    seen = set()

    for record in converted:
        key = tuple(
            int(record["counts"][component])
            for component in component_order
        )

        if key in seen:
            continue

        seen.add(key)

        ln_w, log10_w = lnW_from_counts(
            record["counts"],
            orientation_degeneracies=orientation_degeneracies,
        )

        non_zero_components = sum(
            int(value) > 0
            for value in record["counts"].values()
        )

        if non_zero_components == 1 and ln_w <= 1.0e-12:
            n_samples = 1
        else:
            n_samples = choose_n_samples(
                record,
                args=args,
                ln_w=ln_w,
                log10_w=log10_w
            )

            maxiters = getattr(args, "maxiters", None)

            if maxiters is not None:
                maxiters = int(maxiters)

                if maxiters <= 0:
                    raise ValueError(
                        "--maxiters must be positive."
                    )

                if n_samples > maxiters:
                    LOG.warning(
                        "Capping requested samples for %s from %d to "
                        "--maxiters=%d.",
                        record.get("label") or record["counts"],
                        n_samples,
                        maxiters,
                    )
                    n_samples = maxiters

        label = record.get("label")

        if not label:
            label = "_".join(
                f"{component}_{record['counts'][component]}"
                for component in component_order
            )

        unique_records.append(
            {
                "counts": dict(record["counts"]),
                "ratio": {
                    component: (
                        int(record["counts"][component])
                        / float(total_sites)
                    )
                    for component in component_order
                },
                "n_samples": int(n_samples),
                "lnW": float(ln_w),
                "log10_w": float(log10_w),
                "label": label,
            }
        )

    return unique_records


def read_prepared_p1_cif(cif_file):

    cif_file = Path(cif_file)
    lines = cif_file.read_text(
        errors="replace"
    ).splitlines()

    space_group_number = find_cif_scalar_value(
        lines,
        (
            "_space_group_it_number",
            "_symmetry_int_tables_number",
        ),
    )

    if space_group_number is not None:
        try:
            number = int(round(parse_cif_number(space_group_number)))
        except ValueError:
            number = None

        if number not in (None, 1):
            raise ValueError(
                "Direct enumeration requires a P1 prepared CIF. "
            )

    cell_parameters = [
        parse_cif_number(
            find_cif_scalar_value(lines, (name,))
        )
        for name in (
            "_cell_length_a",
            "_cell_length_b",
            "_cell_length_c",
            "_cell_angle_alpha",
            "_cell_angle_beta",
            "_cell_angle_gamma",
        )
    ]
    cell = np.asarray(
        Cell.fromcellpar(cell_parameters),
        dtype=float,
    )

    _, headers, row_start, row_end = find_atom_site_loop_bounds(
        lines,
        stop_prefixes=("# cspy_disord_",),
    )
    header_index = {
        header: index
        for index, header in enumerate(headers)
    }
    required = {
        "_atom_site_label",
        "_atom_site_type_symbol",
        "_atom_site_fract_x",
        "_atom_site_fract_y",
        "_atom_site_fract_z",
        "_atom_site_disorder_assembly",
        "_atom_site_disorder_group",
    }
    missing = sorted(required.difference(headers))

    if missing:
        raise ValueError(
            "Prepared CIF atom loop is missing columns: "
            + ", ".join(missing)
        )

    rows = []

    for raw_line in lines[row_start:row_end]:
        stripped = raw_line.strip()

        if not stripped or stripped.startswith("#"):
            continue

        tokens = cif_tokens(stripped)

        if len(tokens) != len(headers):
            raise ValueError(
                "Prepared CIF atom row has a different number of values "
                f"from the headers ({len(tokens)} vs {len(headers)}): "
                f"{raw_line}"
            )

        def value(header, default=None):
            if header not in header_index:
                return default
            return tokens[header_index[header]]

        rows.append(
            {
                "label": clean_cif_value(
                    value("_atom_site_label")
                ),
                "symbol": clean_cif_value(
                    value("_atom_site_type_symbol")
                ),
                "fract_x": parse_cif_number(
                    value("_atom_site_fract_x")
                ),
                "fract_y": parse_cif_number(
                    value("_atom_site_fract_y")
                ),
                "fract_z": parse_cif_number(
                    value("_atom_site_fract_z")
                ),
                "occupancy": (
                    parse_cif_number(value("_atom_site_occupancy"))
                    if "_atom_site_occupancy" in header_index
                    else 1.0
                ),
                "assembly": normalise_disorder_label(
                    value("_atom_site_disorder_assembly")
                ) if clean_cif_value(
                    value("_atom_site_disorder_assembly")
                ) else "",
                "group": normalise_disorder_label(
                    value("_atom_site_disorder_group")
                ) if clean_cif_value(
                    value("_atom_site_disorder_group")
                ) else "",
            }
        )

    labels = [
        row["label"]
        for row in rows
    ]

    if len(labels) != len(set(labels)):
        raise ValueError(
            "Prepared CIF atom labels must be unique."
        )

    return {
        "cell": cell,
        "rows": rows,
        "lines": lines,
    }


def _atoms_from_prepared_rows(rows, cell):

    rows = list(rows)
    
    LOG.debug("Building ASE Atoms from %d prepared rows.", len(rows))

    if not rows:
        return Atoms(
            cell=np.asarray(cell, dtype=float),
            pbc=True,
        )

    atoms = Atoms(
        symbols=[
            row["symbol"]
            for row in rows
        ],
        scaled_positions=np.asarray(
            [
                [
                    row["fract_x"],
                    row["fract_y"],
                    row["fract_z"],
                ]
                for row in rows
            ],
            dtype=float,
        ).reshape((-1, 3)),
        cell=np.asarray(cell, dtype=float),
        pbc=True,
    )

    max_length = max(
        len(row["label"])
        for row in rows
    )

    atoms.set_array(
        "labels",
        np.asarray(
            [row["label"] for row in rows],
            dtype=f"<U{max(1, max_length)}",
        ),
    )

    return atoms

def _align_state_to_reference(state, reference):
    """
    Make one complete molecular state whole and translate its connected
    component(s) by integer lattice vectors so they occupy the same periodic
    images as the reference state.

    Alternative mixed-crystal components may have different atom counts and
    chemical formulae. Alignment is therefore based on connected-component
    centroids rather than atom-by-atom correspondence.
    """
    cutoff_scale = 1.5

    reference = make_whole_molecule(
        reference,
        cutoff_scale=cutoff_scale,
        preserve_pbc=False,
    )
    state = make_whole_molecule(
        state,
        cutoff_scale=cutoff_scale,
        preserve_pbc=False,
    )

    reference_components = connected_components_indices(
        reference,
        cutoff_scale=cutoff_scale,
    )
    state_components = connected_components_indices(
        state,
        cutoff_scale=cutoff_scale,
    )

    if len(reference_components) != len(state_components):
        raise ValueError(
            "Alternative states contain different numbers of connected "
            "molecular components: "
            f"reference={len(reference_components)}, "
            f"state={len(state_components)}."
        )

    reference_centroids = component_centroids(
        reference,
        reference_components,
        mass_weighted=False,
    )

    aligned = align_components_to_reference(
        state,
        reference_centroids=reference_centroids,
        cell=np.asarray(reference.get_cell(), dtype=float),
        cutoff_scale=cutoff_scale,
        max_offset=1,
        mass_weighted=False,
        use_labels=False,
    )

    aligned.set_cell(reference.get_cell())
    aligned.set_pbc(False)

    return aligned


def _complete_and_align_site_states(raw_states, cell, site):
    if not raw_states:
        raise ValueError(
            f"Prepared physical site {site} contains no states."
        )

    reference = make_whole_molecule(
        raw_states[0]["atoms"],
        cutoff_scale=1.5,
        preserve_pbc=False,
    )
    reference.set_cell(cell)
    reference.set_pbc(False)

    states = []
    states_by_component = defaultdict(list)

    for raw_state in raw_states:
        aligned = _align_state_to_reference(
            raw_state["atoms"],
            reference=reference,
        )
        state = dict(raw_state)
        state["atoms"] = aligned
        state_index = len(states)
        states.append(state)
        states_by_component[
            state["component"]
        ].append(state_index)

    return {
        "assembly": str(site),
        "site": str(site),
        "states": states,
        "states_by_component": {
            component: tuple(indices)
            for component, indices
            in states_by_component.items()
        },
    }


def prepare_site_states(
    prepared,
    group_map,
    site_maps,
):
    """
    Build complete molecular alternatives for every physical prepared site.
    """
    cell = prepared["cell"]
    rows = prepared["rows"]
    rows_by_label = {
        row["label"]: row
        for row in rows
    }
    rows_by_assembly_group = defaultdict(list)

    for row in rows:
        if row["assembly"] and row["group"]:
            rows_by_assembly_group[
                (row["assembly"], row["group"])
            ].append(row)

    site_specs = []
    consumed_common_labels = set()
    consumed_disorder_keys = set()

    for site, metadata in sorted(site_maps.items()):
        common_labels = [
            str(label)
            for label in metadata.get("common_labels", [])
        ]
        missing_common = [
            label
            for label in common_labels
            if label not in rows_by_label
        ]

        if missing_common:
            raise ValueError(
                f"Prepared site {site} references missing common atom labels "
                f"{missing_common}."
            )

        common_rows = [
            rows_by_label[label]
            for label in common_labels
        ]
        consumed_common_labels.update(common_labels)
        raw_states = []

        for state_position, state_metadata in enumerate(
            metadata.get("allowed_states", []),
            start=1,
        ):
            choices = {
                normalise_disorder_label(assembly):
                normalise_disorder_label(group)
                for assembly, group
                in state_metadata.get("choices", {}).items()
            }
            selected_rows = list(common_rows)

            for assembly_description in metadata.get(
                "assemblies",
                [],
            ):
                assembly = normalise_disorder_label(
                    assembly_description["assembly"]
                )

                if assembly not in choices:
                    raise ValueError(
                        f"Allowed state {state_position} for site {site} "
                        f"does not choose assembly {assembly}."
                    )

                group = choices[assembly]
                key = (assembly, group)
                fragment_rows = rows_by_assembly_group.get(
                    key,
                    [],
                )

                if not fragment_rows:
                    raise ValueError(
                        f"Prepared site {site} state {state_position} "
                        f"references empty fragment {assembly}:{group}."
                    )

                selected_rows.extend(fragment_rows)
                consumed_disorder_keys.add(key)

            atoms = _atoms_from_prepared_rows(
                selected_rows,
                cell,
            )
            raw_states.append(
                {
                    "index": state_position - 1,
                    "group": str(
                        state_metadata.get(
                            "state_index",
                            state_position,
                        )
                    ),
                    "component": str(
                        state_metadata.get(
                            "component",
                            metadata["component"],
                        )
                    ),
                    "mapping_index": int(
                        state_metadata.get(
                            "state_index",
                            state_position,
                        )
                    ),
                    "n_mappings": int(
                        metadata.get(
                            "state_count",
                            len(metadata.get(
                                "allowed_states",
                                [],
                            )),
                        )
                    ),
                    "assembly_choices": choices,
                    "original_joint_state": (
                        state_metadata.get(
                            "original_choices"
                        )
                    ),
                    "original_joint_occupancy": (
                        state_metadata.get(
                            "original_occupancy"
                        )
                    ),
                    "atoms": atoms,
                }
            )

        site_specs.append(
            _complete_and_align_site_states(
                raw_states,
                cell,
                site,
            )
        )

    # Standard full-molecule prepared sites, used by mixed-crystal overlays.
    assemblies_with_site_metadata = {
        normalise_disorder_label(
            assembly_description["assembly"]
        )
        for metadata in site_maps.values()
        for assembly_description in metadata.get(
            "assemblies",
            [],
        )
    }
    full_state_assemblies = sorted(
        {
            assembly
            for assembly, _group in group_map
            if assembly not in assemblies_with_site_metadata
        }
    )

    for assembly in full_state_assemblies:
        groups = sorted(
            {
                group
                for mapped_assembly, group in group_map
                if mapped_assembly == assembly
            },
            key=lambda value: (
                int(value)
                if str(value).isdigit()
                else str(value)
            ),
        )
        raw_states = []

        for state_index, group in enumerate(groups):
            key = (assembly, group)
            state_rows = rows_by_assembly_group.get(
                key,
                [],
            )

            if not state_rows:
                raise ValueError(
                    f"Prepared full state {assembly}:{group} has no atoms."
                )

            mapped = group_map[key]
            consumed_disorder_keys.add(key)
            raw_states.append(
                {
                    "index": state_index,
                    "group": group,
                    "component": mapped["component"],
                    "mapping_index": mapped["mapping_index"],
                    "n_mappings": mapped["n_mappings"],
                    "original_joint_state": mapped.get(
                        "original_joint_state"
                    ),
                    "original_joint_occupancy": mapped.get(
                        "original_joint_occupancy"
                    ),
                    "assembly_choices": {
                        assembly: group,
                    },
                    "atoms": _atoms_from_prepared_rows(
                        state_rows,
                        cell,
                    ),
                }
            )

        site_specs.append(
            _complete_and_align_site_states(
                raw_states,
                cell,
                assembly,
            )
        )

    all_disorder_keys = set(rows_by_assembly_group)
    unused_disorder = sorted(
        all_disorder_keys.difference(
            consumed_disorder_keys
        )
    )

    if unused_disorder:
        raise ValueError(
            "Prepared CIF contains disorder fragments not represented by "
            f"group/site metadata: {unused_disorder}."
        )

    global_ordered_rows = [
        row
        for row in rows
        if not row["assembly"]
        and row["label"] not in consumed_common_labels
    ]
    ordered_atoms = _atoms_from_prepared_rows(
        global_ordered_rows,
        cell,
    )
    ordered_atoms.set_pbc(False)

    if not site_specs:
        raise ValueError(
            "No physical disorder sites were found in the prepared CIF."
        )

    return site_specs, ordered_atoms


def _slot_signature(slot):
    return tuple(
        sorted(
            (
                component,
                len(state_indices),
            )
            for component, state_indices
            in slot["states_by_component"].items()
        )
    )


def _multinomial_count(counts):
    total = int(sum(counts))
    result = math.factorial(total)

    for count in counts:
        result //= math.factorial(int(count))

    return result


def _uniform_slot_configuration_count(
    target_counts,
    component_order,
    signature,
):
    multiplicity_by_component = dict(signature)

    total = _multinomial_count(
        [
            target_counts[component]
            for component in component_order
        ]
    )

    for component in component_order:
        total *= (
            int(multiplicity_by_component.get(component, 0))
            ** int(target_counts[component])
        )

    return total


def _sample_uniform_slots(
    slots,
    target_counts,
    component_order,
    rng,
):
    components = []

    for component in component_order:
        components.extend(
            [component] * int(target_counts[component])
        )

    rng.shuffle(components)

    choices = []

    for slot, component in zip(slots, components):
        state_indices = slot["states_by_component"].get(
            component,
            (),
        )

        if not state_indices:
            raise RuntimeError(
                "Internal sampling error: a supposedly uniform site "
                f"does not allow component {component}."
            )

        choices.append(
            int(rng.choice(state_indices))
        )

    return tuple(choices)


def _build_general_completion_counter(
    slots,
    target_counts,
    component_order,
):
    component_index = {
        component: index
        for index, component in enumerate(component_order)
    }

    @lru_cache(maxsize=None)
    def count_completions(slot_index, remaining):
        if slot_index == len(slots):
            return int(all(value == 0 for value in remaining))

        if sum(remaining) != len(slots) - slot_index:
            return 0

        slot = slots[slot_index]
        total = 0

        for component, state_indices in (
            slot["states_by_component"].items()
        ):
            index = component_index.get(component)

            if index is None or remaining[index] <= 0:
                continue

            next_remaining = list(remaining)
            next_remaining[index] -= 1

            total += (
                len(state_indices)
                * count_completions(
                    slot_index + 1,
                    tuple(next_remaining),
                )
            )

        return total

    initial_remaining = tuple(
        int(target_counts[component])
        for component in component_order
    )

    return count_completions, initial_remaining, component_index


def _sample_general_slots(
    slots,
    target_counts,
    component_order,
    rng,
    completion_counter,
    initial_remaining,
    component_index,
):
    remaining = list(initial_remaining)
    choices = []

    for slot_index, slot in enumerate(slots):
        options = []
        option_weights = []

        for component, state_indices in (
            slot["states_by_component"].items()
        ):
            index = component_index.get(component)

            if index is None or remaining[index] <= 0:
                continue

            next_remaining = list(remaining)
            next_remaining[index] -= 1

            completions = completion_counter(
                slot_index + 1,
                tuple(next_remaining),
            )

            if completions <= 0:
                continue

            weight = len(state_indices) * completions
            options.append(
                (
                    component,
                    index,
                    state_indices,
                )
            )
            option_weights.append(weight)

        if not options:
            raise RuntimeError(
                "No feasible state remained while sampling an exact "
                "composition. This indicates an internal DP inconsistency."
            )

        weight_sum = sum(option_weights)
        draw = int(rng.integers(weight_sum))
        cumulative = 0
        selected = None

        for option, weight in zip(
            options,
            option_weights,
        ):
            cumulative += weight

            if draw < cumulative:
                selected = option
                break

        component, index, state_indices = selected
        remaining[index] -= 1
        choices.append(
            int(rng.choice(state_indices))
        )

    if any(remaining):
        raise RuntimeError(
            "Exact-composition sampling ended with non-zero remaining "
            f"counts: {remaining}."
        )

    return tuple(choices)


def sample_unique_state_choices(
    site_specs,
    supercell,
    target_counts,
    component_order,
    n_samples,
    seed=None,
):
    na, nb, nc = (
        int(value)
        for value in supercell
    )
    number_of_cells = na * nb * nc

    if number_of_cells <= 0:
        raise ValueError(
            f"Invalid supercell dimensions: {supercell}."
        )

    slots = [
        site_spec
        for _cell_index in range(number_of_cells)
        for site_spec in site_specs
    ]

    if sum(target_counts.values()) != len(slots):
        raise ValueError(
            "Requested component counts do not match the number of "
            f"available molecular sites: counts={target_counts}, "
            f"sites={len(slots)}."
        )

    signatures = {
        _slot_signature(slot)
        for slot in slots
    }
    uniform_slots = len(signatures) == 1
    rng = np.random.default_rng(seed)

    if uniform_slots:
        signature = next(iter(signatures))

        for component in component_order:
            if (
                target_counts[component] > 0
                and component not in dict(signature)
            ):
                raise ValueError(
                    f"Component {component} is requested but is not allowed "
                    "at the prepared molecular sites."
                )

        total_configurations = (
            _uniform_slot_configuration_count(
                target_counts,
                component_order,
                signature,
            )
        )

        def draw_one():
            return _sample_uniform_slots(
                slots,
                target_counts,
                component_order,
                rng,
            )

    else:
        (
            completion_counter,
            initial_remaining,
            component_index,
        ) = _build_general_completion_counter(
            slots,
            target_counts,
            component_order,
        )
        total_configurations = completion_counter(
            0,
            initial_remaining,
        )

        def draw_one():
            return _sample_general_slots(
                slots,
                target_counts,
                component_order,
                rng,
                completion_counter,
                initial_remaining,
                component_index,
            )

    if total_configurations <= 0:
        allowed = [
            {
                "assembly": slot["assembly"],
                "components": sorted(
                    slot["states_by_component"]
                ),
            }
            for slot in slots[: len(site_specs)]
        ]

        raise ValueError(
            "The requested composition is not feasible for the prepared "
            f"site restrictions: counts={target_counts}, "
            f"primitive-site options={allowed}."
        )

    if int(n_samples) > total_configurations:
        LOG.warning(
            "Requested %d samples, but only %d distinct assignments exist; "
            "sampling all available assignments.",
            n_samples,
            total_configurations,
        )

        n_samples = int(total_configurations)

    selected = set()
    max_attempts = max(
        1000,
        200 * int(n_samples),
    )

    for _ in range(max_attempts):
        selected.add(draw_one())

        if len(selected) == int(n_samples):
            break

    if len(selected) != int(n_samples):
        raise RuntimeError(
            f"Generated only {len(selected)} of {n_samples} requested unique "
            "state assignments before the rejection-sampling limit. "
            "Reduce n_samples or supply a different random seed."
        )

    return sorted(selected), int(total_configurations)


def describe_state_choices(
    site_specs,
    supercell,
    flat_state_choices,
):
    """
    The ordering matches the sampler: supercell position first (c fastest,
    then b, then a), followed by prepared site order within each cell.
    """
    number_of_sites = len(site_specs)
    number_of_cells = int(np.prod(supercell))
    expected = number_of_sites * number_of_cells

    if len(flat_state_choices) != expected:
        raise ValueError(
            f"Expected {expected} state choices, got "
            f"{len(flat_state_choices)}."
        )

    described = []

    for cell_index in range(number_of_cells):
        for site_index, site_spec in enumerate(site_specs):
            flat_index = cell_index * number_of_sites + site_index
            state_index = int(flat_state_choices[flat_index])
            state = site_spec["states"][state_index]
            described.append(
                {
                    "cell_index": int(cell_index),
                    "prepared_site_index": int(site_index),
                    "assembly": site_spec["assembly"],
                    "physical_site": site_spec.get(
                        "site",
                        site_spec["assembly"],
                    ),
                    "state_index": state_index,
                    "group": state["group"],
                    "assembly_choices": state.get(
                        "assembly_choices"
                    ),
                    "component": state["component"],
                    "mapping_index": int(state["mapping_index"]),
                    "n_mappings": int(state.get("n_mappings", 1)),
                    "original_joint_state": state.get(
                        "original_joint_state"
                    ),
                    "original_joint_occupancy": state.get(
                        "original_joint_occupancy"
                    ),
                }
            )

    return described


def build_primitive_image(
    cell,
    ordered_atoms,
    site_specs,
    site_state_choices,
):

    if len(site_state_choices) != len(site_specs):
        raise ValueError(
            "Expected one state choice for each physical molecular site: "
            f"expected {len(site_specs)}, got {len(site_state_choices)}."
        )

    image = ordered_atoms.copy()
    image.set_cell(cell)
    image.set_pbc(False)

    for site_spec, state_choice in zip(
        site_specs,
        site_state_choices,
    ):
        state_choice = int(state_choice)

        if (
            state_choice < 0
            or state_choice >= len(site_spec["states"])
        ):
            raise IndexError(
                f"Invalid state index {state_choice} for physical site "
                f"{site_spec['site']}."
            )

        image.extend(
            site_spec["states"][
                state_choice
            ]["atoms"]
        )

    image.set_cell(cell)
    image.set_pbc(False)

    return image


def build_sample_structure(
    enumerator,
    cell,
    ordered_atoms,
    site_specs,
    flat_state_choices,
    primitive_cache,
):

    number_of_sites = len(site_specs)
    number_of_cells = int(
        np.prod(enumerator.supercell)
    )
    expected = number_of_sites * number_of_cells

    if len(flat_state_choices) != expected:
        raise ValueError(
            f"Expected {expected} state choices, got "
            f"{len(flat_state_choices)}."
        )

    primitive_indices = []

    for cell_index in range(number_of_cells):
        start = cell_index * number_of_sites
        stop = start + number_of_sites
        cell_choices = tuple(
            int(value)
            for value in flat_state_choices[start:stop]
        )

        primitive_index = primitive_cache.get(
            cell_choices
        )

        if primitive_index is None:
            primitive_image = build_primitive_image(
                cell,
                ordered_atoms,
                site_specs,
                cell_choices,
            )
            primitive_index = len(enumerator.images)
            enumerator.images.append(
                primitive_image
            )
            primitive_cache[
                cell_choices
            ] = primitive_index

        primitive_indices.append(
            primitive_index
        )

    atoms = enumerator.build_supercell_from_primitive_indices(
        primitive_indices
    )

    return atoms, primitive_indices


def _oxidation_states_from_args(args):
    oxidation_states = {}

    for species, oxidation_state in (
        getattr(args, "ox", None) or []
    ):
        oxidation_states[str(species)] = int(
            oxidation_state
        )

    return oxidation_states


def merge_generated_structures(
    structures,
    metadata,
    merge_symops,
    args,
):

    if not getattr(args, "merge", False):
        return structures, metadata

    if not structures:
        return structures, metadata

    groups, group_indices = merge_structures(
        structures,
        algo=getattr(args, "algo", "symm"),
        symops=merge_symops,
        use_disordered_only=getattr(
            args,
            "use_disordered_only",
            False,
        ),
        symprec=float(getattr(args, "symprec", 1.0e-4)),
        oxidation_states=_oxidation_states_from_args(args),
        return_group_indices=True,
        quiet=getattr(args, "quiet", False),
        check_species=not getattr(
            args,
            "ignore_species",
            False,
        ),
    )

    merged_structures = []
    merged_metadata = []

    for group, indices in zip(groups, group_indices):
        representative = group[0]
        multiplicity = int(group[1])
        source_indices = [
            int(index)
            for index in indices
        ]

        item = dict(metadata[source_indices[0]])
        item["multiplicity"] = multiplicity
        item["merged_sample_indices"] = source_indices

        merged_structures.append(representative)
        merged_metadata.append(item)

    LOG.info(
        "Merged %d sampled structures into %d unique structures.",
        len(structures),
        len(merged_structures),
    )

    return merged_structures, merged_metadata


def _safe_slug(value):
    value = re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        str(value).strip(),
    )
    value = value.strip("._")

    return value or "composition"


def write_p1_cif_preserving_molecules(path, atoms):
    path = Path(path)
    cell = np.asarray(atoms.get_cell(), dtype=float)

    if abs(np.linalg.det(cell)) < 1.0e-12:
        raise ValueError(
            f"Cannot write CIF with singular cell: {path}."
        )

    fractional = atoms.get_scaled_positions(wrap=False)
    a, b, c, alpha, beta, gamma = atoms.cell.cellpar()

    element_counts = defaultdict(int)
    labels = []

    for symbol in atoms.get_chemical_symbols():
        element_counts[symbol] += 1
        labels.append(
            f"{symbol}{element_counts[symbol]}"
        )

    with path.open("w") as handle:
        handle.write(
            f"data_{_safe_slug(path.stem)}\n"
        )
        handle.write(f"_cell_length_a {a:.12g}\n")
        handle.write(f"_cell_length_b {b:.12g}\n")
        handle.write(f"_cell_length_c {c:.12g}\n")
        handle.write(
            f"_cell_angle_alpha {alpha:.12g}\n"
        )
        handle.write(
            f"_cell_angle_beta {beta:.12g}\n"
        )
        handle.write(
            f"_cell_angle_gamma {gamma:.12g}\n"
        )
        handle.write(
            "_space_group_name_H-M_alt 'P 1'\n"
        )
        handle.write("_space_group_IT_number 1\n")
        handle.write(
            "_symmetry_space_group_name_H-M 'P 1'\n"
        )
        handle.write(
            "_symmetry_Int_Tables_number 1\n"
        )
        handle.write("loop_\n")
        handle.write(
            "_symmetry_equiv_pos_as_xyz\n"
        )
        handle.write("'x, y, z'\n")
        handle.write("loop_\n")
        handle.write("_atom_site_label\n")
        handle.write("_atom_site_type_symbol\n")
        handle.write("_atom_site_fract_x\n")
        handle.write("_atom_site_fract_y\n")
        handle.write("_atom_site_fract_z\n")
        handle.write("_atom_site_occupancy\n")

        for label, symbol, position in zip(
            labels,
            atoms.get_chemical_symbols(),
            fractional,
        ):
            handle.write(
                f"{label} {symbol} "
                f"{position[0]:.12g} "
                f"{position[1]:.12g} "
                f"{position[2]:.12g} 1.0\n"
            )


def _prepare_results_directory(prefix):
    prefix_path = Path(prefix)
    output_parent = (
        prefix_path.parent
        if str(prefix_path.parent) != "."
        else Path.cwd()
    )
    output_parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_directory = (
        output_parent
        / f"{prefix_path.name}-results"
    )

    if results_directory.exists():
        timestamp = time.strftime(
            "%Y%m%d-%H%M%S"
        )
        backup = results_directory.with_name(
            f"{results_directory.name}-backup-{timestamp}"
        )

        if backup.exists():
            raise FileExistsError(
                f"Backup directory already exists: {backup}."
            )

        results_directory.rename(backup)
        LOG.warning(
            "Moved existing results directory to %s.",
            backup,
        )

    results_directory.mkdir(
        parents=True,
        exist_ok=False,
    )

    return results_directory


def _write_primitive_debug_images(
    enumerator,
    prefix,
):
    primitive_directory = Path(
        f"{prefix}-primitive-cells"
    )
    primitive_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    for index, image in enumerate(enumerator.images):
        path = (
            primitive_directory
            / f"primitive_config_{index:05d}.xyz"
        )
        ase_write(
            path,
            image,
            format="extxyz",
        )

    LOG.info(
        "Wrote %d used primitive images to %s.",
        len(enumerator.images),
        primitive_directory,
    )
    

def prepare_output_database(path, overwrite=False):

    output_db = Path(path)
    output_db.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    database_files = [
        output_db,
        Path(f"{output_db}-wal"),
        Path(f"{output_db}-shm"),
    ]

    existing_files = [
        path
        for path in database_files
        if path.exists()
    ]

    if existing_files and not overwrite:
        raise FileExistsError(
            f"Output database already exists: {output_db}. "
            "Use --overwrite to replace it."
        )

    for suffix in ("", "-wal", "-shm"):
        Path(f"{output_db}{suffix}").unlink(
            missing_ok=True
        )

    return output_db


def run_orgdisord_enumerate(args):

    files = [
        Path(filename)
        for filename in args.files
    ]

    if len(files) != 1:
        raise ValueError(
            "Direct CSPy enumeration expects one prepared CIF. "
            f"Received {len(files)} files."
        )

    cif_file = files[0]

    if not cif_file.exists():
        raise FileNotFoundError(
            f"Input CIF was not found: {cif_file}."
        )

    if cif_file.suffix.lower() != ".cif":
        raise ValueError(
            "Direct CSPy enumeration requires a prepared CIF input."
        )

    if getattr(args, "correlated_assemblies", False):
        raise ValueError(
            "Do not pass the legacy orgdisord --correlated-assemblies flag."
        )

    if args.output_db is None:
        args.output_db = Path(
            f"{args.prefix}.db"
        )
        LOG.info(
            "No --output-db supplied; using %s.",
            args.output_db,
        )

    args.output_db = prepare_output_database(
        args.output_db,
        overwrite=getattr(args, "overwrite", False),
    )

    group_map = read_group_map_from_cif(cif_file)
    
    site_maps = read_site_maps_from_cif(
        cif_file
    )
    
    component_names = {
        str(row["component"])
        for row in group_map.values()
    }

    for site_metadata in site_maps.values():
        for state in site_metadata.get(
            "allowed_states",
            [],
        ):
            component = state.get("component")

            if component is not None:
                component_names.add(str(component))

    component_order = tuple(
        sorted(
            component_names,
            key=_component_sort_key,
        )
    )

    orientation_degeneracies = (
        read_orientation_degeneracies_from_cif(
            cif_file
        )
    )
    orientation_degeneracies.update(
        parse_orientation_degeneracies(args)
    )

    LOG.info(
        "Components: %s.",
        ", ".join(component_order),
    )
    LOG.info(
        "Orientation degeneracies: %s.",
        orientation_degeneracies,
    )

    prepared = read_prepared_p1_cif(
        cif_file
    )
    site_specs, ordered_atoms = prepare_site_states(
        prepared,
        group_map,
        site_maps,
    )

    enumerator = object.__new__(
        OrderedfromDisordered
    )
    enumerator.cell = prepared["cell"]
    enumerator.supercell = tuple(
        int(value)
        for value in args.supercell
    )
    enumerator.images = []

    merge_symops = None

    if getattr(args, "merge", False):

        merge_parser = CifParser(
            str(cif_file),
            correlated_assemblies=False,
            molecular_crystal=not getattr(
                args,
                "not_molecular_crystal",
                False,
            ),
            symprec=float(
                getattr(args, "symprec", 1.0e-4)
            ),
        )
        merge_structure = (
            merge_parser.get_disordered_structure()
        )
        merge_symops = (
            merge_structure.spacegroup.get_symop()
        )

    number_of_cells = int(
        np.prod(enumerator.supercell)
    )
    total_sites = len(site_specs) * number_of_cells

    requested_total_sites = getattr(
        args,
        "total_sites",
        None,
    )

    if (
        requested_total_sites is not None
        and int(requested_total_sites) != total_sites
    ):
        raise ValueError(
            f"--total-sites={requested_total_sites} does not match the "
            f"prepared structure and supercell ({len(site_specs)} sites × "
            f"{number_of_cells} cells = {total_sites})."
        )

    LOG.info(
        "Prepared CIF contains %d independent molecular sites; "
        "supercell %s contains %d sites.",
        len(site_specs),
        tuple(enumerator.supercell),
        total_sites,
    )

    schedule = build_composition_schedule(
        args,
        component_order=component_order,
        total_sites=total_sites,
        orientation_degeneracies=orientation_degeneracies,
    )

    LOG.info(
        "Read %d unique requested compositions.",
        len(schedule),
    )

    if getattr(args, "exclude_ordered", False):
        ordered_atoms = Atoms(
            cell=prepared["cell"],
            pbc=False,
        )

    primitive_cache = {}

    retain_outputs = bool(
        getattr(
            args,
            "keep_orgdisord_outputs",
            False,
        )
        and not getattr(args, "no_write", False)
    )

    temporary_directory = None

    if retain_outputs:
        results_directory = (
            _prepare_results_directory(
                args.prefix
            )
        )
    else:
        temporary_directory = (
            tempfile.TemporaryDirectory(
                prefix=(
                    f"{Path(args.prefix).name}"
                    "_cspy_disord_"
                )
            )
        )
        results_directory = Path(
            temporary_directory.name
        )

    manifest_items = []
    summary_rows = []
    generated_cifs = []

    try:
        for composition_index, item in enumerate(
            schedule
        ):
            seed = getattr(args, "seed", None)

            if seed is not None:
                seed = (
                    int(seed)
                    + composition_index
                )

            LOG.info(
                "Sampling %s: counts=%s, n_samples=%d, lnW=%.6f.",
                item["label"],
                item["counts"],
                item["n_samples"],
                item["lnW"],
            )

            (
                state_choice_samples,
                number_of_possible_assignments,
            ) = sample_unique_state_choices(
                site_specs,
                supercell=enumerator.supercell,
                target_counts=item["counts"],
                component_order=component_order,
                n_samples=item["n_samples"],
                seed=seed,
            )
            
            actual_n_samples = len(state_choice_samples)

            if actual_n_samples != item["n_samples"]:
                LOG.info(
                    "Adjusted sample count for %s from %d to %d.",
                    item["label"],
                    item["n_samples"],
                    actual_n_samples,
                )

                item["n_samples"] = actual_n_samples

            structures = []
            metadata = []

            for sample_index, state_choices in enumerate(
                state_choice_samples
            ):
                atoms, primitive_indices = (
                    build_sample_structure(
                        enumerator,
                        prepared["cell"],
                        ordered_atoms,
                        site_specs,
                        state_choices,
                        primitive_cache,
                    )
                )

                structures.append(atoms)
                metadata.append(
                    {
                        "components": dict(
                            item["counts"]
                        ),
                        "ratio": dict(item["ratio"]),
                        "multiplicity": 1,
                        "orientation_degeneracies": dict(
                            orientation_degeneracies
                        ),
                        "lnW": float(item["lnW"]),
                        "composition_label": item["label"],
                        "sample_index": sample_index,
                        "state_choices": [
                            int(value)
                            for value in state_choices
                        ],
                        "site_states": describe_state_choices(
                            site_specs,
                            enumerator.supercell,
                            state_choices,
                        ),
                        "primitive_indices": [
                            int(value)
                            for value in primitive_indices
                        ],
                        "number_of_possible_state_assignments": int(
                            number_of_possible_assignments
                        ),
                    }
                )

            structures, metadata = (
                merge_generated_structures(
                    structures,
                    metadata,
                    merge_symops,
                    args,
                )
            )

            composition_slug = _safe_slug(
                item["label"]
            )

            for output_index, (
                atoms,
                structure_metadata,
            ) in enumerate(
                zip(structures, metadata)
            ):
                structure_id = (
                    f"{Path(args.prefix).name}_"
                    f"{composition_index:03d}_"
                    f"{composition_slug}_"
                    f"{output_index:05d}"
                )
                cif_path = (
                    results_directory
                    / f"{structure_id}.cif"
                )

                write_p1_cif_preserving_molecules(
                    cif_path,
                    atoms,
                )
                generated_cifs.append(cif_path)

                manifest_item = {
                    "id": structure_id,
                    "file": cif_path.name,
                    **structure_metadata,
                }
                manifest_items.append(
                    manifest_item
                )

                summary_row = {
                    "id": structure_id,
                    "composition_label": item["label"],
                    "multiplicity": structure_metadata.get(
                        "multiplicity",
                        1,
                    ),
                    "lnW": item["lnW"],
                    "n_requested_samples": item[
                        "n_samples"
                    ],
                    "number_of_possible_state_assignments": (
                        number_of_possible_assignments
                    ),
                }

                for component in component_order:
                    summary_row[
                        f"n_{component}"
                    ] = item["counts"][component]
                    summary_row[
                        f"x_{component}"
                    ] = item["ratio"][component]

                summary_rows.append(summary_row)

        if not generated_cifs:
            raise RuntimeError(
                "No structures were generated."
            )

        manifest = {
            "source_file": str(
                cif_file.resolve()
            ),
            "orientation_degeneracies": dict(
                orientation_degeneracies
            ),
            "structures": manifest_items,
        }
        manifest_path = (
            results_directory
            / f"{Path(args.prefix).name}_metadata.json"
        )
        manifest_path.write_text(
            json.dumps(
                manifest,
                indent=2,
                sort_keys=True,
            )
        )

        summary_path = Path(
            f"{args.prefix}.csv"
        )
        summary_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        pd.DataFrame(summary_rows).to_csv(
            summary_path,
            index=False,
        )

        number_written = write_cifs_to_cspy_db(
            cif_files=generated_cifs,
            output_db=args.output_db,
            manifest_path=manifest_path,
            source_file=cif_file,
            clean_cifs=not getattr(
                args,
                "no_reformat_cif",
                False,
            ),
        )

        LOG.info(
            "Stored %d structures in %s.",
            number_written,
            args.output_db,
        )

        if getattr(args, "keep_primitive", False):
            _write_primitive_debug_images(
                enumerator,
                args.prefix,
            )

        if not retain_outputs:
            summary_path.unlink(
                missing_ok=True
            )

        LOG.info(
            "Enumeration completed successfully."
        )

    finally:
        if temporary_directory is not None:
            temporary_directory.cleanup()
            