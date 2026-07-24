import copy
import logging
import re
import tempfile
from fractions import Fraction
from itertools import product
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from cspy import Crystal
from cspy.chem import Element
from cspy.formats.cif import Cif

from .disord_utils import (
    cif_block_value,
    cif_column,
    clean_cif_value,
    extract_cif_symmetry_operations,
    normalise_element_symbol,
    ordered_proxy_crystal,
    parse_cif_document,
    parse_cif_number,
    proxy_records,
    required_cif_float,
    safe_label,
    unique_atom_label,
    write_disordered_cif,
)

LOG = logging.getLogger(__name__)


def parse_group_component_map(text):
    """
    Parse mappings like:

        1=A1,2=A2
        A:1=A1,A:2=A2,B:1=A3,B:2=A4
    """
    mapping = {}

    if text is None or str(text).strip() == "":
        return mapping

    for item in str(text).split(","):
        item = item.strip()

        if not item:
            continue

        key, sep, value = item.partition("=")

        if not sep:
            raise ValueError(
                f"Could not parse {item!r}. Use forms like 1=A1,2=A2."
            )

        key = key.strip()
        value = value.strip()

        if not key:
            raise ValueError(f"Empty group key in {item!r}")

        if not value.startswith("A") or not value[1:].isdigit():
            raise ValueError(
                f"Component labels should look like A1, A2, A3. Got {value!r}"
            )

        mapping[key] = value

    return mapping

def parse_joint_state_specs(values):
    """
    Parse repeated joint-state definitions such as::

        --joint-state "A:1,B:2"
        --joint-state "A:2,B:1"

    Returns a list of dictionaries mapping original assembly labels to groups.
    """
    states = []

    for raw_value in values or []:
        state = {}

        for raw_item in str(raw_value).split(","):
            item = raw_item.strip()

            if not item:
                continue

            assembly, separator, group = item.partition(":")

            if not separator:
                raise ValueError(
                    f"Could not parse joint state item {item!r}. "
                    "Use ASSEMBLY:GROUP entries, for example A:1,B:2."
                )

            assembly = clean_cif_value(assembly)
            group = clean_cif_value(group)

            if not assembly or not group:
                raise ValueError(
                    f"Invalid joint state item {item!r}."
                )

            if assembly in state:
                raise ValueError(
                    f"Assembly {assembly!r} occurs more than once in "
                    f"joint state {raw_value!r}."
                )

            state[assembly] = group

        if not state:
            raise ValueError(
                f"Joint state {raw_value!r} contains no assembly/group choices."
            )

        states.append(state)

    seen = set()
    unique_states = []

    for state in states:
        key = tuple(sorted(state.items()))

        if key in seen:
            continue

        seen.add(key)
        unique_states.append(state)

    return unique_states


def _write_minimal_existing_disorder_proxy_cif(
    path,
    original_lines,
    parsed,
    use_original_space_group=True,
):
    """Write the ordered proxy using CSPy's standard cif writer."""
    block = parsed["cif_document"]["block"]

    a = required_cif_float(block, "_cell_length_a")
    b = required_cif_float(block, "_cell_length_b")
    c = required_cif_float(block, "_cell_length_c")
    alpha = required_cif_float(block, "_cell_angle_alpha")
    beta = required_cif_float(block, "_cell_angle_beta")
    gamma = required_cif_float(block, "_cell_angle_gamma")

    space_group_name = None
    space_group_number = None

    if use_original_space_group:
        space_group_name = cif_block_value(
            block,
            "_space_group_name_h-m_alt",
            "_symmetry_space_group_name_h-m",
        )
        space_group_number = cif_block_value(
            block,
            "_space_group_it_number",
            "_symmetry_int_tables_number",
        )

    if not space_group_name and not space_group_number:
        space_group_name = "P 1"
        space_group_number = 1

    if space_group_name is not None:
        space_group_name = clean_cif_value(space_group_name)

    if space_group_number is not None:
        space_group_number = int(round(parse_cif_number(space_group_number)))

    symmetry_operations = (
        extract_cif_symmetry_operations(original_lines)
        if use_original_space_group
        else ["x,y,z"]
    )

    first_atom = parsed["records"][0]
    data_block = {
        "cell_length_a": a,
        "cell_length_b": b,
        "cell_length_c": c,
        "cell_angle_alpha": alpha,
        "cell_angle_beta": beta,
        "cell_angle_gamma": gamma,
    }

    if space_group_name:
        data_block["space_group_name_H-M_alt"] = space_group_name
        data_block["symmetry_space_group_name_H-M"] = space_group_name

    if space_group_number is not None:
        data_block["space_group_IT_number"] = space_group_number
        data_block["symmetry_Int_Tables_number"] = space_group_number

    if symmetry_operations:
        # Operations are kept without embedded spaces so the current CSPy
        # Cif loop parser can read them back as one token per operation.
        data_block["symmetry_equiv_pos_as_xyz"] = [
            re.sub(r"\s+", "", str(operation))
            for operation in symmetry_operations
        ]

    data_block.update(
        {
            "atom_site_label": [
                safe_label(first_atom["label"]) or "X1"
            ],
            "atom_site_type_symbol": [first_atom["symbol"]],
            "atom_site_fract_x": [first_atom["fract_x"]],
            "atom_site_fract_y": [first_atom["fract_y"]],
            "atom_site_fract_z": [first_atom["fract_z"]],
            "atom_site_occupancy": [1.0],
        }
    )

    Cif({"cspy_existing_disorder_proxy": data_block,}).to_file(path)

def _load_existing_disorder_crystal(
    input_cif,
    original_lines,
    allow_p1_fallback=True,
):

    parsed = _parse_existing_disorder_atom_loop(original_lines)
    errors = []

    modes = (True, False) if allow_p1_fallback else (True,)

    for use_original_space_group in modes:
        temporary_path = None

        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".cif",
                prefix="cspy_existing_disorder_proxy_",
                delete=False,
            ) as handle:
                temporary_path = Path(handle.name)

            _write_minimal_existing_disorder_proxy_cif(
                temporary_path,
                original_lines,
                parsed,
                use_original_space_group=use_original_space_group,
            )

            crystal = Crystal.load(str(temporary_path))

            if not use_original_space_group:
                LOG.warning("The original space-group could not be loaded by mol-CSPy. Using a P1 proxy.")

            return crystal
        except Exception as exc:
            errors.append(
                ("original space group" if use_original_space_group else "P1", exc)
            )
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    details = "; ".join(f"{mode}: {error}" for mode, error in errors)
    raise ValueError(
        f"Could not construct a minimal mol-CSPy Crystal proxy for {input_cif}. "
        f"Attempts failed as follows: {details}"
    )

def _sortable_disorder_value(value):
    value = str(value)

    if value.isdigit():
        return (0, int(value))

    return (1, value)



def _parse_existing_disorder_atom_loop(lines):
    """Read disorder atom-site columns using CSPy's standard cif parser."""
    cif_document = parse_cif_document(lines)
    block = cif_document["block"]

    required_columns = (
        "_atom_site_label",
        "_atom_site_fract_x",
        "_atom_site_fract_y",
        "_atom_site_fract_z",
        "_atom_site_disorder_assembly",
        "_atom_site_disorder_group",
    )

    columns = {name: cif_column(block, name) for name in required_columns}

    labels = columns["_atom_site_label"]
    n_rows = len(labels)

    for name, values in columns.items():
        if len(values) != n_rows:
            raise ValueError(
                f"CIF atom-site column {name!r} has {len(values)} values; "
                f"expected {n_rows}."
            )

    type_symbols = cif_column(block, "_atom_site_type_symbol", required=False,)
    occupancies = cif_column(block, "_atom_site_occupancy", required=False,)

    for name, values in (
        ("_atom_site_type_symbol", type_symbols),
        ("_atom_site_occupancy", occupancies),
    ):
        if values is not None and len(values) != n_rows:
            raise ValueError(
                f"CIF atom-site column {name!r} has {len(values)} values; "
                f"expected {n_rows}."
            )

    records = []

    for source_index in range(n_rows):
        label = clean_cif_value(labels[source_index])
        symbol_value = (
            type_symbols[source_index]
            if type_symbols is not None
            else label
        )
        symbol = normalise_element_symbol(symbol_value, label=label,)

        assembly = clean_cif_value(
            columns["_atom_site_disorder_assembly"][source_index]
        )
        group = clean_cif_value(
            columns["_atom_site_disorder_group"][source_index]
        )

        # Treat incomplete or zero-valued disorder tags as common atoms.
        if not assembly or not group or group == "0":
            assembly = ""
            group = ""

        occupancy = 1.0

        if occupancies is not None:
            raw_occupancy = clean_cif_value(
                occupancies[source_index]
            )

            if raw_occupancy:
                occupancy = parse_cif_number(raw_occupancy)

        records.append(
            {
                "source_index": source_index,
                "label": label,
                "symbol": symbol,
                "fract_x": parse_cif_number(
                    columns["_atom_site_fract_x"][source_index]
                ),
                "fract_y": parse_cif_number(
                    columns["_atom_site_fract_y"][source_index]
                ),
                "fract_z": parse_cif_number(
                    columns["_atom_site_fract_z"][source_index]
                ),
                "occupancy": occupancy,
                "disorder_assembly": assembly,
                "disorder_group": group,
            }
        )

    return {
        "lines": cif_document["lines"],
        "cif_document": cif_document,
        "records": records,
    }

def _default_group_by_assembly(records):
    defaults = {}

    assemblies = sorted(
        {
            row["disorder_assembly"]
            for row in records
            if row["disorder_assembly"]
        }
    )

    for assembly in assemblies:
        groups = sorted(
            {
                row["disorder_group"]
                for row in records
                if row["disorder_assembly"] == assembly
            },
            key=_sortable_disorder_value,
        )

        if not groups:
            continue

        defaults[assembly] = groups[0]

    return defaults

def _select_ordered_state_choices(records, defaults, selected_groups=None):

    selected_groups = {
        str(assembly): str(group)
        for assembly, group in (selected_groups or {}).items()
    }
    selected = []

    for row in records:
        assembly = row["disorder_assembly"]
        group = row["disorder_group"]

        if not assembly:
            selected.append(row)
            continue

        wanted_group = selected_groups.get(
            assembly,
            defaults[assembly],
        )

        if group == wanted_group:
            selected.append(row)

    return selected

def _map_proxy_molecules_to_rows(proxy, selected_rows):

    molecules = proxy.asym_mols()

    if not molecules:
        raise RuntimeError(
            "mol-CSPy did not identify any molecules in the ordered proxy crystal."
        )

    row_frac = np.asarray(
        [
            [row["fract_x"], row["fract_y"], row["fract_z"]]
            for row in selected_rows
        ],
        dtype=float,
    )
    row_symbols = np.asarray([row["symbol"] for row in selected_rows], dtype=object)
    globally_used = set()
    mapped_molecules = []

    for molecule_index, molecule in enumerate(molecules, start=1):
        molecule_frac = np.asarray(
            proxy.to_fractional(np.asarray(molecule.positions, dtype=float)),
            dtype=float,
        )
        molecule_symbols = np.asarray(
            [
                Element.from_atomic_number(int(number)).symbol
                for number in molecule.atomic_numbers
            ],
            dtype=object,
        )

        delta = molecule_frac[:, None, :] - row_frac[None, :, :]
        delta -= np.rint(delta)
        cost = np.sum(delta * delta, axis=2)
        cost[molecule_symbols[:, None] != row_symbols[None, :]] = 1.0e6

        for row_index in globally_used:
            cost[:, row_index] = 1.0e6

        molecule_atom_indices, selected_row_indices = linear_sum_assignment(cost)

        if len(molecule_atom_indices) != len(molecule_frac):
            raise RuntimeError(
                f"Could not map every atom in proxy molecule {molecule_index} "
                "back to an atom-site row."
            )

        ordered_pairs = sorted(
            zip(molecule_atom_indices, selected_row_indices),
            key=lambda item: item[0],
        )

        worst_cost = max(float(cost[i, j]) for i, j in ordered_pairs)

        if worst_cost > 1.0e-8:
            raise RuntimeError(
                "Could not reliably map mol-CSPy molecule atoms back to the "
                f"CIF atom-site rows (molecule {molecule_index}, maximum "
                f"fractional-coordinate mismatch {worst_cost:.3e}). The "
                "existing-disorder input may need to be expanded to P1 first."
            )

        mapped_rows = [selected_rows[j] for _, j in ordered_pairs]
        globally_used.update(j for _, j in ordered_pairs)
        mapped_molecules.append(mapped_rows)

    if len(globally_used) != len(selected_rows):
        missing = [
            row["label"]
            for index, row in enumerate(selected_rows)
            if index not in globally_used
        ]
        raise RuntimeError(
            "mol-CSPy molecular reconstruction left atom-site rows unmapped: "
            + ", ".join(missing[:20])
        )

    return mapped_molecules

def _state_occupancy(records, assembly, group, number_of_groups):
    values = [
        float(row["occupancy"])
        for row in records
        if row["disorder_assembly"] == assembly
        and row["disorder_group"] == group
    ]

    if not values:
        return 1.0 / float(number_of_groups)

    reference = values[0]

    if any(abs(value - reference) > 1.0e-6 for value in values[1:]):
        LOG.warning(
            "Disorder state %s:%s has inconsistent atom occupancies %s; "
            "using their median for the completed molecule.",
            assembly,
            group,
            sorted(set(values)),
        )
        return float(np.median(values))

    return reference

def _groups_by_assembly(records):
    result = {}

    for row in records:
        assembly = row["disorder_assembly"]
        group = row["disorder_group"]

        if not assembly:
            continue

        result.setdefault(assembly, set()).add(group)

    return {
        assembly: sorted(groups, key=_sortable_disorder_value)
        for assembly, groups in result.items()
    }

def _occupancy_by_state(records, groups_by_assembly):
    return {
        (assembly, group): _state_occupancy(
            records,
            assembly,
            group,
            number_of_groups=len(groups),
        )
        for assembly, groups in groups_by_assembly.items()
        for group in groups
    }

def _infer_joint_states_by_occupancy(
    assemblies,
    groups_by_assembly,
    occupancy_by_state,
    tolerance,
):

    assemblies = tuple(sorted(assemblies))
    reference_assembly = assemblies[0]
    reference_groups = groups_by_assembly[reference_assembly]
    mappings = {}

    for assembly in assemblies[1:]:
        trial_groups = groups_by_assembly[assembly]

        if len(trial_groups) != len(reference_groups):
            raise ValueError(
                "Cannot infer correlated joint states because assemblies "
                f"{reference_assembly!r} and {assembly!r} have different "
                f"numbers of groups ({len(reference_groups)} and "
                f"{len(trial_groups)}). Supply explicit --joint-state values "
                "or use --multi-assembly-mode independent."
            )

        mapping = {}

        for reference_group in reference_groups:
            reference_occupancy = occupancy_by_state[
                (reference_assembly, reference_group)
            ]
            matches = [
                trial_group
                for trial_group in trial_groups
                if abs(
                    occupancy_by_state[(assembly, trial_group)]
                    - reference_occupancy
                ) <= tolerance
            ]

            if len(matches) != 1:
                raise ValueError(
                    "Could not infer a unique occupancy-based pairing for "
                    f"{reference_assembly}:{reference_group} "
                    f"(occupancy {reference_occupancy:.6g}) in assembly "
                    f"{assembly!r}. Candidate groups were {matches}. "
                    "Supply explicit --joint-state values."
                )

            mapping[reference_group] = matches[0]

        if len(set(mapping.values())) != len(mapping):
            raise ValueError(
                f"Occupancy matching between assemblies {reference_assembly!r} "
                f"and {assembly!r} is not one-to-one. Supply explicit "
                "--joint-state values."
            )

        mappings[assembly] = mapping

    joint_states = []

    for reference_group in reference_groups:
        state = {
            reference_assembly: reference_group,
        }

        for assembly in assemblies[1:]:
            state[assembly] = mappings[assembly][reference_group]

        joint_states.append(state)

    return joint_states

def _resolve_joint_states(
    assemblies,
    groups_by_assembly,
    occupancy_by_state,
    explicit_joint_states,
    mode,
    occupancy_tolerance,
):
    assemblies = tuple(sorted(assemblies))
    assembly_set = set(assemblies)
    applicable_explicit = [
        state
        for state in explicit_joint_states
        if set(state) == assembly_set
    ]

    if len(assemblies) == 1:
        assembly = assemblies[0]
        return [
            {assembly: group}
            for group in groups_by_assembly[assembly]
        ]

    if applicable_explicit:
        joint_states = applicable_explicit
    elif mode == "explicit":
        raise ValueError(
            "A molecule contains several disorder assemblies "
            f"{assemblies}, but no matching explicit --joint-state values "
            "were supplied."
        )
    elif mode == "independent":
        joint_states = [
            dict(zip(assemblies, group_choices))
            for group_choices in product(
                *(groups_by_assembly[assembly] for assembly in assemblies)
            )
        ]
    elif mode == "auto":
        joint_states = _infer_joint_states_by_occupancy(
            assemblies,
            groups_by_assembly,
            occupancy_by_state,
            tolerance=float(occupancy_tolerance),
        )
    else:
        raise ValueError(
            f"Unknown multi-assembly mode: {mode!r}."
        )

    for state in joint_states:
        if set(state) != assembly_set:
            raise ValueError(
                f"Joint state {state} does not define exactly assemblies "
                f"{assemblies}."
            )

        for assembly, group in state.items():
            if group not in groups_by_assembly[assembly]:
                raise ValueError(
                    f"Joint state {state} selects unknown state "
                    f"{assembly}:{group}."
                )

    represented = {
        (assembly, group)
        for state in joint_states
        for assembly, group in state.items()
    }
    expected = {
        (assembly, group)
        for assembly in assemblies
        for group in groups_by_assembly[assembly]
    }
    missing = sorted(expected.difference(represented))

    if missing:
        raise ValueError(
            "The resolved joint states do not represent every original "
            f"assembly/group state. Missing: {missing}."
        )

    return joint_states

def _joint_state_original_occupancy(state, occupancy_by_state, mode):
    values = np.asarray(
        [
            occupancy_by_state[(assembly, group)]
            for assembly, group in sorted(state.items())
        ],
        dtype=float,
    )

    if mode == "independent" and len(values) > 1:
        return float(np.prod(values))

    if len(values) > 1 and np.max(values) - np.min(values) > 0.02:
        LOG.warning(
            "Joint state %s combines differing experimental occupancies %s; "
            "using their median as provenance metadata.",
            state,
            values.tolist(),
        )

    return float(np.median(values))


def _joint_state_component_key(state):
    """
    Return a stable key for a complete joint disorder state.
    """

    return "+".join(
        f"{assembly}:{group}"
        for assembly, group in sorted(
            (
                str(assembly),
                str(group),
            )
            for assembly, group in state.items()
        )
    )
    

def _joint_component(
    state,
    user_component_map,
    existing_component,
):
    # First check whether the complete joint state has an explicit
    joint_key = _joint_state_component_key(state)

    joint_component = user_component_map.get(
        joint_key
    )

    if joint_component is not None:
        return str(joint_component)

    mapped = []

    for assembly, group in sorted(state.items()):
        component = (
            user_component_map.get(
                f"{assembly}:{group}"
            )
            or user_component_map.get(str(group))
        )

        if component is not None:
            mapped.append(component)

    unique = sorted(set(mapped))

    if len(unique) > 1:
        raise ValueError(
            f"Joint state {state} maps to conflicting CSPy components "
            f"{unique}. Map the complete joint state using a key such as "
            f"{joint_key}=A1."
        )

    if unique:
        return unique[0]

    if not re.fullmatch(
        r"A\d+",
        str(existing_component),
    ):
        raise ValueError(
            "--existing-component must look like A1, A2, A3, ...; "
            f"got {existing_component!r}."
        )

    return str(existing_component)

def _molecule_assembly_set(molecule_rows):
    return frozenset(
        row["disorder_assembly"]
        for row in molecule_rows
        if row["disorder_assembly"]
    )

def _fractional_circular_centroid(frac):
    frac = np.asarray(frac, dtype=float)
    angles = 2.0 * np.pi * frac
    mean_sin = np.mean(np.sin(angles), axis=0)
    mean_cos = np.mean(np.cos(angles), axis=0)
    centroid = np.arctan2(mean_sin, mean_cos) / (2.0 * np.pi)
    return np.mod(centroid, 1.0)

def _block_fractional_centroid(block):
    return _fractional_circular_centroid(
        [
            [row["fract_x"], row["fract_y"], row["fract_z"]]
            for row in block
        ]
    )

def _molecule_fractional_centroid(crystal, molecule):
    frac = crystal.to_fractional(np.asarray(molecule.positions, dtype=float))
    return _fractional_circular_centroid(frac)

def _periodic_assignment(reference_centres, trial_centres, description):
    reference_centres = np.asarray(reference_centres, dtype=float)
    trial_centres = np.asarray(trial_centres, dtype=float)

    if len(reference_centres) != len(trial_centres):
        raise ValueError(
            f"Cannot pair {description}: different numbers of objects "
            f"({len(reference_centres)} and {len(trial_centres)})."
        )

    delta = reference_centres[:, None, :] - trial_centres[None, :, :]
    delta -= np.rint(delta)
    cost = np.sum(delta * delta, axis=2)
    row_indices, column_indices = linear_sum_assignment(cost)

    ordered = [None] * len(reference_centres)

    for row_index, column_index in zip(row_indices, column_indices):
        ordered[int(row_index)] = int(column_index)

    worst = max(
        float(np.sqrt(cost[row_index, column_index]))
        for row_index, column_index in zip(row_indices, column_indices)
    ) if len(row_indices) else 0.0

    if worst > 0.20:
        LOG.warning(
            "Large periodic fractional-centroid displacement while pairing %s: %.4f",
            description,
            worst,
        )

    return ordered

def _expand_completed_block_to_p1(crystal, block, title):
    proxy = ordered_proxy_crystal(
        crystal,
        proxy_records(block),
        title=title,
    )
    p1_crystal = proxy.as_P1()
    molecules = list(p1_crystal.asym_mols())

    if not molecules:
        raise RuntimeError(
            f"No P1 molecules were generated while expanding {title}."
        )

    expected_atoms = len(block)

    for molecule_index, molecule in enumerate(molecules, start=1):
        if len(molecule.atomic_numbers) != expected_atoms:
            raise RuntimeError(
                f"P1 expansion of {title} produced molecule {molecule_index} with "
                f"{len(molecule.atomic_numbers)} atoms; expected {expected_atoms}."
            )

    molecules.sort(
        key=lambda molecule: tuple(
            np.round(_molecule_fractional_centroid(p1_crystal, molecule), 10)
        )
    )
    return p1_crystal, molecules

def _append_p1_molecule_records(
    atom_records,
    crystal,
    molecule,
    assembly,
    group,
    component,
    occupancy,
    used_labels,
    source,
    mapping_index=1,
    n_mappings=1,
    original_joint_state=None,
    original_joint_occupancy=None,
):
    frac = np.asarray(
        crystal.to_fractional(np.asarray(molecule.positions, dtype=float)),
        dtype=float,
    )
    frac = np.mod(frac, 1.0)

    for atom_index, (atomic_number, xyz) in enumerate(
        zip(molecule.atomic_numbers, frac),
        start=1,
    ):
        symbol = Element.from_atomic_number(int(atomic_number)).symbol

        if assembly == ".":
            preferred_label = f"ORD_{symbol}{atom_index}"
        else:
            preferred_label = f"{component}_{assembly}_{symbol}{atom_index}"

        label = unique_atom_label(preferred_label, used_labels)
        atom_records.append(
            {
                "label": label,
                "symbol": symbol,
                "fract_x": float(xyz[0]),
                "fract_y": float(xyz[1]),
                "fract_z": float(xyz[2]),
                "occupancy": float(occupancy),
                "disorder_assembly": assembly,
                "disorder_group": group,
                "component": component,
                "source": str(source),
                "mapping_index": int(mapping_index),
                "raw_mapping_index": int(mapping_index),
                "n_mappings": int(n_mappings),
                "overlay_rmsd": None,
                "symmetry_operation": None,
                "original_joint_state": copy.deepcopy(original_joint_state),
                "original_joint_occupancy": original_joint_occupancy,
            }
        )

def _parse_symmetry_coordinate_expression(expression):
    """
    Convert one CIF symmetry-coordinate expression to linear coefficients.

    Examples
    --------
    ``-x+1/2`` becomes ``([-1, 0, 0], 1/2)``.
    """
    expression = re.sub(r"\s+", "", str(expression))

    if not expression:
        raise ValueError("Empty CIF symmetry-coordinate expression.")

    if expression[0] not in "+-":
        expression = "+" + expression

    terms = re.findall(r"[+-][^+-]+", expression)
    coefficients = np.zeros(3, dtype=int)
    translation = Fraction(0, 1)
    variable_index = {"x": 0, "y": 1, "z": 2}

    for term in terms:
        sign = -1 if term[0] == "-" else 1
        body = term[1:]

        variable = next(
            (name for name in "xyz" if name in body.lower()),
            None,
        )

        if variable is None:
            translation += sign * Fraction(body)
            continue

        lower = body.lower()

        if lower.count(variable) != 1:
            raise ValueError(
                f"Unsupported CIF symmetry term {term!r}."
            )

        coefficient_text = lower.replace(variable, "").replace("*", "")

        if coefficient_text in {"", "1"}:
            coefficient = 1
        elif coefficient_text == "-1":
            coefficient = -1
        else:
            coefficient_fraction = Fraction(coefficient_text)

            if coefficient_fraction.denominator != 1:
                raise ValueError(
                    "Fractional variable coefficients are not supported in "
                    f"CIF symmetry term {term!r}."
                )

            coefficient = int(coefficient_fraction)

        coefficients[variable_index[variable]] += sign * coefficient

    return coefficients, float(translation)

def _symmetry_affine_operations(lines):
    """
    Return CIF symmetry operations as ``(rotation, translation, text)``.

    Fractional row vectors are transformed as::

        transformed = rotation @ fractional + translation
    """
    operation_strings = extract_cif_symmetry_operations(lines)

    if not operation_strings:
        operation_strings = ["x,y,z"]

    operations = []
    seen = set()

    for operation_text in operation_strings:
        coordinate_expressions = [
            item.strip()
            for item in str(operation_text).split(",")
        ]

        if len(coordinate_expressions) != 3:
            raise ValueError(
                f"Invalid CIF symmetry operation {operation_text!r}."
            )

        rotation = np.zeros((3, 3), dtype=int)
        translation = np.zeros(3, dtype=float)

        for output_index, expression in enumerate(coordinate_expressions):
            coefficients, offset = (
                _parse_symmetry_coordinate_expression(expression)
            )
            rotation[output_index] = coefficients
            translation[output_index] = offset

        key = (
            tuple(rotation.reshape(-1).tolist()),
            tuple(np.mod(translation, 1.0).round(12).tolist()),
        )

        if key in seen:
            continue

        seen.add(key)
        operations.append(
            (
                rotation,
                translation,
                str(operation_text),
            )
        )

    return operations

def _transform_fractional_position(position, operation):
    rotation, translation, _operation_text = operation
    position = np.asarray(position, dtype=float)
    transformed = rotation @ position + translation
    return np.mod(transformed, 1.0)

def _transform_source_row(source_row, operation):
    row = copy.deepcopy(source_row)
    transformed = _transform_fractional_position(
        [
            source_row["fract_x"],
            source_row["fract_y"],
            source_row["fract_z"],
        ],
        operation,
    )
    row["fract_x"] = float(transformed[0])
    row["fract_y"] = float(transformed[1])
    row["fract_z"] = float(transformed[2])
    return row

def _transformed_block(block, operation):
    return [
        _transform_source_row(row, operation)
        for row in block
    ]

def _deduplicated_family_operations(reference_block, operations, tolerance=1.0e-7):
    """
    Return symmetry operations producing unique physical molecular sites.
    """
    retained = []
    retained_centres = []

    for operation in operations:
        transformed = _transformed_block(reference_block, operation)
        centre = _block_fractional_centroid(transformed)

        duplicate = False

        for existing in retained_centres:
            delta = centre - existing
            delta -= np.rint(delta)

            if np.linalg.norm(delta) <= tolerance:
                duplicate = True
                break

        if duplicate:
            continue

        retained.append(operation)
        retained_centres.append(centre)

    order = sorted(
        range(len(retained)),
        key=lambda index: tuple(np.round(retained_centres[index], 10)),
    )

    return [retained[index] for index in order]

def _prepared_assembly_suffix(original_assembly, assembly_index):
    cleaned = re.sub(
        r"[^A-Za-z0-9]+",
        "",
        str(original_assembly),
    )

    if not cleaned:
        cleaned = f"R{assembly_index:02d}"

    return cleaned

def _make_prepared_atom_record(
    source_row,
    operation,
    label,
    assembly,
    group,
    component,
    occupancy,
    source,
    physical_site,
    role,
    original_assembly=None,
    mapping_index=1,
    n_mappings=1,
):
    transformed = _transform_source_row(source_row, operation)

    return {
        "label": str(label),
        "symbol": transformed["symbol"],
        "fract_x": transformed["fract_x"],
        "fract_y": transformed["fract_y"],
        "fract_z": transformed["fract_z"],
        "occupancy": float(occupancy),
        "disorder_assembly": str(assembly),
        "disorder_group": str(group),
        "component": str(component),
        "source": str(source),
        "mapping_index": int(mapping_index),
        "raw_mapping_index": int(mapping_index),
        "n_mappings": int(n_mappings),
        "overlay_rmsd": None,
        "symmetry_operation": operation[2],
        "physical_site": str(physical_site),
        "site_role": str(role),
        "original_assembly": (
            None if original_assembly is None
            else str(original_assembly)
        ),
    }

def _family_fragment_rows(
    crystal,
    records,
    defaults,
    families,
    assemblies,
    groups_by_assembly,
):
    """
    Return explicit fragment rows for every family/assembly/group.

    The molecular assignment uses complete ordered proxies, but only the atoms
    belonging to the requested original assembly/group are retained.
    """
    reference_centres = [
        family["centroid"]
        for family in families
    ]
    fragments = [
        {
            assembly: {}
            for assembly in assemblies
        }
        for _family in families
    ]

    for assembly in assemblies:
        for group in groups_by_assembly[assembly]:
            selected = _select_ordered_state_choices(
                records,
                defaults,
                selected_groups={assembly: group},
            )
            proxy = ordered_proxy_crystal(
                crystal,
                proxy_records(selected),
                title=(
                    "existing_fragment_"
                    f"{safe_label(assembly)}_{safe_label(group)}"
                ),
            )
            mapped_molecules = _map_proxy_molecules_to_rows(
                proxy,
                selected,
            )
            candidates = [
                molecule_rows
                for molecule_rows in mapped_molecules
                if _molecule_assembly_set(molecule_rows)
                == frozenset(assemblies)
            ]

            if len(candidates) != len(families):
                raise RuntimeError(
                    f"State {assembly}:{group} produced {len(candidates)} "
                    f"molecules for assembly cluster {assemblies}; expected "
                    f"{len(families)}."
                )

            assignment = _periodic_assignment(
                reference_centres,
                [
                    _block_fractional_centroid(block)
                    for block in candidates
                ],
                f"fragment state {assembly}:{group}",
            )

            for family_index, candidate_index in enumerate(assignment):
                candidate = candidates[candidate_index]
                explicit_rows = [
                    row
                    for row in candidate
                    if row["disorder_assembly"] == assembly
                    and row["disorder_group"] == group
                ]

                if not explicit_rows:
                    raise RuntimeError(
                        f"No explicit atoms were found for "
                        f"{assembly}:{group} in molecular family "
                        f"{family_index + 1}."
                    )

                fragments[family_index][assembly][group] = explicit_rows

    return fragments

def build_fragment_p1_existing_disorder_records(
    crystal,
    parsed,
    group_components=None,
    source="",
    joint_states=None,
    multi_assembly_mode="auto",
    occupancy_tolerance=0.02,
    existing_component="A1",
):
    """
    Convert existing non-P1 molecular disorder to a fragment-preserving P1 CIF.

    Common molecular scaffold atoms are written once as ordered atom rows.
    Every original disorder region remains a separate prepared assembly.
    Site-level metadata records the allowed combinations of those assemblies.

    Thus a molecule controlled by original assemblies A and B becomes, for
    example, prepared assemblies ``S001A`` and ``S001B`` rather than one
    collapsed complete-molecule assembly.
    """
    records = parsed["records"]
    defaults = _default_group_by_assembly(records)

    if not defaults:
        raise ValueError("No explicit disorder assemblies were found.")

    groups_by_assembly = _groups_by_assembly(records)
    occupancy_by_state = _occupancy_by_state(
        records,
        groups_by_assembly,
    )
    explicit_joint_states = parse_joint_state_specs(joint_states)
    user_component_map = parse_group_component_map(group_components)

    default_selected = _select_ordered_state_choices(
        records,
        defaults,
    )
    default_proxy = ordered_proxy_crystal(
        crystal,
        proxy_records(default_selected),
        title="existing_disorder_default_state",
    )
    default_molecules = _map_proxy_molecules_to_rows(
        default_proxy,
        default_selected,
    )

    ordered_reference_molecules = []
    families_by_cluster = {}

    for molecule_rows in default_molecules:
        assemblies = tuple(
            sorted(_molecule_assembly_set(molecule_rows))
        )

        if not assemblies:
            ordered_reference_molecules.append(molecule_rows)
            continue

        families_by_cluster.setdefault(
            assemblies,
            [],
        ).append(
            {
                "rows": molecule_rows,
                "assemblies": assemblies,
                "centroid": _block_fractional_centroid(molecule_rows),
            }
        )

    if not families_by_cluster:
        raise RuntimeError(
            "No molecules containing explicit disorder were identified."
        )

    available_clusters = {
        frozenset(cluster)
        for cluster in families_by_cluster
    }
    unused_explicit = [
        state
        for state in explicit_joint_states
        if frozenset(state) not in available_clusters
    ]

    if unused_explicit:
        raise ValueError(
            "Explicit --joint-state definitions do not match any molecule's "
            f"assembly cluster: {unused_explicit}. Available clusters are "
            f"{sorted(tuple(sorted(cluster)) for cluster in available_clusters)}."
        )

    operations = _symmetry_affine_operations(
        parsed["lines"]
    )
    atom_records = []
    site_metadata = []
    used_labels = set()
    output_crystal = None
    next_site_index = 1

    for assemblies, families in sorted(families_by_cluster.items()):
        allowed_original_states = _resolve_joint_states(
            assemblies,
            groups_by_assembly,
            occupancy_by_state,
            explicit_joint_states,
            mode=multi_assembly_mode,
            occupancy_tolerance=occupancy_tolerance,
        )

        state_components = {
            tuple(sorted(
                (str(assembly), str(group))
                for assembly, group in state.items()
            )): _joint_component(
                state,
                user_component_map,
                existing_component,
            )
            for state in allowed_original_states
        }

        components = set(state_components.values())
        site_component = str(existing_component)

        component = next(iter(components))
        families = sorted(
            families,
            key=lambda family: tuple(
                np.round(family["centroid"], 10)
            ),
        )
        fragments_by_family = _family_fragment_rows(
            crystal,
            records,
            defaults,
            families,
            assemblies,
            groups_by_assembly,
        )

        for family_index, family in enumerate(families):
            reference_block = family["rows"]
            common_rows = [
                row
                for row in reference_block
                if not row["disorder_assembly"]
            ]

            if not common_rows:
                LOG.warning(
                    "Molecular family %d for assemblies %s has no common "
                    "scaffold atoms.",
                    family_index + 1,
                    assemblies,
                )

            # Obtain a P1 crystal object with the original cell and P1 space
            # group for the output writer and ordered-molecule expansion.
            p1_crystal, _molecules = _expand_completed_block_to_p1(
                crystal,
                reference_block,
                title=(
                    "existing_fragment_reference_"
                    + "_".join(safe_label(value) for value in assemblies)
                    + f"_{family_index + 1}"
                ),
            )

            if output_crystal is None:
                output_crystal = p1_crystal

            family_operations = _deduplicated_family_operations(
                reference_block,
                operations,
            )

            for operation in family_operations:
                physical_site = f"S{next_site_index:03d}"
                next_site_index += 1
                common_labels = []
                prepared_assemblies = []
                prepared_name_by_original = {}

                for assembly_index, original_assembly in enumerate(
                    assemblies,
                    start=1,
                ):
                    suffix = _prepared_assembly_suffix(
                        original_assembly,
                        assembly_index,
                    )
                    prepared_assembly = (
                        f"{physical_site}{suffix}"
                    )
                    prepared_name_by_original[
                        original_assembly
                    ] = prepared_assembly
                    prepared_assemblies.append(
                        {
                            "assembly": prepared_assembly,
                            "original_assembly": str(original_assembly),
                            "groups": [
                                str(group)
                                for group in groups_by_assembly[
                                    original_assembly
                                ]
                            ],
                        }
                    )

                for common_index, source_row in enumerate(
                    common_rows,
                    start=1,
                ):
                    preferred = (
                        f"{site_component}_{physical_site}_COMMON_"
                        f"{safe_label(source_row['label']) or common_index}"
                    )
                    label = unique_atom_label(
                        preferred,
                        used_labels,
                    )
                    common_labels.append(label)
                    atom_records.append(
                        _make_prepared_atom_record(
                            source_row=source_row,
                            operation=operation,
                            label=label,
                            assembly=".",
                            group=".",
                            component=site_component,
                            occupancy=1.0,
                            source=source,
                            physical_site=physical_site,
                            role="common",
                            n_mappings=len(allowed_original_states),
                        )
                    )

                for assembly_index, original_assembly in enumerate(
                    assemblies,
                    start=1,
                ):
                    prepared_assembly = (
                        prepared_name_by_original[
                            original_assembly
                        ]
                    )
                    original_groups = groups_by_assembly[
                        original_assembly
                    ]
                    group_occupancy = (
                        1.0 / float(len(original_groups))
                    )

                    for mapping_index, original_group in enumerate(
                        original_groups,
                        start=1,
                    ):  
                        
                        fragment_component = (
                            user_component_map.get(
                                f"{original_assembly}:{original_group}"
                            )
                            or user_component_map.get(str(original_group))
                            or str(existing_component)
                        )
                        
                        fragment_rows = fragments_by_family[
                            family_index
                        ][original_assembly][original_group]

                        for fragment_index, source_row in enumerate(
                            fragment_rows,
                            start=1,
                        ):
                            preferred = (
                                f"{fragment_component}_{prepared_assembly}_"
                                f"G{safe_label(original_group)}_"
                                f"{safe_label(source_row['label']) or fragment_index}"
                            )
                            label = unique_atom_label(
                                preferred,
                                used_labels,
                            )
                            atom_records.append(
                                _make_prepared_atom_record(
                                    source_row=source_row,
                                    operation=operation,
                                    label=label,
                                    assembly=prepared_assembly,
                                    group=str(original_group),
                                    component=fragment_component,
                                    occupancy=group_occupancy,
                                    source=source,
                                    physical_site=physical_site,
                                    role="fragment",
                                    original_assembly=original_assembly,
                                    mapping_index=mapping_index,
                                    n_mappings=len(allowed_original_states),
                                )
                            )

                allowed_states = []

                for state_index, original_state in enumerate(
                    allowed_original_states,
                    start=1,
                ):
                    
                    state_key = tuple(sorted(
                        (str(assembly), str(group))
                        for assembly, group in original_state.items()
                    ))

                    state_component = state_components[state_key]
                    
                    prepared_choices = {
                        prepared_name_by_original[assembly]: str(group)
                        for assembly, group in sorted(
                            original_state.items()
                        )
                    }
                    allowed_states.append(
                        {
                            "state_index": state_index,
                            "component": state_component,
                            "choices": prepared_choices,
                            "original_choices": {
                                str(assembly): str(group)
                                for assembly, group in sorted(
                                    original_state.items()
                                )
                            },
                            "original_occupancy": (
                                _joint_state_original_occupancy(
                                    original_state,
                                    occupancy_by_state,
                                    mode=multi_assembly_mode,
                                )
                            ),
                        }
                    )

                site_metadata.append(
                    {
                        "site": physical_site,
                        "component": site_component,
                        "components": sorted(components),
                        "common_labels": common_labels,
                        "assemblies": prepared_assemblies,
                        "allowed_states": allowed_states,
                        "state_count": len(allowed_states),
                        "mode": str(multi_assembly_mode),
                    }
                )

    # Fully ordered molecules that are not part of a disordered site are
    # expanded and retained as global ordered atoms.
    for ordered_index, block in enumerate(
        ordered_reference_molecules,
        start=1,
    ):
        p1_crystal, molecules = _expand_completed_block_to_p1(
            crystal,
            block,
            title=f"existing_ordered_{ordered_index}",
        )

        if output_crystal is None:
            output_crystal = p1_crystal

        for molecule in molecules:
            _append_p1_molecule_records(
                atom_records=atom_records,
                crystal=p1_crystal,
                molecule=molecule,
                assembly=".",
                group=".",
                component="ORDERED",
                occupancy=1.0,
                used_labels=used_labels,
                source=source,
            )

    if output_crystal is None:
        raise RuntimeError(
            "No atoms were generated for the prepared P1 disorder model."
        )

    LOG.info(
        "Expanded existing disorder to P1 with %d physical molecular sites, "
        "%d prepared partial assemblies, and %d atom-site rows.",
        len(site_metadata),
        sum(len(site["assemblies"]) for site in site_metadata),
        len(atom_records),
    )

    return output_crystal, atom_records, site_metadata

def annotate_existing_disorder_cif(
    input_cif,
    output_cif,
    group_components=None,
    joint_states=None,
    multi_assembly_mode="auto",
    occupancy_tolerance=0.02,
    existing_component="A1",
):
    """
    Convert existing molecular disorder to a fragment-preserving P1 model.

    Common molecular scaffold atoms are written once. Original disorder
    regions remain separate prepared assemblies, while allowed correlated
    combinations are stored as site-level metadata for direct enumeration.
    """
    input_cif = Path(input_cif)
    output_cif = Path(output_cif)

    original_lines = input_cif.read_text(
        errors="replace"
    ).splitlines()
    parsed = _parse_existing_disorder_atom_loop(
        original_lines
    )

    crystal = _load_existing_disorder_crystal(
        input_cif,
        original_lines,
        allow_p1_fallback=False,
    )

    (
        p1_crystal,
        atom_records,
        site_metadata,
    ) = build_fragment_p1_existing_disorder_records(
        crystal=crystal,
        parsed=parsed,
        group_components=group_components,
        source=input_cif,
        joint_states=joint_states,
        multi_assembly_mode=multi_assembly_mode,
        occupancy_tolerance=occupancy_tolerance,
        existing_component=existing_component,
    )

    write_disordered_cif(
        output_cif,
        p1_crystal,
        atom_records,
        site_metadata=site_metadata,
    )

    LOG.info(
        "Prepared existing-disorder CIF %s -> %s as P1 with %d physical "
        "molecular sites and %d separate partial disorder assemblies.",
        input_cif,
        output_cif,
        len(site_metadata),
        sum(len(site["assemblies"]) for site in site_metadata),
    )
