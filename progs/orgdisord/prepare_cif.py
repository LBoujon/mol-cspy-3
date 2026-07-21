import copy
import logging
import re
from itertools import permutations, product
from pathlib import Path

import cspy
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
try:
    from scipy.integrate import simpson
except ImportError:
    from scipy.integrate import simps as simpson

from cspy import Crystal
from cspy.chem import Element
from cspy.db.clustering_loops import find_duplicates
from cspy.linalg.kabsch import centroid, kabsch_rotation_matrix, rmsd_points
from cspy.templating.analogue_production import get_iso_overlays

from .disord_utils import ordered_proxy_crystal, safe_label, write_disordered_cif
from .existing_disorder import annotate_existing_disorder_cif

LOG = logging.getLogger("cspy.progs.orgdisord.prepare_cif")

PXRD_METHOD = "cdtw_cos"
PXRD_COS_THRESHOLD = 0.80
PXRD_CDTW_THRESHOLD = 10.0


def calculate_pxrd_descriptor(crystal, atom_records, dx=0.02):

    proxy = ordered_proxy_crystal(crystal, atom_records)
    pp = proxy.calculate_powder_pattern()

    if pp is None or pp.pattern is None:
        return None

    x = np.asarray(pp.pattern, dtype=float)
    area = simpson(x, dx=dx)

    if area <= 0 or not np.isfinite(area):
        return None

    return x / area


def make_ordered_proxy_records(atom_records, target_assembly, target_group):
    assemblies = sorted({row["disorder_assembly"] for row in atom_records})

    default_group = {}

    for assembly in assemblies:
        groups = sorted({
            int(row["disorder_group"])
            for row in atom_records
            if row["disorder_assembly"] == assembly
        })
        default_group[assembly] = groups[0]

    selected = []

    for row in atom_records:
        assembly = row["disorder_assembly"]
        group = int(row["disorder_group"])

        if assembly == target_assembly:
            keep = group == int(target_group)
        else:
            keep = group == default_group[assembly]

        if keep:
            new_row = copy.deepcopy(row)
            new_row["occupancy"] = 1.0
            selected.append(new_row)

    return selected


def deduplicate_disorder_groups_by_pxrd(
    crystal,
    atom_records,
    method=PXRD_METHOD,
    cos_thresh=PXRD_COS_THRESHOLD,
    cdtw_thresh=PXRD_CDTW_THRESHOLD,
):

    keep_by_assembly = {}
    assemblies = sorted({
        row["disorder_assembly"]
        for row in atom_records
        if row["disorder_assembly"] != "."
    })

    for assembly in assemblies:
        keep_by_assembly[assembly] = set()

        components = sorted({
            row["component"]
            for row in atom_records
            if row["disorder_assembly"] == assembly
        })

        for component in components:
            groups = sorted({
                int(row["disorder_group"])
                for row in atom_records
                if row["disorder_assembly"] == assembly
                and row["component"] == component
            })

            if len(groups) <= 1:
                keep_by_assembly[assembly].update(groups)
                continue

            ids = []
            xs = []

            for group in groups:
                proxy_records = make_ordered_proxy_records(
                    atom_records,
                    target_assembly=assembly,
                    target_group=group,
                )

                xrd = calculate_pxrd_descriptor(
                    crystal,
                    proxy_records,
                )

                if xrd is None:
                    LOG.warning(
                        "PXRD calculation failed for %s component %s group %s; keeping it.",
                        assembly,
                        component,
                        group,
                    )
                    keep_by_assembly[assembly].add(group)
                    continue

                ids.append(group)
                xs.append(xrd)

            if len(ids) <= 1:
                keep_by_assembly[assembly].update(ids)
                continue

            # No energies/densities avaliable needs to be based on PXRD only
            eds = np.zeros((len(ids), 2), dtype=float)
            xs = np.asarray(xs, dtype=float)

            duplicate_to = find_duplicates(
                eds,
                xs,
                method=method,
                ethresh=1.0e9,
                dthresh=1.0e9,
                cdtw_thresh=cdtw_thresh,
                cos_thresh=cos_thresh,
            )

            for i, parent in enumerate(duplicate_to):
                group = ids[i]

                if parent == -1:
                    keep_by_assembly[assembly].add(group)
                else:
                    parent_group = ids[parent]
                    LOG.info(
                        "PXRD duplicate in %s component %s: removing group %s as duplicate of group %s",
                        assembly,
                        component,
                        group,
                        parent_group,
                    )

    # Rebuild records and renumber groups within each assembly.
    new_records = []

    for row in atom_records:
        if row["disorder_assembly"] == ".":
            new_records.append(copy.deepcopy(row))

    for assembly in assemblies:
        kept_groups = sorted(keep_by_assembly[assembly])

        if not kept_groups:
            continue

        group_map = {
            old_group: new_group
            for new_group, old_group in enumerate(kept_groups, start=1)
        }

        new_occupancy = 1.0 / len(kept_groups)

        for row in atom_records:
            if row["disorder_assembly"] != assembly:
                continue

            old_group = int(row["disorder_group"])

            if old_group not in group_map:
                continue

            new_row = copy.deepcopy(row)
            new_row["disorder_group"] = str(group_map[old_group])
            new_row["occupancy"] = new_occupancy
            new_records.append(new_row)

    return new_records


def parse_component_specs(component_args):
    """
    Parse:
        --component A1 host
        --component A2 guest.xyz
        --component A3 guest.res

    Returns:
        [("A1", "host"), ("A2", Path("guest.xyz")), ...]
    """

    specs = []

    for label, source in component_args:
        label = str(label)

        if not re.fullmatch(r"A\d+", label):
            raise ValueError(
                f"Component label should look like A1, A2, A3, ... Got: {label}"
            )

        if source == "host":
            specs.append((label, "host"))
        else:
            source = Path(source)

            if not source.exists():
                raise FileNotFoundError(f"Component file not found: {source}")

            specs.append((label, source))
            
    labels = [label for label, _ in specs]

    if len(labels) != len(set(labels)):
        raise ValueError(f"Duplicate component labels found: {labels}")
    
    host_labels = [label for label, source in specs if source == "host"]

    if len(host_labels) > 1:
        raise ValueError(
            f"Only one host component is currently supported. Got: {host_labels}"
        )

    return specs


def molecule_match_rmsd(mol_a, mol_b, heavy_only=True, atom_match_mode="heavy"):
    """
    Compare two already-overlaid molecules and return the best RMSD
    """
    nums_a = np.asarray(mol_a.atomic_numbers, dtype=int)
    nums_b = np.asarray(mol_b.atomic_numbers, dtype=int)

    xyz_a = np.asarray(mol_a.positions, dtype=float)
    xyz_b = np.asarray(mol_b.positions, dtype=float)

    if heavy_only:
        keep_a = nums_a != 1
        keep_b = nums_b != 1

        nums_a = nums_a[keep_a]
        nums_b = nums_b[keep_b]
        xyz_a = xyz_a[keep_a]
        xyz_b = xyz_b[keep_b]

    if len(xyz_a) != len(xyz_b):
        return np.inf

    if atom_match_mode == "heavy":
        # Treat all heavy atoms as equivalent.
        match_a = np.ones(len(nums_a), dtype=int)
        match_b = np.ones(len(nums_b), dtype=int)

    elif atom_match_mode == "exact":
        match_a = nums_a
        match_b = nums_b

    elif atom_match_mode == "any":
        match_a = np.ones(len(nums_a), dtype=int)
        match_b = np.ones(len(nums_b), dtype=int)

    else:
        raise ValueError("atom_match_mode must be 'exact', 'heavy', or 'any'")

    # Compare centred coordinates so tiny translation differences do not matter
    xyz_a = xyz_a - xyz_a.mean(axis=0)
    xyz_b = xyz_b - xyz_b.mean(axis=0)

    # Penalise non-compatible atom matches.
    d = cdist(xyz_a, xyz_b)

    for i in range(len(match_a)):
        for j in range(len(match_b)):
            if match_a[i] != match_b[j]:
                d[i, j] = 1.0e6

    row_idx, col_idx = linear_sum_assignment(d)

    if np.any(d[row_idx, col_idx] > 1.0e5):
        return np.inf

    diff = xyz_a[row_idx] - xyz_b[col_idx]

    return float(np.sqrt(np.mean(np.sum(diff**2, axis=1))))


def generate_isomorphic_mappings(
    host_mol,
    guest_mol,
    host_indices=None,
    guest_indices=None,
    atom_match_mode="heavy",
):
    host_indices = host_indices or [
        i for i, number in enumerate(host_mol.atomic_numbers)
        if int(number) != 1
    ]

    guest_indices = guest_indices or [
        i for i, number in enumerate(guest_mol.atomic_numbers)
        if int(number) != 1
    ]

    if len(host_indices) != len(guest_indices):
        raise ValueError(
            "Automatic heavy-atom mapping failed because the host and guest "
            f"have different numbers of heavy atoms: "
            f"{len(host_indices)} vs {len(guest_indices)}. "
            "Provide --host-map and --guest-map manually, or use principal-axis overlay."
        )

    possibilities = get_iso_overlays(
        host_mol,
        guest_mol,
        list(host_indices),
        list(guest_indices),
        [],
        atom_match_mode,
    )

    unique = []
    seen = set()

    for host_map, guest_map in possibilities:
        key = (tuple(host_map), tuple(guest_map))

        if key in seen:
            continue

        seen.add(key)
        unique.append((list(host_map), list(guest_map)))

    return unique


def parse_int_csv(text, as_set=False):
    if text is None:
        return None

    values = [
        int(item.strip())
        for item in str(text).split(",")
        if item.strip()
    ]

    if as_set:
        return set(values)

    return values


def load_molecule_from_file(path, molecule_index=0):

    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".xyz":
        return cspy.Molecule.from_xyz_file(str(path))

    if suffix in [".res", ".cif"]:
        crystal = Crystal.load(str(path))
        mols = crystal.asym_mols()

        if molecule_index >= len(mols):
            raise IndexError(
                f"Requested molecule index {molecule_index}, but {path} has "
                f"only {len(mols)} asymmetric-unit molecules."
            )

        return mols[molecule_index]

    raise ValueError(f"Unsupported molecule file type: {path}")


def overlay_score(reference, trial):

    ref_nums = np.asarray(reference.atomic_numbers, dtype=int)
    trial_nums = np.asarray(trial.atomic_numbers, dtype=int)

    ref_xyz = np.asarray(reference.positions, dtype=float)
    trial_xyz = np.asarray(trial.positions, dtype=float)

    scores = []

    for number, pos in zip(ref_nums, ref_xyz):
        if number == 1:
            continue

        keep = trial_nums == number

        if not np.any(keep):
            # If the guest does not have this element, fall back to any heavy atom
            # but add a penalty so exact element matches are preferred.
            keep = trial_nums != 1
            penalty = 1.0
        else:
            penalty = 0.0

        d = np.linalg.norm(trial_xyz[keep] - pos, axis=1)
        scores.append(float(np.min(d)) + penalty)

    if not scores:
        return np.inf

    return float(np.mean(scores))


def molecule_skeleton(mol):
    """Return heavy-atom coordinates centred on their centroid."""
    atomic_numbers = np.asarray(mol.atomic_numbers, dtype=int)
    coords = np.asarray(mol.positions, dtype=float)[atomic_numbers != 1]

    if not len(coords):
        raise ValueError("Molecule has no non-H atoms for overlay.")

    return coords - centroid(coords)


def reorder_atoms_by_distance(coords_a, coords_b):

    distances = cdist(coords_a, coords_b)
    row_idx, col_idx = linear_sum_assignment(distances)

    ordered = np.zeros_like(coords_b)
    ordered[row_idx] = coords_b[col_idx]

    return ordered, col_idx


def overlay_by_principal_axes(reference, moving):

    ref_axes = np.asarray(reference.axes(method="moi"), dtype=float).T
    mov_axes = np.asarray(moving.axes(method="moi"), dtype=float).T
    
    if np.linalg.det(ref_axes) < 0:
        ref_axes[:, -1] *= -1.0

    if np.linalg.det(mov_axes) < 0:
        mov_axes[:, -1] *= -1.0

    ref_centroid = reference.center_of_mass
    mov_centroid = moving.center_of_mass

    best_positions = None
    best_score = np.inf

    # Try axis permutations because principal axes can be swapped
    # for near-symmetric molecules.
    for perm in permutations([0, 1, 2]):
        P = np.eye(3)[:, perm]

        # Try sign flips because principal axes have arbitrary direction.
        for signs in product([-1, 1], repeat=3):
            S = np.diag(signs)

            # Avoid improper rotations/reflections unless you deliberately want them.
            R = mov_axes @ P @ S @ ref_axes.T

            if np.linalg.det(R) < 0:
                continue

            trial = copy.deepcopy(moving)
            trial.positions = (trial.positions - mov_centroid) @ R + ref_centroid

            score = overlay_score(reference, trial)

            if score < best_score:
                best_score = score
                best_positions = trial.positions.copy()

    if best_positions is None:
        raise RuntimeError("Principal-axis overlay failed to find a valid rotation.")

    out = copy.deepcopy(moving)
    out.positions = best_positions

    return out, best_score


def overlay_molecule(reference, moving, method="auto", reference_indices=None, moving_indices=None,):
    """
    Overlay moving molecule onto reference molecule.

    method:
        auto      -> Kabsch if possible, otherwise principal-axis fallback
        kabsch    -> require equal heavy-atom counts
        principal -> principal-axis fallback
        centroid  -> translation only
    """
    reference = copy.deepcopy(reference)
    moving = copy.deepcopy(moving)

    ref_centroid = centroid(reference.positions)
    mov_centroid = centroid(moving.positions)

    if method == "centroid":
        moving.positions = moving.positions - mov_centroid + ref_centroid
        return moving, np.nan

    elif method == "principal":
        return overlay_by_principal_axes(reference, moving)
    
    elif method == "mapped-kabsch":
        return overlay_by_mapped_kabsch(
            reference,
            moving,
            reference_indices=reference_indices,
            moving_indices=moving_indices,
        )

    ref_coords = molecule_skeleton(reference)
    mov_coords = molecule_skeleton(moving)

    if len(ref_coords) != len(mov_coords):
        if method == "kabsch":
            raise ValueError(
                "Kabsch overlay requires the same number of heavy atoms. "
                f"Reference has {len(ref_coords)}, moving has {len(mov_coords)}."
            )

        LOG.warning(
            "Heavy-atom counts differ (%d vs %d). Falling back to principal-axis overlay.",
            len(ref_coords),
            len(mov_coords),
        )
        return overlay_by_principal_axes(reference, moving)

    mov_ordered, _ = reorder_atoms_by_distance(
        ref_coords,
        mov_coords,
    )

    rotation = kabsch_rotation_matrix(
        mov_ordered,
        ref_coords,
    )

    out = copy.deepcopy(moving)
    out.positions = (
        (out.positions - mov_centroid)
        @ rotation
        + ref_centroid
    )

    out_coords = molecule_skeleton(out)

    out_ordered, _ = reorder_atoms_by_distance(
        ref_coords,
        out_coords,
    )

    fit_rmsd = rmsd_points(
        ref_coords,
        out_ordered,
        reorient=False,
    )

    return out, float(fit_rmsd)


def overlay_by_mapped_kabsch(
    reference,
    moving,
    reference_indices,
    moving_indices,
):
    """
    Overlay a molecule using explicitly corresponding atom indices.

    ``reference_indices[k]`` is paired with
    ``moving_indices[k]``.
    """
    reference = copy.deepcopy(reference)
    moving = copy.deepcopy(moving)

    reference_positions = np.asarray(
        reference.positions,
        dtype=float,
    )
    moving_positions = np.asarray(
        moving.positions,
        dtype=float,
    )

    reference_indices = np.asarray(
        reference_indices,
        dtype=int,
    )
    moving_indices = np.asarray(
        moving_indices,
        dtype=int,
    )

    reference_selected = reference_positions[
        reference_indices
    ]
    moving_selected = moving_positions[
        moving_indices
    ]

    reference_centroid = centroid(
        reference_selected
    )
    moving_centroid = centroid(
        moving_selected
    )

    reference_centred = (
        reference_selected - reference_centroid
    )
    moving_centred = (
        moving_selected - moving_centroid
    )

    rotation = kabsch_rotation_matrix(
        moving_centred,
        reference_centred,
    )

    out = copy.deepcopy(moving)
    out.positions = (
        (moving_positions - moving_centroid)
        @ rotation
        + reference_centroid
    )

    fitted_selected = out.positions[
        moving_indices
    ]

    fit_rmsd = rmsd_points(
        reference_selected,
        fitted_selected,
        reorient=False,
    )

    return out, float(fit_rmsd)


def get_symmetry_operations(mol):

    result = mol.get_symmetry_equivalent_atoms()

    if isinstance(result, tuple):
        equivalent_atoms, symmetry_operations = result
    else:
        equivalent_atoms = result
        symmetry_operations = getattr(mol, "symmetry_operations_with_mappings", [])

    proper_ops = []

    for op in symmetry_operations:
        det = float(op.get("determinant", np.linalg.det(op["rotation_matrix"])))

        # ignore reflections/inversions as may change chiral molecuels 
        if det > 0:
            proper_ops.append(op)

    if not proper_ops:
        proper_ops = [
            {
                "operation_index": 0,
                "operation_string": "x, y, z",
                "mapping": list(range(len(mol))),
                "rotation_matrix": np.eye(3),
                "translation_vector": np.zeros(3),
                "affine_matrix": np.eye(4),
                "determinant": 1.0,
            }
        ]

    return proper_ops


def apply_symmetry(placed_mol, reference_mol, operation):

    out = copy.deepcopy(placed_mol)

    rotation_matrix = np.asarray(operation["rotation_matrix"], dtype=float)
    translation_vector = np.asarray(
        operation.get("translation_vector", np.zeros(3)),
        dtype=float,
    )

    out.rotate_about_point(
        point=reference_mol.center_of_mass,
        alpha=None,
        beta=None,
        gamma=None,
        rotation_matrix=rotation_matrix,
    )

    out.translate(translation_vector)

    return out


def principal_symmetry_overlays(
    reference,
    moving,
    atom_match_mode="heavy",
    duplicate_rmsd_threshold=0.05,
):
    """
    Place moving molecule by principal axes, then use host molecular symmetry
    operations to generate possible equivalent-fitting guest orientations.
    """
    base_mol, base_score = overlay_by_principal_axes(reference, moving)

    host_ops = get_symmetry_operations(reference)
    guest_ops = get_symmetry_operations(moving)

    LOG.info(
        "Principal-symmetry overlay: host has %d operations; guest has %d operations.",
        len(host_ops),
        len(guest_ops),
    )

    raw_candidates = [
        {
            "mol": base_mol,
            "score": base_score,
            "operation_index": 0,
            "operation_string": "principal_base",
        }
    ]

    for op in host_ops:
        trial = apply_symmetry(
            placed_mol=base_mol,
            reference_mol=reference,
            operation=op,
        )

        raw_candidates.append(
            {
                "mol": trial,
                "score": overlay_score(reference, trial),
                "operation_index": op.get("operation_index", -1),
                "operation_string": op.get("operation_string", "unknown"),
            }
        )

    kept = []

    for candidate in sorted(raw_candidates, key=lambda x: x["score"]):
        duplicate = False

        for previous in kept:
            dup_rmsd = molecule_match_rmsd(
                candidate["mol"],
                previous["mol"],
                heavy_only=True,
                atom_match_mode=atom_match_mode,
            )

            if dup_rmsd < duplicate_rmsd_threshold:
                duplicate = True
                LOG.info(
                    "Skipping duplicate principal-symmetry candidate %s with RMSD %.6f",
                    candidate["operation_string"],
                    dup_rmsd,
                )
                break

        if duplicate:
            continue

        kept.append(candidate)

    return kept


def make_atom_records_for_site(
    crystal,
    site_index,
    site_mol,
    component_specs,
    component_molecules,
    overlay_method="auto",
    host_map=None,
    guest_map=None,
    all_isomorphic_overlays=False,
    atom_match_mode="heavy",
    duplicate_detection="rmsd",
    assembly_label=None,
):
    records = []

    assembly = assembly_label or f"S{site_index + 1:03d}"

    alternatives = []

    for component_label, source in component_specs:
        if source == "host":
            alternatives.append(
                {
                    "component": component_label,
                    "source": source,
                    "mol": copy.deepcopy(site_mol),
                    "host_map": None,
                    "guest_map": None,
                    "mapping_index": 1,
                    "raw_mapping_index": 1,
                    "n_mappings": 1,
                    "overlay_rmsd": 0.0,
                    "symmetry_operation": None,
                }
            )
            continue

        guest_mol = component_molecules[component_label]

        if overlay_method == "mapped-kabsch" and all_isomorphic_overlays:
            mappings = generate_isomorphic_mappings(
                host_mol=site_mol,
                guest_mol=guest_mol,
                host_indices=host_map,
                guest_indices=guest_map,
                atom_match_mode=atom_match_mode,
            )

            if not mappings:
                raise RuntimeError(
                    f"No isomorphic overlays found for component {component_label} "
                    f"at site {site_index}."
                )

            component_alternatives = []
            unique_guest_overlays = []

            for raw_mapping_index, (this_host_map, this_guest_map) in enumerate(
                mappings,
                start=1,
            ):
                mol, overlay_rmsd = overlay_molecule(
                    site_mol,
                    guest_mol,
                    method=overlay_method,
                    reference_indices=this_host_map,
                    moving_indices=this_guest_map,
                )

                duplicate = False

                if duplicate_detection == "rmsd":
                    for previous_mol in unique_guest_overlays:
                        dup_rmsd = molecule_match_rmsd(
                            mol,
                            previous_mol,
                            heavy_only=True,
                            atom_match_mode=atom_match_mode,
                        )

                        if dup_rmsd < 0.05:
                            duplicate = True
                            LOG.info(
                                "Skipping duplicate overlay for component %s at site %d: "
                                "raw mapping %d duplicates an existing overlay with RMSD %.6f",
                                component_label,
                                site_index,
                                raw_mapping_index,
                                dup_rmsd,
                            )
                            break

                if duplicate:
                    continue

                unique_guest_overlays.append(copy.deepcopy(mol))

                component_alternatives.append(
                    {
                        "component": component_label,
                        "source": source,
                        "mol": mol,
                        "host_map": this_host_map,
                        "guest_map": this_guest_map,
                        "mapping_index": len(component_alternatives) + 1,
                        "raw_mapping_index": raw_mapping_index,
                        "n_mappings": None,
                        "overlay_rmsd": overlay_rmsd,
                        "symmetry_operation": None,
                    }
                )

            n_kept_mappings = len(component_alternatives)

            for alternative in component_alternatives:
                alternative["n_mappings"] = n_kept_mappings

                LOG.info(
                    "Site %s component %s raw mapping %d kept as unique mapping %d/%d, "
                    "overlay RMSD %.6f",
                    assembly,
                    component_label,
                    alternative["raw_mapping_index"],
                    alternative["mapping_index"],
                    n_kept_mappings,
                    alternative["overlay_rmsd"],
                )

                alternatives.append(alternative)

        elif overlay_method == "principal":
            candidates = principal_symmetry_overlays(
                reference=site_mol,
                moving=guest_mol,
                atom_match_mode=atom_match_mode,
            )

            n_kept_mappings = len(candidates)

            for candidate_index, candidate in enumerate(candidates, start=1):
                alternatives.append(
                    {
                        "component": component_label,
                        "source": source,
                        "mol": candidate["mol"],
                        "host_map": None,
                        "guest_map": None,
                        "mapping_index": candidate_index,
                        "raw_mapping_index": candidate.get("operation_index", candidate_index),
                        "n_mappings": n_kept_mappings,
                        "overlay_rmsd": candidate["score"],
                        "symmetry_operation": candidate.get("operation_string", "unknown"),
                    }
                )

        else:
            mol, overlay_rmsd = overlay_molecule(
                site_mol,
                guest_mol,
                method=overlay_method,
                reference_indices=host_map,
                moving_indices=guest_map,
            )

            alternatives.append(
                {
                    "component": component_label,
                    "source": source,
                    "mol": mol,
                    "host_map": host_map,
                    "guest_map": guest_map,
                    "mapping_index": 1,
                    "raw_mapping_index": 1,
                    "n_mappings": 1,
                    "overlay_rmsd": overlay_rmsd,
                    "symmetry_operation": None,
                }
            )

    n_groups = len(alternatives)

    occupancy = 1.0 / n_groups

    for group_index, alternative in enumerate(alternatives, start=1):
        component_label = alternative["component"]
        mol = alternative["mol"]
        overlay_rmsd = alternative["overlay_rmsd"]
        mapping_index = int(alternative.get("mapping_index", 1))

        frac = crystal.to_fractional(np.asarray(mol.positions, dtype=float))

        for atom_index, (number, xyz) in enumerate(
            zip(mol.atomic_numbers, frac),
            start=1,
        ):
            symbol = Element.from_atomic_number(int(number)).symbol

            # Include mapping index in the label if there are multiple guest mappings.
            mapping_index = alternative.get("mapping_index", 1)

            if mapping_index > 1:
                label = safe_label(
                    f"{component_label}m{mapping_index}_{assembly}_{symbol}{atom_index}"
                )
            else:
                label = safe_label(
                    f"{component_label}_{assembly}_{symbol}{atom_index}"
                )

            records.append(
                {
                    "label": label,
                    "symbol": symbol,
                    "fract_x": float(xyz[0]),
                    "fract_y": float(xyz[1]),
                    "fract_z": float(xyz[2]),
                    "occupancy": occupancy,
                    "disorder_assembly": assembly,
                    "disorder_group": "." if assembly == "." else str(group_index),
                    "component": component_label,
                    "source": str(alternative.get("source", "")),
                    "mapping_index": int(alternative.get("mapping_index", 1)),
                    "raw_mapping_index": int(alternative.get("raw_mapping_index", 1)),
                    "n_mappings": int(alternative.get("n_mappings", 1)),
                    "overlay_rmsd": overlay_rmsd,
                    "symmetry_operation": alternative.get("symmetry_operation"),
                }
            )

    return records


def write_overlay_report(output_file, atom_records):
    rows = {}

    for row in atom_records:
        key = (
            row["disorder_assembly"],
            row["disorder_group"],
            row["component"],
            row.get("mapping_index", 1),
        )
        rows[key] = row["overlay_rmsd"]

    report = Path(output_file).with_suffix(".overlay_report.csv")

    with open(report, "w") as handle:
        handle.write("assembly,group,component,mapping_index,overlay_rmsd\n")

        for (assembly, group, component, mapping_index), value in sorted(rows.items()):
            handle.write(
                f"{assembly},{group},{component},{mapping_index},{value}\n"
            )

    LOG.info("Wrote overlay report: %s", report)


def run_prepare_cif(args):
    host_file = Path(args.host)

    if not host_file.exists():
        raise FileNotFoundError(f"Host file not found: {host_file}")
    
    if getattr(args, "existing_disorder", False):
        annotate_existing_disorder_cif(
            input_cif=host_file,
            output_cif=args.output,
            group_components=getattr(args, "group_components", None),
            joint_states=getattr(args, "joint_state", None),
            multi_assembly_mode=getattr(
                args,
                "multi_assembly_mode",
                "auto",
            ),
            occupancy_tolerance=getattr(
                args,
                "joint_occupancy_tolerance",
                0.02,
            ),
            existing_component=getattr(
                args,
                "existing_component",
                "A1",
            ),
        )
        return

    if not args.component:
        raise ValueError(
            "prepare-cif requires --component unless --existing-disorder is used."
        )

    host_crystal = Crystal.load(str(host_file))

    if args.supercell is not None:
        host_crystal = host_crystal.as_P1_supercell(tuple(args.supercell))
    else:
        # Safer for disorder enumeration: use explicit P1 molecules.
        host_crystal = host_crystal.as_P1()

    host_mols = host_crystal.asym_mols()

    if not host_mols:
        raise RuntimeError("No asymmetric-unit molecules found in host crystal.")

    component_specs = parse_component_specs(args.component)
    
    host_components = [
        label
        for label, source in component_specs
        if source == "host"
    ]

    if len(host_components) != 1:
        raise ValueError(
            "Provide a host component, e.g. "
            "--component A1 host --component A2 guest.xyz"
        )

    host_map = parse_int_csv(args.host_map)
    guest_map = parse_int_csv(args.guest_map)

    component_molecules = {}

    for component_label, source in component_specs:
        if source == "host":
            continue

        component_molecules[component_label] = load_molecule_from_file(
            source,
            molecule_index=args.guest_molecule_index,
        )

    atom_records = []

    for site_index, site_mol in enumerate(host_mols):
        assembly_label = f"S{site_index + 1:03d}"

        site_records = make_atom_records_for_site(
            crystal=host_crystal,
            site_index=site_index,
            site_mol=site_mol,
            component_specs=component_specs,
            component_molecules=component_molecules,
            overlay_method=args.overlay_method,
            host_map=host_map,
            guest_map=guest_map,
            all_isomorphic_overlays=args.all_isomorphic_overlays,
            atom_match_mode=args.atom_match_mode,
            duplicate_detection=args.duplicate_detection,
            assembly_label=assembly_label,
        )

        atom_records.extend(site_records)

    if args.duplicate_detection == "pxrd":
        atom_records = deduplicate_disorder_groups_by_pxrd(host_crystal, atom_records)

    write_disordered_cif(args.output, host_crystal, atom_records)

    if args.overlay_report:
        write_overlay_report(args.output, atom_records)
