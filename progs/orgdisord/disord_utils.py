import copy
import json
import logging
import os
import re
import shutil
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from cspy import Crystal
from cspy.crystal import AsymmetricUnit
from cspy.db.datastore import CspDataStore
from cspy.formats.cif import Cif

LOG = logging.getLogger("cspy.progs.orgdisord.disord_utils")

CRYSTAL_FIELDS = (
    "id",
    "spacegroup",
    "energy",
    "density",
    "molecule_id",
    "file_content",
)

TRIAL_FIELDS = (
    "id",
    "minimization_step",
    "trial_number",
    "valid",
    "minimization_time",
    "metadata",
)


def ordered_proxy_crystal(
    crystal: Crystal,
    atom_records: Iterable[Mapping[str, Any]],
    title: str = "ordered_proxy",
) -> Crystal:
    """Build an ordered crystal from disorder-style atom records."""
    records = [
        {
            "label": row["label"],
            "element": row["symbol"],
            "position": [row["fract_x"], row["fract_y"], row["fract_z"]],
            "occupation": 1.0,
        }
        for row in atom_records
    ]
    return Crystal(
        copy.deepcopy(crystal.unit_cell),
        copy.deepcopy(crystal.space_group),
        AsymmetricUnit.from_records(records),
        titl=title,
    )


def iter_cif_loops(lines: Iterable[str]):
    """Yield ``(headers, rows)`` for loops in CIF text."""
    headers: list[str] = []
    rows: list[str] = []
    in_loop = False

    for raw_line in [*lines, "loop_"]:
        line = raw_line.strip()
        if line == "loop_":
            if in_loop and headers:
                yield headers, rows
            in_loop = True
            headers, rows = [], []
            continue
        if not in_loop or not line or line.startswith("#"):
            continue
        if line.startswith("_") and not rows:
            headers.append(line.split()[0])
            continue
        if line.startswith("_"):
            if headers:
                yield headers, rows
            in_loop = False
            headers, rows = [], []
            continue
        rows.append(raw_line)


def cif_tokens(line: object) -> list[str]:
    """Split one simple CIF row while retaining whitespace inside quotes."""
    tokens: list[str] = []
    current: list[str] = []
    quote: str | None = None

    for char in str(line).strip():
        if quote is not None:
            if char == quote:
                quote = None
            else:
                current.append(char)
            continue
        if char in ("'", '"'):
            quote = char
        elif char.isspace():
            if current:
                tokens.append("".join(current))
                current = []
        else:
            current.append(char)

    if quote is not None:
        raise ValueError(f"Unterminated quoted CIF value in line: {line!r}")
    if current:
        tokens.append("".join(current))
    return tokens


def clean_cif_value(value: object) -> str:
    """Strip CIF quoting and map missing markers to an empty string."""
    cleaned = str(value).strip().strip("'\"")
    return "" if cleaned in {"", ".", "?"} else cleaned


def normalise_cif_key(name: object) -> str:
    """Return a case-insensitive key matching CSPy's ``Cif`` data model."""
    return str(name).lstrip("_").lower()


def parse_cif_document(
    lines: Iterable[str],
) -> dict[str, Any]:
    """Parse standard CIF data with CSPy's :class:`Cif` class.

    Raw lines are retained because the current CSPy CIF model intentionally
    discards comments, including the JSON comments used by cspy-disord.
    """
    source_lines = list(lines)
    cif = Cif.from_string("\n".join(source_lines))

    if not cif.data:
        raise ValueError("The CIF contains no data block.")

    block_name, raw_block = next(iter(cif.data.items()))
    block: dict[str, Any] = {}

    for key, value in raw_block.items():
        normalised = normalise_cif_key(key)

        if normalised in block:
            raise ValueError(
                "The CIF contains data names that differ only by case: "
                f"{key!r}."
            )

        block[normalised] = value

    return {
        "cif": cif,
        "block_name": block_name,
        "block": block,
        "lines": source_lines,
    }


def cif_block_value(
    block: Mapping[str, Any],
    *names: str,
    default: Any = None,
) -> Any:
    """Return the first matching value from a normalised CSPy CIF block."""
    for name in names:
        key = normalise_cif_key(name)

        if key in block:
            return block[key]

    return default


def required_cif_float(
    block: Mapping[str, Any],
    name: str,
) -> float:
    """Return one required scalar CIF value as a float."""
    value = cif_block_value(block, name)

    if value is None or isinstance(
        value,
        (list, tuple, np.ndarray),
    ):
        raise ValueError(
            f"Required scalar CIF value {name!r} was not found."
        )

    return parse_cif_number(value)


def cif_column(
    block: Mapping[str, Any],
    name: str,
    required: bool = True,
) -> list[Any] | None:
    """Return one loop column parsed by CSPy's :class:`Cif` class."""
    value = cif_block_value(block, name)

    if value is None:
        if required:
            raise ValueError(
                f"Required CIF loop column {name!r} was not found."
            )

        return None

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, (list, tuple)):
        return list(value)

    raise ValueError(
        f"CIF data name {name!r} is scalar, but a loop column "
        "was expected."
    )


def format_cif_token(value: object) -> str:
    """Format a token for a simple whitespace-separated CIF row."""
    text = str(value)
    if text in {".", "?"}:
        return text
    if not text:
        return "."
    if any(char.isspace() for char in text) or text.startswith(("_", "#", ";")):
        return "'" + text.replace("'", "''") + "'"
    return text


def parse_cif_number(value: object) -> float:
    """Parse a CIF number, removing a trailing standard uncertainty."""
    cleaned = clean_cif_value(value)
    if not cleaned:
        raise ValueError(f"Missing CIF numeric value: {value!r}")
    return float(re.sub(r"\([^)]*\)$", "", cleaned))


def format_cif_number(value: object) -> str:
    """Format a finite number for CIF output."""
    if value is None:
        return "?"
    number = float(value)
    if not np.isfinite(number):
        return "?"
    return f"{number:.8f}".rstrip("0").rstrip(".")


# Backwards-compatible name used by the original preparation modules.
cif_number = format_cif_number


def safe_label(text: object) -> str:
    """Return a CIF-label-safe representation of text."""
    return re.sub(r"[^A-Za-z0-9_]+", "_", str(text))


def proxy_records(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Convert parsed atom rows to the ordered-proxy record shape."""
    return [
        {
            "label": row["label"],
            "symbol": row["symbol"],
            "fract_x": row["fract_x"],
            "fract_y": row["fract_y"],
            "fract_z": row["fract_z"],
            "occupancy": 1.0,
        }
        for row in records
    ]


def unique_atom_label(label: object, used_labels: set[str]) -> str:
    """Return and reserve a label that is unique within ``used_labels``."""
    base = safe_label(label) or "X"
    candidate = base
    suffix = 2
    while candidate in used_labels:
        candidate = f"{base}_{suffix}"
        suffix += 1
    used_labels.add(candidate)
    return candidate


# Backwards-compatible private aliases.
_proxy_records = proxy_records
_unique_atom_label = unique_atom_label


def find_cif_scalar_value(
    lines: Iterable[str],
    tag_names: Iterable[str],
) -> str | None:
    """Return the first matching scalar CIF value."""
    wanted = {str(tag).lower() for tag in tag_names}
    source = list(lines)

    for index, raw_line in enumerate(source):
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        tokens = cif_tokens(stripped)
        if not tokens or tokens[0].lower() not in wanted:
            continue
        if len(tokens) > 1:
            return " ".join(tokens[1:])

        next_index = index + 1
        while next_index < len(source):
            candidate = source[next_index].strip()
            if candidate and not candidate.startswith("#"):
                break
            next_index += 1
        if next_index >= len(source):
            return None

        candidate = source[next_index].strip()
        lowered = candidate.lower()
        if (
            candidate.startswith("_")
            or lowered == "loop_"
            or lowered.startswith("data_")
            or lowered.startswith("save_")
            or candidate.startswith(";")
        ):
            return None
        values = cif_tokens(candidate)
        return " ".join(values) if values else None
    return None


def cif_scalar_float(lines: Iterable[str], tag: str) -> float:
    """Read one required scalar CIF number."""
    value = find_cif_scalar_value(lines, [tag])
    if value is None:
        raise ValueError(f"Required CIF scalar {tag} was not found.")
    try:
        return parse_cif_number(value)
    except ValueError as exc:
        raise ValueError(
            f"Could not parse CIF scalar {tag}={value!r} as a number."
        ) from exc


def extract_cif_symmetry_operations(lines: Iterable[str]) -> list[str]:
    """Read symmetry operation strings from either common CIF header."""
    candidates = (
        "_space_group_symop_operation_xyz",
        "_symmetry_equiv_pos_as_xyz",
    )
    for headers, rows in iter_cif_loops(lines):
        header = next((name for name in candidates if name in headers), None)
        if header is None:
            continue
        operation_index = headers.index(header)
        operation_is_last = operation_index == len(headers) - 1
        operations: list[str] = []
        for row in rows:
            tokens = cif_tokens(row)
            if len(tokens) <= operation_index:
                continue
            raw = (
                " ".join(tokens[operation_index:])
                if operation_is_last
                else tokens[operation_index]
            )
            operation = re.sub(r"\s+", "", clean_cif_value(raw))
            if operation:
                operations.append(operation)
        if operations:
            return operations
    return []


def find_atom_site_loop_bounds(
    lines: Iterable[str],
    *,
    stop_prefixes: Sequence[str] = (),
) -> tuple[int, list[str], int, int]:
    """Return start/header/row bounds for the CIF atom-site loop."""
    source = list(lines)
    index = 0
    while index < len(source):
        if source[index].strip().lower() != "loop_":
            index += 1
            continue
        loop_start = index
        index += 1
        headers: list[str] = []
        while index < len(source):
            text = source[index].strip()
            if text.startswith("_"):
                headers.append(text.split()[0])
                index += 1
                continue
            if not text or text.startswith("#"):
                index += 1
                continue
            break
        if "_atom_site_label" not in headers:
            continue
        row_start = index
        while index < len(source):
            text = source[index].strip()
            lowered = text.lower()
            if (
                lowered == "loop_"
                or lowered.startswith("data_")
                or text.startswith("_")
                or any(text.startswith(prefix) for prefix in stop_prefixes)
            ):
                break
            index += 1
        return loop_start, headers, row_start, index
    raise ValueError("Could not find the CIF atom-site loop.")


def normalise_element_symbol(value: object, label: object = "") -> str:
    """Infer and normalise an element symbol from a CIF value or label."""
    match = re.match(r"([A-Za-z]{1,2})", clean_cif_value(value))
    if match is None:
        match = re.match(r"([A-Za-z]{1,2})", clean_cif_value(label))
    if match is None:
        raise ValueError(f"Could not determine the element for CIF label {label!r}.")
    symbol = match.group(1)
    return symbol[0].upper() + symbol[1:].lower()


def normalise_disorder_label(value: object) -> str:
    """Normalise integer-like assembly/group labels to integer strings."""
    cleaned = clean_cif_value(value)
    if not cleaned:
        return ""
    try:
        number = float(cleaned)
    except ValueError:
        return cleaned
    return str(int(number)) if number.is_integer() else cleaned


def read_json_comment_records(path: str | Path, prefix: str) -> list[Any]:
    """Read JSON payloads from comments beginning with ``prefix``."""
    records: list[Any] = []
    with Path(path).open("r", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped.startswith(prefix):
                records.append(json.loads(stripped.removeprefix(prefix).strip()))
    return records


def read_orientation_degeneracies_from_cif(
    cif_file: str | Path | None,
) -> dict[str, float]:
    """Read component orientation-degeneracy metadata from a prepared CIF."""
    if cif_file is None or not Path(cif_file).exists():
        return {}
    records = read_json_comment_records(
        cif_file,
        "# cspy_disord_orientation_degeneracies ",
    )
    if not records:
        return {}
    return {str(component): float(factor) for component, factor in records[-1].items()}


def read_group_map_from_cif(
    cif_file: str | Path,
) -> dict[tuple[str, str], dict[str, Any]]:
    """Read prepared-CIF disorder-group metadata."""
    group_map: dict[tuple[str, str], dict[str, Any]] = {}
    for row in read_json_comment_records(cif_file, "# cspy_disord_group_map "):
        assembly = normalise_disorder_label(row["assembly"])
        group = normalise_disorder_label(row["group"])
        key = (assembly, group)
        if key in group_map:
            raise ValueError(
                "Duplicate prepared-CIF group-map entry for "
                f"assembly={assembly}, group={group}."
            )
        group_map[key] = {
            "assembly": assembly,
            "group": group,
            "component": str(row["component"]),
            "mapping_index": int(row.get("mapping_index", 1)),
            "raw_mapping_index": int(row.get("raw_mapping_index", 1)),
            "n_mappings": int(row.get("n_mappings", 1)),
            "overlay_rmsd": row.get("overlay_rmsd"),
            "original_joint_state": row.get("original_joint_state"),
            "original_joint_occupancy": row.get("original_joint_occupancy"),
            "physical_site": row.get("physical_site"),
            "site_role": row.get("site_role"),
            "original_assembly": row.get("original_assembly"),
        }
    if not group_map:
        raise ValueError(
            "No cspy-disord group-map comments were found in "
            f"{cif_file}. Run cspy-disord prepare-cif first."
        )
    return group_map


def read_site_maps_from_cif(cif_file: str | Path) -> dict[str, dict[str, Any]]:
    """Read fragment-preserving physical-site metadata from a prepared CIF."""
    site_maps: dict[str, dict[str, Any]] = {}
    for row in read_json_comment_records(cif_file, "# cspy_disord_site_map "):
        site = str(row["site"])
        if site in site_maps:
            raise ValueError(f"Duplicate cspy-disord site metadata for {site}.")
        site_maps[site] = row
    return site_maps


def rows_to_columns(
    rows: Sequence[Mapping[str, Any]],
    fields: Sequence[str],
) -> dict[str, list[Any]]:
    """Convert row dictionaries into the column-oriented datastore shape."""
    return {field: [row[field] for row in rows] for field in fields}

def query_rows(db, query, *args):
    """
    Execute a query and materialise all rows before the database is closed.
    """
    result = db.query(query, *args)

    if hasattr(result, "fetchall"):
        return result.fetchall()

    return list(result)


def write_structure_rows(
    output_db: str | Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    description: str | None = None,
) -> None:
    """Write crystal and trial-structure row dictionaries to a CSPy database."""
    if not rows:
        return
    db = CspDataStore(str(output_db))
    try:
        db.insert_many(
            "crystal",
            rows_to_columns(rows, CRYSTAL_FIELDS),
            replace=True,
        )
        db.insert_many(
            "trial_structure",
            rows_to_columns(rows, TRIAL_FIELDS),
            replace=True,
        )
        if description:
            db.add_metadata(description)
    finally:
        db.close()


def safe_spacegroup_number(crystal: Crystal) -> int | None:
    """Return a crystal's international-tables number when available."""
    try:
        return crystal.space_group.international_tables_number
    except (AttributeError, TypeError):
        return None


def _cif_text_before_end(cif: Cif) -> str:
    """Return CIF text without the terminal ``#END`` marker."""
    text = cif.to_string().rstrip()

    if text.endswith("#END"):
        text = text[:-4].rstrip()

    return text + "\n"


def orientation_degeneracies_from_site_metadata(
    site_metadata: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    """
    Count the number of explicit states belonging to each component.
    """

    values_by_component: dict[str, set[int]] = {}

    for site in site_metadata:
        site_name = str(site.get("site", "?"))
        allowed_states = site.get("allowed_states", [])

        if not allowed_states:
            raise ValueError(
                f"Prepared site {site_name} has no allowed states."
            )

        counts_for_site: dict[str, int] = {}

        for state in allowed_states:
            component = state.get("component")

            if component is None:
                raise ValueError(
                    f"Prepared site {site_name} contains a state "
                    "without a component label."
                )

            component = str(component)

            counts_for_site[component] = (
                counts_for_site.get(component, 0) + 1
            )

        for component, count in counts_for_site.items():
            values_by_component.setdefault(component, set()).add(int(count))

    cleaned: dict[str, int] = {}

    for component, values in values_by_component.items():
        if len(values) != 1:
            raise ValueError(
                "Inconsistent orientation degeneracy for component "
                f"{component}: {sorted(values)}."
            )

        cleaned[component] = next(iter(values))

    return cleaned


def write_disordered_cif(
    output_file: str | Path,
    crystal: Crystal,
    atom_records: Sequence[Mapping[str, Any]],
    site_metadata: Sequence[Mapping[str, Any]] | None = None,
) -> None:
    """Write a prepared P1 disorder CIF using CSPy's standard CIF writer. """
    unit_cell = crystal.unit_cell
    group_map: dict[str, dict[str, Any]] = {}

    data_block: dict[str, Any] = {
        "cell_length_a": float(unit_cell.a),
        "cell_length_b": float(unit_cell.b),
        "cell_length_c": float(unit_cell.c),
        "cell_angle_alpha": float(unit_cell.alpha_deg),
        "cell_angle_beta": float(unit_cell.beta_deg),
        "cell_angle_gamma": float(unit_cell.gamma_deg),
        "space_group_name_H-M_alt": "P 1",
        "space_group_IT_number": 1,
        "symmetry_space_group_name_H-M": "P 1",
        "symmetry_Int_Tables_number": 1,
        "symmetry_equiv_pos_as_xyz": ["x,y,z"],
        "atom_site_label": [],
        "atom_site_type_symbol": [],
        "atom_site_fract_x": [],
        "atom_site_fract_y": [],
        "atom_site_fract_z": [],
        "atom_site_U_iso_or_equiv": [],
        "atom_site_adp_type": [],
        "atom_site_occupancy": [],
        "atom_site_disorder_assembly": [],
        "atom_site_disorder_group": [],
    }

    for row in atom_records:
        data_block["atom_site_label"].append(str(row["label"]))
        data_block["atom_site_type_symbol"].append(str(row["symbol"]))
        data_block["atom_site_fract_x"].append(float(row["fract_x"]))
        data_block["atom_site_fract_y"].append(float(row["fract_y"]))
        data_block["atom_site_fract_z"].append(float(row["fract_z"]))
        data_block["atom_site_U_iso_or_equiv"].append(0.05)
        data_block["atom_site_adp_type"].append("Uiso")
        data_block["atom_site_occupancy"].append(float(row["occupancy"]))
        data_block["atom_site_disorder_assembly"].append(
            str(row["disorder_assembly"])
        )
        data_block["atom_site_disorder_group"].append(
            str(row["disorder_group"])
        )

        assembly = row["disorder_assembly"]
        group = row["disorder_group"]
        if assembly == "." or group == ".":
            continue
        site_label = f"{assembly}_{group}"
        component = str(row["component"])
        n_mappings = int(row.get("n_mappings", 1))
        overlay_rmsd = row.get("overlay_rmsd")
        if overlay_rmsd is not None and np.isfinite(float(overlay_rmsd)):
            overlay_rmsd = float(overlay_rmsd)
        else:
            overlay_rmsd = None
        group_map[site_label] = {
            "assembly": assembly,
            "group": group,
            "site_label": site_label,
            "component": component,
            "mapping_index": int(row.get("mapping_index", 1)),
            "raw_mapping_index": int(row.get("raw_mapping_index", 1)),
            "n_mappings": n_mappings,
            "overlay_rmsd": overlay_rmsd,
            "original_joint_state": row.get("original_joint_state"),
            "original_joint_occupancy": row.get("original_joint_occupancy"),
            "physical_site": row.get("physical_site"),
            "site_role": row.get("site_role"),
            "original_assembly": row.get("original_assembly"),
        }

    if site_metadata:
        cleaned_degeneracies = (orientation_degeneracies_from_site_metadata(site_metadata))
        
    else:
        values_by_component: dict[str, set[int]] = {}

        for row in atom_records:
            assembly = str(row["disorder_assembly"])
            group = str(row["disorder_group"])

            if assembly == "." or group == ".":
                continue

            component = str(row["component"])
            n_mappings = int(row.get("n_mappings", 1))

            values_by_component.setdefault(component, set()).add(n_mappings)

        cleaned_degeneracies: dict[str, int] = {}

        for component, values in values_by_component.items():
            if len(values) != 1:
                raise ValueError(
                    "Inconsistent orientation degeneracy for component "
                    f"{component}: {sorted(values)}."
                )

            cleaned_degeneracies[component] = next(iter(values))

    cif = Cif({"cspy_disordered": data_block})

    comment_lines = ["", "# cspy_disord_group_map_begin",]

    for site_label in sorted(group_map):
        comment_lines.append("# cspy_disord_group_map " + json.dumps(group_map[site_label],sort_keys=True,)
        )

    comment_lines.append("# cspy_disord_group_map_end")

    if site_metadata:
        comment_lines.append("# cspy_disord_site_map_begin")

        for site in sorted(
            site_metadata,
            key=lambda item: str(item["site"]),
        ):
            comment_lines.append("# cspy_disord_site_map "+ json.dumps(site,sort_keys=True,))

        comment_lines.append("# cspy_disord_site_map_end")

    comment_lines.append("# cspy_disord_orientation_degeneracies "+ json.dumps(cleaned_degeneracies,sort_keys=True,))
    comment_lines.append("#END")

    output_file = Path(output_file)
    output_file.write_text(_cif_text_before_end(cif)+ "\n".join(comment_lines)+ "\n")

    LOG.info("Wrote disordered CIF: %s", output_file)


def is_likely_text(path: str | Path, blocksize: int = 512) -> bool:
    """Return whether a file appears to be readable text."""
    try:
        with Path(path).open("rb") as handle:
            return b"\x00" not in handle.read(blocksize)
    except OSError:
        return False


def process_cif_atom_line(
    line: str,
    headers: Sequence[str],
    label_counts: dict[str, int],
) -> str:
    """Renumber atom labels and round fractional coordinates in one CIF row."""
    parts = cif_tokens(line)
    if len(parts) < len(headers):
        return line if line.endswith("\n") else line + "\n"

    label_index = headers.index("_atom_site_label")
    coordinate_indices = [
        headers.index("_atom_site_fract_x"),
        headers.index("_atom_site_fract_y"),
        headers.index("_atom_site_fract_z"),
    ]
    element_value = (
        parts[headers.index("_atom_site_type_symbol")]
        if "_atom_site_type_symbol" in headers
        else parts[label_index]
    )
    element = normalise_element_symbol(element_value, parts[label_index])
    label_counts[element] = label_counts.get(element, 0) + 1
    parts[label_index] = f"{element}{label_counts[element]}"
    for index in coordinate_indices:
        parts[index] = f"{parse_cif_number(parts[index]):.6f}"
    return "  ".join(format_cif_token(part) for part in parts) + "\n"


def ensure_cspy_spacegroup_tags(lines: Sequence[str]) -> tuple[list[str], bool]:
    """Add legacy symmetry tags expected by older CSPy CIF paths."""
    has_hm = any(
        line.strip().startswith("_symmetry_space_group_name_H-M") for line in lines
    )
    has_number = any(
        line.strip().startswith("_symmetry_Int_Tables_number") for line in lines
    )
    has_equiv = any(
        line.strip().startswith("_symmetry_equiv_pos_as_xyz") for line in lines
    )
    hm_value = find_cif_scalar_value(
        lines,
        (
            "_symmetry_space_group_name_H-M",
            "_space_group_name_H-M_alt",
            "_space_group_name_H-M",
        ),
    ) or "P 1"
    number_value = find_cif_scalar_value(
        lines,
        ("_symmetry_Int_Tables_number", "_space_group_IT_number"),
    ) or "1"
    insert_lines: list[str] = []
    if not has_hm:
        insert_lines.append(
            f"_symmetry_space_group_name_H-M    {format_cif_token(hm_value)}\n"
        )
    if not has_number:
        insert_lines.append(f"_symmetry_Int_Tables_number       {number_value}\n")
    if not has_equiv:
        insert_lines.extend(
            ["loop_\n", "_symmetry_equiv_pos_as_xyz\n", "x,y,z\n"]
        )
    return ([lines[0], *insert_lines, *lines[1:]], True) if insert_lines else (list(lines), False)


def normalise_cif_file(
    path: str | Path,
    *,
    make_backup: bool = True,
    dry_run: bool = False,
) -> tuple[bool, str]:
    """Normalise a generated CIF for reliable loading through CSPy."""
    path = Path(path)
    if not is_likely_text(path):
        return False, "binary_or_unreadable"
    lines = path.read_text(encoding="utf-8", errors="surrogateescape").splitlines(True)
    if not lines:
        return False, "empty_file"

    new_lines = [f"data_{path.stem}\n"]
    in_atom_loop = False
    headers: list[str] = []
    label_counts: dict[str, int] = {}
    modified = new_lines[0] != lines[0]

    for line in lines[1:]:
        stripped = line.strip()
        if stripped == "loop_":
            in_atom_loop = True
            headers = []
            new_lines.append(line.lstrip())
            continue
        if in_atom_loop and stripped.startswith("_atom_site_"):
            headers.append(stripped.split()[0])
            new_lines.append(line.lstrip())
            continue
        if in_atom_loop and stripped.startswith("_"):
            in_atom_loop = False
            new_lines.append(line.lstrip())
            continue
        if in_atom_loop and headers and stripped and not stripped.startswith("#"):
            new_line = process_cif_atom_line(stripped, headers, label_counts)
            modified = modified or new_line.strip() != stripped
            new_lines.append(new_line)
        else:
            new_lines.append(line.lstrip())

    cleaned_lines: list[str] = []
    in_symmetry_loop = False
    for line in new_lines:
        stripped = line.strip()
        if stripped == "_symmetry_equiv_pos_as_xyz":
            in_symmetry_loop = True
            cleaned_lines.append(line)
            continue
        if in_symmetry_loop:
            if stripped.startswith("_") or stripped == "loop_":
                in_symmetry_loop = False
            elif stripped:
                cleaned_lines.append("x,y,z\n")
                modified = True
                in_symmetry_loop = False
                continue
        cleaned_lines.append(line)

    new_lines, added_tags = ensure_cspy_spacegroup_tags(cleaned_lines)
    modified = modified or added_tags
    if not modified and new_lines == [line.lstrip() for line in lines]:
        return True, "unchanged"
    if dry_run:
        return True, "would_modify"

    file_descriptor, temporary_name = tempfile.mkstemp(dir=path.parent)
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)
    try:
        temporary_path.write_text(
            "".join(new_lines),
            encoding="utf-8",
            errors="surrogateescape",
        )
        if make_backup:
            shutil.copy2(path, str(path) + ".bak")
        shutil.copystat(path, temporary_path)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return True, "modified"
