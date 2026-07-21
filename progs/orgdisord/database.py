import json
import logging
from pathlib import Path
from typing import Any

from cspy import Crystal

from .disord_utils import (
    normalise_cif_file,
    read_orientation_degeneracies_from_cif,
    safe_spacegroup_number,
    write_structure_rows,
)

LOG = logging.getLogger("cspy.progs.orgdisord.database")

# Preserve the original private helper name for existing callers/tests.
_safe_spacegroup_number = safe_spacegroup_number

def load_orgdisord_manifest(
    manifest_path: str | Path,
) -> dict[str, dict[str, Any]]:
    """Load enumeration provenance keyed by generated structure ID."""
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        LOG.warning("No orgdisord metadata manifest found: %s", manifest_path)
        return {}

    manifest = json.loads(manifest_path.read_text())
    source_file = manifest.get("source_file")
    orientation_degeneracies = manifest.get("orientation_degeneracies", {})
    metadata_by_id: dict[str, dict[str, Any]] = {}

    for item in manifest.get("structures", []):
        structure_id = str(item["id"])
        metadata = {
            key: value
            for key, value in item.items()
            if key not in {"id", "file"}
        }
        metadata.setdefault("source_file", source_file)
        metadata.setdefault("multiplicity", 1)
        metadata.setdefault(
            "orientation_degeneracies",
            orientation_degeneracies,
        )
        metadata_by_id[structure_id] = metadata

    return metadata_by_id


def load_cif_as_db_row(
    cif_file: str | Path,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Load one CIF and return a combined crystal/trial datastore row."""
    cif_file = Path(cif_file)
    crystal = Crystal.load(str(cif_file))
    return {
        "id": cif_file.stem,
        "spacegroup": safe_spacegroup_number(crystal),
        "density": crystal.density,
        "energy": None,
        "molecule_id": "",
        "file_content": crystal.to_shelx_string(),
        "minimization_step": None,
        "trial_number": 0,
        "valid": True,
        "minimization_time": None,
        "metadata": json.dumps(metadata),
    }


def write_cifs_to_cspy_db(
    cif_files,
    output_db: str | Path = "configurations.db",
    manifest_path: str | Path | None = None,
    source_file: str | Path | None = None,
    clean_cifs: bool = True,
) -> int:
    """Store generated CIF files in a CSPy database without optimisation."""
    cif_files = [Path(path) for path in sorted(cif_files)]
    output_db = Path(output_db)
    if not cif_files:
        raise ValueError("No CIF files found to write to the database.")

    metadata_by_id = (
        load_orgdisord_manifest(manifest_path)
        if manifest_path is not None
        else {}
    )
    source_degeneracies = read_orientation_degeneracies_from_cif(source_file)

    if clean_cifs:
        for cif_file in cif_files:
            ok, status = normalise_cif_file(
                cif_file,
                make_backup=False,
                dry_run=False,
            )
            if not ok:
                raise RuntimeError(
                    f"Could not reformat CIF {cif_file}: {status}"
                )

    rows = []
    resolved_source = (
        str(Path(source_file).resolve())
        if source_file is not None
        else None
    )

    for cif_file in cif_files:
        default_metadata = {
            "source_file": resolved_source,
            "components": None,
            "ratio": None,
            "multiplicity": 1,
            "orientation_degeneracies": source_degeneracies,
        }
        metadata = dict(
            metadata_by_id.get(cif_file.stem, default_metadata)
        )
        if not metadata.get("orientation_degeneracies"):
            metadata["orientation_degeneracies"] = source_degeneracies
        rows.append(load_cif_as_db_row(cif_file, metadata))

    write_structure_rows(
        output_db,
        rows,
        description=f"Added {len(rows)} orgdisord enumerated structures.",
    )
    LOG.info("Wrote %d structures to %s", len(rows), output_db)
    return len(rows)
