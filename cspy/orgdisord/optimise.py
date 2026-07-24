import csv
import json
import logging
import os
import shutil
import subprocess
import time

import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from cspy import Crystal
from cspy.db.datastore import CspDataStore

from .disord_utils import query_rows, write_structure_rows
from cspy.formats import DmacrysSummary


LOG = logging.getLogger(__name__)

ERROR_FIELDS = [
    "id",
    "error",
    "returncode",
    "workdir",
    "time",
    "stdout_file",
    "stderr_file",
    "summary_file",
]


def read_structures_from_db(input_db):
    db = CspDataStore(str(input_db))

    try:
        rows = query_rows(
            db,
            """
            select C.id, C.file_content, T.metadata
            from crystal C
            join trial_structure T on T.id == C.id
            where T.valid == 1
            order by C.id
            """,
        )
    finally:
        db.close()

    structures = []

    for structure_id, file_content, metadata_text in rows:
        if metadata_text:
            metadata = json.loads(metadata_text)
        else:
            metadata = {}

        structures.append(
            {
                "id": structure_id,
                "file_content": file_content,
                "metadata": metadata,
            }
        )

    return structures

def stage_input_file_for_cspy_opt(path, workdir):
    
    if path is None:
        return None

    source = Path(path)
    
    if not source.exists():
        raise FileNotFoundError(f"Input file not found: {source}")
    
    destination = Path(workdir) / source.name
    shutil.copy2(source.resolve(), destination)

    return destination.name

def resolve_n_workers(args):
    if args.n_workers is not None:
        return int(args.n_workers)

    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")

    if slurm_cpus is None:
        return 1

    total_cpus = int(slurm_cpus)

    cpus_per_worker = int(args.gaussian_cpus or 1)

    return max(1, total_cpus // cpus_per_worker)

def build_cspy_opt_command(res_name, args, multipole_arg=None, axis_arg=None):
    cmd = [
        "cspy-opt",
        res_name,
        "--log-level",
        args.log_level,
        "--calculation-type",
        args.calculation_type,
    ]
    
    if args.calculation_type == "dmacrys":
        cmd += ["-p", args.potential]
        
        if multipole_arg is not None:
            cmd += ["--multipole", str(multipole_arg)]

        if args.basis_set is not None:
            cmd += ["--basis-set", args.basis_set]

        if args.method is not None:
            cmd += ["--method", args.method]

        if args.gaussian_cpus is not None:
            cmd += ["-j", str(args.gaussian_cpus)]

        if args.cutoff is not None:
            cmd += ["--cutoff", str(args.cutoff)]

        if args.dma_switch is not None:
            cmd += ["--dma-switch", str(args.dma_switch)]

        if axis_arg is not None:
            cmd += ["--axis", str(axis_arg)]
            
        if args.reorder_atoms_if_high_rmsd:
            cmd.append("--reorder-atoms-if-high-rmsd")
            cmd += ["--reorder-method", args.reorder_method]

    elif args.calculation_type == "mace":
        cmd += ["--mace_model", args.mace_model]

    else:
        raise ValueError(
            f"Unsupported calculation type: {args.calculation_type}"
        )

    if args.single_point:
        cmd.append("--single-point")


    cmd.append("--outputs")
    cmd.append("--no-cleanup")

    return cmd


def list_workdir_files(workdir):
    workdir = Path(workdir)
    return ", ".join(sorted(p.name for p in workdir.iterdir()))


def find_optimised_cif(workdir, structure_id=None):
    """
    Return the optimised CIF produced by cspy-opt.

    Do not try to reconstruct the filename from args.method/args.multipole,
    because cspy-opt may include method names from cspy.toml.
    """
    workdir = Path(workdir)
    opt_cifs = sorted(workdir.glob("*.opt.cif"))

    if not opt_cifs:
        raise FileNotFoundError(
            "No .opt.cif file produced by cspy-opt. "
            f"Files present in workdir: {list_workdir_files(workdir)}"
        )

    if structure_id is not None:
        exact = workdir / f"{structure_id}.opt.cif"
        if exact.exists():
            return exact

    if len(opt_cifs) == 1:
        return opt_cifs[0]

    raise RuntimeError(
        f"More than one .opt.cif found in {workdir}: "
        + ", ".join(str(p.name) for p in opt_cifs)
    )


def find_dmacrys_summary(workdir, opt_cif=None):
    workdir = Path(workdir)

    if opt_cif is not None:
        opt_cif = Path(opt_cif)
        stem = opt_cif.name[:-8]  # remove ".opt.cif"
        candidate = workdir / f"{stem}.dmacrys_summary"

        if candidate.exists():
            return candidate

    summaries = sorted(workdir.glob("*.dmacrys_summary"))

    if len(summaries) == 1:
        return summaries[0]

    if not summaries:
        raise FileNotFoundError(
            "No .dmacrys_summary file produced by cspy-opt. "
            f"Files present in workdir: {list_workdir_files(workdir)}"
        )

    raise RuntimeError(
        f"More than one .dmacrys_summary found in {workdir}: "
        + ", ".join(path.name for path in summaries)
    )


def parse_dmacrys_summary(
    summary_file: str | Path,
) -> dict[str, object]:
    """Parse a DMACRYS summary using CSPy's standard parser."""
    summary_path = Path(summary_file)

    info: dict[str, object] = {
        "summary_file": str(summary_path),
        "valid": None,
        "error": None,
        "final_lattice_energy": None,
        "n_molecules": None,
        "cell_volume": None,
        "density": None,
    }

    if not summary_path.exists():
        return info

    contents = summary_path.read_text(
        errors="replace"
    )
    summary = DmacrysSummary(contents)

    valid = getattr(summary, "valid", None)

    error = getattr(summary, "error", None)

    if error is not None:
        valid = False

    info.update(
        {
            "valid": valid,
            "error": error,
            "final_lattice_energy": getattr(summary, "final_energy", None,),
            "n_molecules": getattr(summary, "z", None,),
            "cell_volume": getattr(summary, "final_volume", None,),
            "density": getattr(summary,"final_density",None,),
        }
    )

    return info

def normalise_dmacrys_energy(summary_info, metadata):
    """
    Convert DMACRYS final lattice energy to kJ/mol per molecule using:

        E_per_molecule = E_dmacrys * Z_dmacrys / Z_true
    """
    raw_energy = summary_info.get("final_lattice_energy")
    dmacrys_z = summary_info.get("n_molecules")
    components = metadata.get("components")

    if raw_energy is None:
        raise ValueError("Could not parse Final Lattice Energy from DMACRYS summary.")

    if dmacrys_z is None:
        raise ValueError("Could not parse DMACRYS reported Z from summary.")

    if not components:
        raise ValueError("Could not normalise energy because metadata['components'] is missing.")

    true_z = sum(int(round(float(value))) for value in components.values())

    if true_z <= 0:
        raise ValueError(f"Invalid true Z from metadata components: {components}")

    corrected_energy = float(raw_energy) * float(dmacrys_z) / float(true_z)

    info = {
        "dmacrys_raw_final_lattice_energy": raw_energy,
        "dmacrys_reported_z": dmacrys_z,
        "true_z_from_components": true_z,
        "corrected_energy_per_molecule": corrected_energy,
        "energy_normalisation": "energy = raw_energy * dmacrys_reported_z / true_z_from_components",
    }

    if int(round(float(dmacrys_z))) != int(round(float(true_z))):
        LOG.warning(
            "Correcting DMACRYS energy normalisation: raw=%s, dmacrys_z=%s, true_z=%s, corrected=%s",
            raw_energy,
            dmacrys_z,
            true_z,
            corrected_energy,
        )

    return corrected_energy, info


def failure_result(structure_id, error, returncode, workdir, elapsed):
    workdir = Path(workdir)

    return {
        "id": structure_id,
        "success": False,
        "error": error,
        "returncode": returncode,
        "workdir": str(workdir),
        "time": float(elapsed),
        "stdout_file": str(workdir / f"{structure_id}.cspy-opt.stdout"),
        "stderr_file": str(workdir / f"{structure_id}.cspy-opt.stderr"),
        "summary_file": str(workdir / f"{structure_id}.dmacrys_summary"),
    }


def write_row_to_db(row, output_db):
    """Write one combined crystal/trial row to a CSPy database."""
    write_structure_rows(output_db, [row])


def run_one_optimisation(structure, args):
    structure_id = structure["id"]
    workdir = Path(args.workdir) / structure_id
    workdir.mkdir(parents=True, exist_ok=True)

    res_name = f"{structure_id}.res"
    (workdir / res_name).write_text(structure["file_content"])

    try:
        if args.cspy_config_resolved is not None:
            shutil.copy2(args.cspy_config_resolved, workdir / "cspy.toml")

        if args.multipole is not None:
            multipole_source = Path(args.multipole)

            multipole_arg = stage_input_file_for_cspy_opt(
                multipole_source,
                workdir,
            )
        else: 
            multipole_arg = None

        if args.axis is not None:
            axis_source = Path(args.axis)

            axis_arg = stage_input_file_for_cspy_opt(
                axis_source,
                workdir,
            )
        else: 
            axis_arg = None

        cmd = build_cspy_opt_command(
            res_name,
            args,
            multipole_arg=multipole_arg,
            axis_arg=axis_arg,
        )

    except Exception as exc:
        return failure_result(
            structure_id=structure_id,
            error=repr(exc),
            returncode=None,
            workdir=workdir,
            elapsed=0.0,
        )

    LOG.info("Starting optimisation for %s", structure_id)
    LOG.debug("Command for %s: %s", structure_id, " ".join(cmd))

    start = time.time()

    completed = subprocess.run(
        cmd,
        cwd=workdir,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    elapsed = time.time() - start

    stdout_path = workdir / f"{structure_id}.cspy-opt.stdout"
    stderr_path = workdir / f"{structure_id}.cspy-opt.stderr"

    stdout_path.write_text(completed.stdout or "")
    stderr_path.write_text(completed.stderr or "")

    if completed.returncode != 0:
        return failure_result(
            structure_id=structure_id,
            error=(
                f"cspy-opt returned non-zero exit code {completed.returncode}. "
                f"See {stderr_path.name}"
            ),
            returncode=completed.returncode,
            workdir=workdir,
            elapsed=elapsed,
        )

    try:
        opt_cif = find_optimised_cif(
            workdir,
            structure_id=structure_id,
        )

        crystal = Crystal.load(str(opt_cif))

        summary_file = find_dmacrys_summary(
            workdir,
            opt_cif=opt_cif,
        )

        summary_info = parse_dmacrys_summary(summary_file)
        
        if (
            args.calculation_type == "dmacrys"
            and not args.single_point
            and summary_info["valid"] is not True
        ):
            raise RuntimeError(
                "DMACRYS did not report a valid minimisation. "
                f"See {summary_file.name}."
            )

        metadata = dict(structure["metadata"])
        metadata["minimisation"] = {
            "calculation_type": args.calculation_type,
            "single_point": args.single_point,
        }

        if args.calculation_type == "dmacrys":
            metadata["minimisation"].update(
                {
                    "potential": args.potential,
                    "multipole": str(args.multipole) if args.multipole else None,
                    "axis": str(args.axis) if args.axis else None,
                    "method": args.method,
                    "basis_set": args.basis_set,
                }
            )

            energy, energy_info = normalise_dmacrys_energy(summary_info, metadata)
            metadata["minimisation"].update(energy_info)

        elif args.calculation_type == "mace":
            metadata["minimisation"]["mace_model"] = args.mace_model
            energy = summary_info.get("final_lattice_energy")

        density = summary_info.get("density") or crystal.density

        row = {
            "id": structure_id,
            "spacegroup": crystal.space_group.international_tables_number,
            "density": density,
            "energy": energy,
            "molecule_id": "",
            "file_content": crystal.to_shelx_string(),
            "minimization_step": 1,
            "trial_number": 0,
            "valid": True,
            "minimization_time": elapsed,
            "metadata": json.dumps(metadata),
        }

    except Exception as exc:
        return failure_result(
            structure_id=structure_id,
            error=repr(exc),
            returncode=completed.returncode,
            workdir=workdir,
            elapsed=elapsed,
        )

    if not args.keep_workdirs:
        try:
            shutil.rmtree(workdir)
            LOG.info("Removed successful optimisation workdir: %s", workdir)
        except FileNotFoundError:
            pass
        except Exception as exc:
            LOG.warning(
                "Could not remove successful optimisation workdir %s: %r",
                workdir,
                exc,
            )

    return {
        "success": True,
        "row": row,
        "id": structure_id,
        "workdir": str(workdir),
        "time": elapsed,
    }


def run_cspy_disord_optimise(args):
    input_db = Path(args.database)
    output_db = Path(args.output_db) if args.output_db else input_db.with_name(
        input_db.name[:-3] + ".opt.db"
        if input_db.name.endswith(".db")
        else input_db.name + ".opt.db"
    )

    workdir = Path(args.workdir) if args.workdir is not None else Path(
        f"work_{input_db.stem}"
    )

    args.workdir = workdir

    if input_db.resolve() == output_db.resolve():
        raise ValueError(
            "Refusing to write optimised structures into the input database. "
            "Choose a different --output-db."
        )

    args.n_workers = resolve_n_workers(args)

    resume = bool(args.resume)

    if not resume and output_db.exists():
        LOG.warning("Deleting existing output database: %s", output_db)
        output_db.unlink()

    if not resume and Path(args.errors_file).exists():
        Path(args.errors_file).unlink()

    structures = read_structures_from_db(input_db)

    if resume and output_db.exists():
        db = CspDataStore(str(output_db))
        done = {row[0] for row in db.query("select id from crystal")}
        db.close()

        structures = [structure for structure in structures if structure["id"] not in done]
        LOG.info("Resume mode: skipping %d completed structures", len(done))

    if args.max_structures is not None:
        structures = structures[: args.max_structures]

    LOG.info("Optimising %d structures from %s", len(structures), input_db)
    LOG.info("Output database: %s", output_db)

    workdir = Path(args.workdir)

    if args.fresh_workdir and workdir.exists():
        resolved = workdir.resolve()
        unsafe_paths = {
            Path(".").resolve(),
            Path("/").resolve(),
            Path.home().resolve(),
        }

        if resolved in unsafe_paths:
            raise ValueError(f"Refusing to delete unsafe workdir: {workdir}")

        LOG.warning("Deleting existing optimisation workdir: %s", workdir)
        shutil.rmtree(workdir)

    workdir.mkdir(parents=True, exist_ok=True)

    output_db.parent.mkdir(parents=True, exist_ok=True)

    if args.cspy_config is None:
        candidate = Path("cspy.toml")
        args.cspy_config_resolved = candidate.resolve() if candidate.exists() else None
    else:
        cspy_config = Path(args.cspy_config)

        if not cspy_config.exists():
            raise FileNotFoundError(f"cspy config file not found: {cspy_config}")

        args.cspy_config_resolved = cspy_config.resolve()

    if args.cspy_config_resolved is not None:
        LOG.info("Using cspy config: %s", args.cspy_config_resolved)
    else:
        LOG.info("No cspy.toml supplied or found in submit directory.")

    success_count = 0
    failure_count = 0

    status = OptimiseStatus(
        total=len(structures),
        n_workers=args.n_workers,
        status_file=args.status_file,
    )
    status.write()

    with ThreadPoolExecutor(max_workers=args.n_workers) as executor:
        future_to_structure = {
            executor.submit(run_one_optimisation, structure, args): structure
            for structure in structures
        }

        for future in as_completed(future_to_structure):
            structure = future_to_structure[future]
            try:
                result = future.result()
            except Exception as exc:
                result = failure_result(
                    structure_id=structure["id"],
                    error=repr(exc),
                    returncode=None,
                    workdir=Path(args.workdir) / structure["id"],
                    elapsed=0.0,
                )

            if result.get("success", False):
                write_row_to_db(result["row"], output_db)
                success_count += 1

                LOG.info(
                    "Finished optimisation for %s",
                    result["id"],
                )

            else:
                errors_file = Path(args.errors_file)
                errors_file.parent.mkdir(parents=True, exist_ok=True)

                write_header = (
                    not errors_file.exists()
                    or errors_file.stat().st_size == 0
                )

                with open(errors_file, "a", newline="") as handle:
                    writer = csv.DictWriter(
                        handle,
                        fieldnames=ERROR_FIELDS,
                    )

                    if write_header:
                        writer.writeheader()

                    writer.writerow(
                        {
                            field: result.get(field)
                            for field in ERROR_FIELDS
                        }
                    )

                if not args.keep_workdirs:
                    failed_workdir = Path(result.get("workdir", ""))

                    try:
                        shutil.rmtree(failed_workdir)
                        LOG.info(
                            "Removed failed optimisation workdir: %s",
                            failed_workdir,
                        )
                    except FileNotFoundError:
                        pass
                    except Exception as exc:
                        LOG.warning(
                            "Could not remove failed optimisation workdir %s: %r",
                            failed_workdir,
                            exc,
                        )

                failure_count += 1

                LOG.warning(
                    "Failed optimisation for %s: %s",
                    result.get("id", structure["id"]),
                    result.get("error", "unknown error"),
                )

            status.update(
                success=result.get("success", False),
                elapsed=result.get("time", 0.0),
            )

    if success_count:
        db = CspDataStore(str(output_db))
        db.add_metadata(
            f"Added {success_count} cspy-disord optimised structures. "
            f"Failed optimisations: {failure_count}."
        )
        db.close()

    LOG.info(
        "Optimisation complete: %d successful, %d failed. Output DB: %s",
        success_count,
        failure_count,
        output_db,
    )

    return success_count


class OptimiseStatus:
    def __init__(self, total, n_workers, status_file):
        self.total = int(total)
        self.n_workers = max(1, int(n_workers))
        self.status_file = Path(status_file)
        self.status_file.parent.mkdir(parents=True, exist_ok=True)
        self.start_time = time.time()
        self.completed = 0
        self.valid = 0
        self.failed = 0
        self.total_opt_time = 0.0

    def update(self, success, elapsed):
        self.completed += 1
        self.total_opt_time += float(elapsed)

        if success:
            self.valid += 1
        else:
            self.failed += 1

        self.write()

    def write(self):
        remaining = max(self.total - self.completed, 0)
        wall_time = time.time() - self.start_time

        if self.completed:
            mean_opt_time = self.total_opt_time / self.completed
            effective_workers = min(self.n_workers, max(remaining, 1))
            eta_s = (mean_opt_time * remaining) / effective_workers
        else:
            mean_opt_time = 0.0
            eta_s = None

        df = pd.DataFrame(
            {
                "target": [self.total],
                "completed": [self.completed],
                "remaining": [remaining],
                "valid": [self.valid],
                "failed": [self.failed],
                "n_workers": [self.n_workers],
                "wall_time_s": [round(wall_time, 2)],
                "mean_opt_time_s": [round(mean_opt_time, 2)],
                "eta_s": [round(eta_s, 2) if eta_s is not None else None],
            },
            index=["optimise"],
        )

        with open(self.status_file, "w") as handle:
            handle.write(df.to_string())
            handle.write("\n")
