"""Pattern comparison for structural matches using the critic2 executable."""

from __future__ import annotations

import csv
import hashlib
import logging
import os
import re
import shutil
import subprocess
import tempfile

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


LOG = logging.getLogger(__name__)

_GPWDF_RE = re.compile(r"\bDIFF\s*=\s*([0-9.Ee+-]+)")
_GVCPWDF_RE = re.compile(r"\bDIFF\s*=\s*([0-9.Ee+-]+)")
_ENV_EXECUTABLE = "CSPY_CRITIC2_EXECUTABLE"


@dataclass(frozen=True)
class PatternComparison:
    """Pattern-comparison result for one structural match."""

    structure_id: str
    rmsd: float
    gpwdf: float | None = None
    gvcpwdf: float | None = None
    status: str = "ok"
    error: str = ""
    reference_sha256: str = ""
    candidate_sha256: str = ""


def resolve_critic2_executable(executable: str | None = None) -> str:
    """Resolve critic2 from an explicit value, the environment, or ``PATH``."""

    requested = executable or os.environ.get(_ENV_EXECUTABLE) or "critic2"
    resolved = shutil.which(os.path.expandvars(os.path.expanduser(requested)))
    if resolved is None:
        raise RuntimeError(
            "critic2 pattern comparison was requested, but the executable "
            f"was not found: {requested}. Pass --critic2-executable, set "
            f"{_ENV_EXECUTABLE}, or make critic2 available on PATH by "
            "activating its conda environment or loading its cluster module."
        )
    return str(Path(resolved).resolve())


def _critic2_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = "1"
    return environment


def _run_critic2(
    executable: str,
    command: str,
    score_pattern: re.Pattern,
    structure_id: str,
    method: str,
    timeout: float,
    working_directory: Path,
) -> float:
    environment = _critic2_environment()
    environment["PWD"] = str(working_directory)
    try:
        result = subprocess.run(
            [executable, "-q"],
            input=command,
            text=True,
            capture_output=True,
            timeout=timeout,
            env=environment,
            cwd=working_directory,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"critic2 {method} comparison timed out for {structure_id}"
        ) from exc

    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "no output"
        raise RuntimeError(
            f"critic2 {method} comparison failed for {structure_id}: {detail}"
        )

    matches = score_pattern.findall(result.stdout)
    if not matches:
        detail = result.stdout.strip() or result.stderr.strip() or "no output"
        raise RuntimeError(
            f"critic2 did not report a {method} score for {structure_id}: {detail}"
        )
    return float(matches[-1])


def require_nlopt(executable: str, timeout: float) -> None:
    """Check that critic2 was compiled with NLopt/GVCPWDF support."""

    try:
        result = subprocess.run(
            [executable, "--version"],
            input="",
            text=True,
            capture_output=True,
            timeout=timeout,
            env=_critic2_environment(),
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("timed out while checking critic2 features") from exc
    output = f"{result.stdout}\n{result.stderr}"
    if result.returncode != 0 or re.search(
        r"features:.*\bNLopt\b", output, flags=re.IGNORECASE
    ) is None:
        raise RuntimeError(
            f"GVCPWDF requires critic2 compiled with NLopt support; {executable} "
            "does not report NLopt in its feature list"
        )


def _comparison_sort_key(row: PatternComparison) -> tuple:
    missing = float("inf")
    return (
        row.status != "ok",
        missing if row.gvcpwdf is None else row.gvcpwdf,
        missing if row.gpwdf is None else row.gpwdf,
        row.rmsd,
        row.structure_id,
    )


def read_pattern_comparisons(output: str | Path) -> list[PatternComparison]:
    """Read a current or legacy comparison CSV for restart/checkpoint use."""

    path = Path(output)
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"structure_id", "rmsd", "gpwdf"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"Invalid critic2 checkpoint CSV: {path}")
        variable_cell_field = (
            "gvcpwdf" if "gvcpwdf" in reader.fieldnames else "vcpwdf"
        )
        if variable_cell_field not in reader.fieldnames:
            raise ValueError(f"Invalid critic2 checkpoint CSV: {path}")

        rows = []
        for record in reader:
            status = record.get("status") or "ok"
            gpwdf_text = record.get("gpwdf", "").strip()
            gvcpwdf_text = record.get(variable_cell_field, "").strip()
            rows.append(
                PatternComparison(
                    structure_id=record["structure_id"],
                    rmsd=float(record["rmsd"]),
                    gpwdf=float(gpwdf_text) if gpwdf_text else None,
                    gvcpwdf=float(gvcpwdf_text) if gvcpwdf_text else None,
                    status=status,
                    error=record.get("error", ""),
                    reference_sha256=record.get("reference_sha256", ""),
                    candidate_sha256=record.get("candidate_sha256", ""),
                )
            )
    return rows


def write_pattern_comparisons(
    comparisons: Iterable[PatternComparison],
    output: str | Path,
    *,
    log: bool = True,
) -> None:
    """Atomically write structural and critic2 pattern results to CSV."""

    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(comparisons, key=_comparison_sort_key)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "structure_id",
                    "rmsd",
                    "gpwdf",
                    "gvcpwdf",
                    "status",
                    "error",
                    "reference_sha256",
                    "candidate_sha256",
                ]
            )
            for row in rows:
                writer.writerow(
                    [
                        row.structure_id,
                        row.rmsd,
                        "" if row.gpwdf is None else row.gpwdf,
                        "" if row.gvcpwdf is None else row.gvcpwdf,
                        row.status,
                        row.error,
                        row.reference_sha256,
                        row.candidate_sha256,
                    ]
                )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    log_method = LOG.info if log else LOG.debug
    log_method("Wrote %d critic2 pattern comparisons to %s", len(rows), path)


def compare_patterns(
    reference_structure: str,
    candidates: Sequence[tuple[str, float, str]],
    *,
    jobs: int = 1,
    executable: str | None = None,
    timeout: float = 300.0,
    reference_suffix: str = ".cif",
    previous: Sequence[PatternComparison] = (),
    retry_errors: bool = True,
    fail_fast: bool = False,
    checkpoint: str | Path | None = None,
) -> list[PatternComparison]:
    """Compare patterns for candidates that already matched structurally.

    Each candidate runs in an isolated directory because critic2 creates fixed-name
    auxiliary files. Successful checkpoint rows are reused. Failed rows are retried
    by default and are retained with ``status=error`` if the retry also fails.
    """

    if jobs < 1:
        raise ValueError("jobs must be at least 1")
    if timeout <= 0:
        raise ValueError("critic2 timeout must be greater than zero")

    candidate_by_id = {candidate[0]: candidate for candidate in candidates}
    if len(candidate_by_id) != len(candidates):
        raise ValueError("critic2 candidate structure IDs must be unique")

    reference_sha256 = hashlib.sha256(reference_structure.encode()).hexdigest()
    candidate_sha256 = {
        structure_id: hashlib.sha256(contents.encode()).hexdigest()
        for structure_id, _rmsd, contents in candidates
    }
    result_by_id = {}
    for row in previous:
        candidate = candidate_by_id.get(row.structure_id)
        reusable = (
            candidate is not None
            and row.reference_sha256 == reference_sha256
            and row.candidate_sha256 == candidate_sha256[row.structure_id]
            and row.rmsd == candidate[1]
            and (row.status == "ok" or not retry_errors)
        )
        if reusable:
            result_by_id[row.structure_id] = row
    pending = [
        candidate for candidate in candidates if candidate[0] not in result_by_id
    ]
    if not pending:
        return sorted(result_by_id.values(), key=_comparison_sort_key)

    resolved_executable = resolve_critic2_executable(executable)
    require_nlopt(resolved_executable, timeout)
    suffix = (
        reference_suffix
        if reference_suffix.startswith(".")
        else f".{reference_suffix}"
    )

    with tempfile.TemporaryDirectory(prefix="cspy-critic2-") as directory:
        root = Path(directory)

        def compare(index: int, candidate) -> PatternComparison:
            structure_id, rmsd, candidate_structure = candidate
            workspace = root / f"candidate-{index:06d}"
            workspace.mkdir()
            reference_file = workspace / f"reference{suffix}"
            candidate_file = workspace / "candidate.res"
            reference_file.write_text(reference_structure, encoding="utf-8")
            candidate_file.write_text(candidate_structure, encoding="utf-8")
            try:
                gpwdf = _run_critic2(
                    resolved_executable,
                    f"COMPARE {reference_file.name} {candidate_file.name} "
                    "CRYSTAL GPWDF\nEND\n",
                    _GPWDF_RE,
                    structure_id,
                    "GPWDF",
                    timeout,
                    workspace,
                )
                gvcpwdf = _run_critic2(
                    resolved_executable,
                    f"COMPAREVC {candidate_file.name} {reference_file.name} "
                    "GLOBAL QUICK\nEND\n",
                    _GVCPWDF_RE,
                    structure_id,
                    "GVCPWDF",
                    timeout,
                    workspace,
                )
                return PatternComparison(
                    structure_id,
                    rmsd,
                    gpwdf,
                    gvcpwdf,
                    reference_sha256=reference_sha256,
                    candidate_sha256=candidate_sha256[structure_id],
                )
            except Exception as exc:
                if fail_fast:
                    raise
                LOG.error("Pattern comparison failed for %s: %s", structure_id, exc)
                return PatternComparison(
                    structure_id=structure_id,
                    rmsd=rmsd,
                    status="error",
                    error=str(exc),
                    reference_sha256=reference_sha256,
                    candidate_sha256=candidate_sha256[structure_id],
                )

        with ThreadPoolExecutor(max_workers=min(jobs, len(pending))) as executor:
            futures = {
                executor.submit(compare, index, candidate): candidate[0]
                for index, candidate in enumerate(pending)
            }
            for future in as_completed(futures):
                row = future.result()
                result_by_id[row.structure_id] = row
                if checkpoint is not None:
                    write_pattern_comparisons(
                        result_by_id.values(), checkpoint, log=False
                    )

    return sorted(result_by_id.values(), key=_comparison_sort_key)
