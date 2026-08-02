import hashlib
import subprocess

from unittest.mock import Mock, call, patch

import numpy as np
import pytest
from pymatgen.core import Lattice, Structure

from cspy.db.clustering import calculate_missing_xrd, main as clustering_main
from cspy.db.critic2_patterns import (
    PatternComparison,
    compare_patterns,
    read_pattern_comparisons,
    resolve_critic2_executable,
    write_pattern_comparisons,
)
from cspy.ml.descriptors import PowderPattern


@patch("cspy.db.clustering.Crystal")
def test_calculate_missing_xrd_uses_platon_result(crystal_cls):
    crystal = crystal_cls.from_shelx_string.return_value
    platon = Mock(pattern=np.array([1.0, 2.0, 3.0]))
    crystal.calculate_powder_pattern.return_value = platon

    result = calculate_missing_xrd("test-id", "res contents")

    np.testing.assert_array_equal(result, platon.pattern)
    crystal.calculate_powder_pattern.assert_called_once_with()


@pytest.mark.parametrize("platon_result", [None, RuntimeError("Platon failed")])
@patch("cspy.db.clustering.Crystal")
def test_calculate_missing_xrd_falls_back_to_pymatgen(crystal_cls, platon_result):
    crystal = crystal_cls.from_shelx_string.return_value
    fallback = Mock(pattern=np.array([1.0, 2.0, 3.0]))
    crystal.calculate_powder_pattern.side_effect = [platon_result, fallback]

    result = calculate_missing_xrd("test-id", "res contents")

    np.testing.assert_array_equal(result, fallback.pattern)
    assert crystal.calculate_powder_pattern.call_args_list == [
        call(),
        call(method="pymatgen"),
    ]


@patch("cspy.db.clustering.Crystal")
def test_calculate_missing_xrd_returns_none_when_both_methods_fail(crystal_cls):
    crystal = crystal_cls.from_shelx_string.return_value
    crystal.calculate_powder_pattern.side_effect = [None, RuntimeError("failed")]

    assert calculate_missing_xrd("test-id", "res contents") is None


@patch("cspy.db.clustering.Crystal")
def test_explicit_pxrd_backend_does_not_mix_methods(crystal_cls):
    crystal = crystal_cls.from_shelx_string.return_value
    crystal.calculate_powder_pattern.return_value = None

    result, backend = calculate_missing_xrd(
        "test-id",
        "res contents",
        backend="platon",
        return_backend=True,
    )

    assert result is None
    assert backend is None
    crystal.calculate_powder_pattern.assert_called_once_with()


def test_pymatgen_cif_is_binned_on_the_existing_grid():
    structure = Structure(Lattice.cubic(10.0), ["C"], [[0.0, 0.0, 0.0]])

    pattern = PowderPattern.from_pymatgen_cif_string(structure.to(fmt="cif"))

    assert pattern is not None
    assert pattern.tt_range == (0, 20)
    assert pattern.separation == 0.02
    assert pattern.nbins == 1000
    assert np.max(pattern.pattern) > 0


def test_pymatgen_clustering_import_does_not_require_ccdc():
    from cspy.db.compack_clustering import iterative_pymatgen_batch

    assert callable(iterative_pymatgen_batch)


def test_compack_backend_reports_missing_ccdc():
    from cspy.db import compack_clustering

    if compack_clustering.PackingSimilarity is not None:
        pytest.skip("CSD Python API is installed")
    with pytest.raises(RuntimeError, match="csd-python-api"):
        compack_clustering.mercury_compack("", [], "ref", [], {})


@pytest.mark.parametrize(
    ("two_theta_range", "separation"), [((0, 20), 0), ((20, 0), 0.02), ((0, 20), 0.03)]
)
def test_pymatgen_grid_rejects_invalid_bins(two_theta_range, separation):
    with pytest.raises(ValueError):
        PowderPattern.from_pymatgen_cif_string(
            "unused", two_theta_range=two_theta_range, separation=separation
        )


@patch("cspy.db.critic2_patterns.shutil.which", return_value="/usr/bin/critic2")
@patch("cspy.db.critic2_patterns.subprocess.run")
def test_critic2_patterns_run_only_for_structural_candidates(run, which):
    run.side_effect = [
        subprocess.CompletedProcess(
            [], 0, stdout="features: LAPACK(external) NLopt\n", stderr=""
        ),
        subprocess.CompletedProcess([], 0, stdout=" DIFF = 0.42\n", stderr=""),
        subprocess.CompletedProcess(
            [], 0, stdout=" DIFF = 0.51\n DIFF = 0.07\n", stderr=""
        ),
    ]

    result = compare_patterns(
        "reference cif",
        [("match-1", 0.05, "candidate cif")],
        jobs=1,
    )

    assert len(result) == 1
    assert result[0].structure_id == "match-1"
    assert result[0].gpwdf == 0.42
    assert result[0].gvcpwdf == 0.07
    assert result[0].status == "ok"
    assert run.call_count == 3
    assert run.call_args_list[0].args[0] == ["/usr/bin/critic2", "--version"]
    assert run.call_args_list[1].kwargs["input"].startswith(
        "COMPARE reference.cif candidate.res"
    )
    assert run.call_args_list[1].kwargs["cwd"] == run.call_args_list[2].kwargs["cwd"]
    assert run.call_args_list[1].kwargs["env"]["PWD"] == str(
        run.call_args_list[1].kwargs["cwd"]
    )
    gvcpwdf_command = run.call_args_list[2].kwargs["input"]
    assert gvcpwdf_command.startswith("COMPAREVC candidate.res reference.cif")
    assert " GLOBAL QUICK\n" in gvcpwdf_command
    assert gvcpwdf_command.index("candidate.res") < gvcpwdf_command.index(
        "reference.cif"
    )
    which.assert_called_once_with("critic2")


@patch("cspy.db.critic2_patterns.shutil.which")
@patch("cspy.db.critic2_patterns.subprocess.run")
def test_critic2_is_not_required_without_structural_matches(run, which):
    assert compare_patterns("reference cif", [], jobs=1) == []
    which.assert_not_called()
    run.assert_not_called()


@patch("cspy.db.critic2_patterns.shutil.which", return_value=None)
def test_critic2_patterns_report_missing_executable(which):
    with pytest.raises(RuntimeError, match="executable was not found"):
        compare_patterns(
            "reference cif",
            [("match-1", 0.05, "candidate cif")],
            jobs=1,
        )


@patch("cspy.db.critic2_patterns.shutil.which", return_value="/usr/bin/critic2")
@patch("cspy.db.critic2_patterns.subprocess.run")
def test_gvcpwdf_reports_critic2_without_nlopt(run, which):
    run.return_value = subprocess.CompletedProcess(
        [], 0, stdout="features: LAPACK(external)\n", stderr=""
    )

    with pytest.raises(RuntimeError, match="compiled with NLopt"):
        compare_patterns(
            "reference cif",
            [("match-1", 0.05, "candidate cif")],
            jobs=1,
        )


def test_critic2_pattern_csv_contains_both_stages(tmp_path):
    output = tmp_path / "matches.csv"

    write_pattern_comparisons(
        [PatternComparison("match-1", 0.05, 0.42, 0.07)],
        output,
    )

    assert output.read_text().splitlines() == [
        "structure_id,rmsd,gpwdf,gvcpwdf,status,error,"
        "reference_sha256,candidate_sha256",
        "match-1,0.05,0.42,0.07,ok,,,",
    ]


@patch("cspy.db.critic2_patterns.shutil.which")
def test_critic2_executable_uses_environment(which, monkeypatch):
    monkeypatch.setenv("CSPY_CRITIC2_EXECUTABLE", "/cluster/apps/critic2")
    which.return_value = "/cluster/apps/critic2"

    assert resolve_critic2_executable() == "/cluster/apps/critic2"
    which.assert_called_once_with("/cluster/apps/critic2")


@patch("cspy.db.critic2_patterns.shutil.which", return_value="/conda/env/bin/critic2")
def test_critic2_executable_uses_active_job_path(which, monkeypatch):
    monkeypatch.delenv("CSPY_CRITIC2_EXECUTABLE", raising=False)

    assert resolve_critic2_executable() == "/conda/env/bin/critic2"
    which.assert_called_once_with("critic2")


@patch("cspy.db.critic2_patterns.shutil.which", return_value="/usr/bin/critic2")
@patch("cspy.db.critic2_patterns.subprocess.run")
def test_critic2_candidate_failure_is_recorded_and_checkpointed(
    run, which, tmp_path
):
    run.side_effect = [
        subprocess.CompletedProcess(
            [], 0, stdout="features: LAPACK(external) NLopt\n", stderr=""
        ),
        subprocess.CompletedProcess([], 1, stdout="", stderr="bad structure"),
    ]
    output = tmp_path / "checkpoint.csv"

    rows = compare_patterns(
        "reference cif",
        [("bad", 0.2, "candidate cif")],
        jobs=1,
        checkpoint=output,
    )

    assert rows[0].status == "error"
    assert "bad structure" in rows[0].error
    assert read_pattern_comparisons(output) == rows


@patch("cspy.db.critic2_patterns.shutil.which")
@patch("cspy.db.critic2_patterns.subprocess.run")
def test_critic2_resume_does_not_invoke_executable(run, which):
    reference = "reference cif"
    candidate = "candidate cif"
    previous = [
        PatternComparison(
            "done",
            0.05,
            0.4,
            0.1,
            reference_sha256=hashlib.sha256(reference.encode()).hexdigest(),
            candidate_sha256=hashlib.sha256(candidate.encode()).hexdigest(),
        )
    ]

    assert compare_patterns(
        reference,
        [("done", 0.05, candidate)],
        previous=previous,
    ) == previous
    which.assert_not_called()
    run.assert_not_called()


@patch("cspy.db.critic2_patterns.shutil.which", return_value="/usr/bin/critic2")
@patch("cspy.db.critic2_patterns.subprocess.run")
def test_critic2_runs_each_candidate_in_an_isolated_directory(run, which):
    working_directories = []

    def fake_run(*args, **kwargs):
        if args[0][-1] == "--version":
            return subprocess.CompletedProcess(
                [], 0, stdout="features: NLopt\n", stderr=""
            )
        working_directories.append(kwargs["cwd"])
        (kwargs["cwd"] / "test.db").write_text("auxiliary")
        return subprocess.CompletedProcess([], 0, stdout=" DIFF = 0.1\n", stderr="")

    run.side_effect = fake_run
    compare_patterns(
        "reference cif",
        [("first", 0.1, "candidate"), ("second", 0.2, "candidate")],
        jobs=2,
    )

    assert len({path for path in working_directories}) == 2
    assert all(not path.exists() for path in working_directories)


@patch("cspy.db.clustering.structure_search")
def test_structure_search_does_not_create_unused_output_database(
    structure_search, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)

    clustering_main(
        ["input.db", "-m", "pymatgen", "--compack_exp_str", "reference.res"]
    )

    structure_search.assert_called_once()
    assert not (tmp_path / "output.db").exists()
