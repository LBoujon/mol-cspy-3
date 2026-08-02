import pytest
import math
from unittest.mock import patch

from cspy.apps import mol_dis
from cspy.flex.flex_molecule import FlexMolecule
from cspy.flex.internals import Internal
from cspy.templates import get_template


def _render_flexible_job(name="CDB", torsions=()):
    return get_template("slurm_script_fmcsp").render(
        name=name,
        molecule_name="trans-stilbene",
        mname="trans-stilbene",
        sg="",
        partition="batch",
        computing=1,
        nodes=1,
        user="user",
        conda="mol-cspy",
        calc_dir="/scratch/user",
        dftb_dir="",
        number_of_torsions=dict.fromkeys(torsions, 1),
        total_configs=7,
        angular_step="15.*D",
        bound="45.*D",
        method="PBE1PBE",
        basis="6-311G**",
    )


def test_moldis_help_does_not_initialize_mpi():
    with pytest.raises(SystemExit, match="0"):
        mol_dis.main(["--help"])


def test_flexible_slurm_job_passes_three_dofs_as_separate_arguments():
    script = _render_flexible_job(
        torsions=("1_2_3_4", "2_3_4_5", "3_4_5_6")
    )

    assert "--scan " not in script
    assert '--scan_dofs "$dof1" "$dof2" "$dof3"' in script
    assert '--scan_dofs "[' not in script


def test_joint_database_job_accepts_one_or_several_components():
    script = _render_flexible_job(name="jointDB")

    assert 'if [ "$#" -eq 1 ]; then' in script
    assert 'cspy-moldis "$@" --jointdb' in script
    assert "--scan_dofs" not in script


def test_three_seven_point_dofs_create_343_grid_points():
    dofs = [
        Internal(
            f"{index}_{index + 1}_{index + 2}_{index + 3}",
            original_value=0.0,
            initial_offset=math.radians(-45),
            number_of_steps=7,
            step_size=math.radians(15),
        )
        for index in range(3)
    ]

    assert len(FlexMolecule.make_combination_of_internal_steps(dofs)) == 343


@patch("cspy.flex.flex_molecule.sobol_vector")
def test_sobol_scan_spans_the_complete_requested_angular_range(sobol_vector):
    sobol_vector.side_effect = ([0.0], [1.0])
    dof = Internal(
        "0_1_2_3",
        original_value=0.0,
        initial_offset=math.radians(-45),
        number_of_steps=7,
        step_size=math.radians(15),
    )

    points = FlexMolecule.make_sobol_points_of_internals([dof], 2)
    offsets = [math.degrees(point[0].get_current_offset()) for point in points]

    assert offsets == pytest.approx([-45.0, 45.0])
