import json
import sqlite3
import pytest
import os
import numpy as np
import importlib.util

from argparse import Namespace
from pathlib import Path
from collections.abc import Callable, Sequence

from cspy import Crystal
from cspy.db.datastore import CspDataStore
from cspy.apps.disord import main

from progs.orgdisord.calc import (
    calculate_ensemble_for_group,
    run_cspy_disord_calc,
)
from progs.orgdisord.optimise import (
    build_cspy_opt_command,
)

def requires_orgdisord(test_function):
    orgdisord_available = (importlib.util.find_spec("orgdisord") is not None)

    return pytest.mark.skipif(not orgdisord_available,
        reason="orgdisord is not installed",
        )(test_function)
    
    
ANT_RES_STRING = """TITL C14H10
CELL 0.7 8.5841 5.964 9.6568 90 105.4698 90
LATT -1
SYMM -x,1/2+y,-z
SFAC C H
C1    1       0.971413180000       0.184039380000       0.131808290000
C2    1       0.085709760000       0.051362340000       0.219790440000
C3    1       0.169705850000       0.123800510000       0.360822140000
C4    1       0.287158640000       0.991916880000       0.453259160000
C5    1       0.131446320000       0.340761500000       0.408931260000
C6    1       0.011237920000       0.473909360000       0.313498050000
C7    1       0.933642250000       0.398273770000       0.179308680000
C8    1       0.368552220000       0.064707970000       0.591070590000
C9    1       0.212839900000       0.413552580000       0.546742690000
C10   1       0.330291050000       0.281669890000       0.639169590000
C11   1       0.488772240000       0.931560270000       0.686515180000
C12   1       0.414287130000       0.354108050000       0.780201290000
C13   1       0.566355750000       0.007206450000       0.820693510000
C14   1       0.528595340000       0.221431180000       0.868194820000
H1    2       0.908597790000       0.126898640000       0.024977140000
H2    2       0.114637400000       0.888114730000       0.183799600000
H3    2       0.315925340000       0.828610030000       0.417037520000
H4    2       0.982676390000       0.637037180000       0.349897320000
H5    2       0.842607800000       0.501128730000       0.107946620000
H6    2       0.184073210000       0.576859430000       0.582964340000
H7    2       0.517322700000       0.768421520000       0.650104190000
H8    2       0.385371110000       0.517355830000       0.816203500000
H9    2       0.657390750000       0.904340730000       0.892055230000
H10   2       0.591410720000       0.278571910000       0.975025970000
END
"""
ACR_XYZ_STRING = """23
C13H9N
C       1.335498531800       2.517590134920       0.001469222592
C       2.701397142446       2.517590134920       0.001469222592
C       3.423004059772       1.289317934076       0.001469222592
C       2.765587284356       0.091872907121       0.001217135406
C      -1.245908413918      -2.470691130749      -0.002760433317
C      -2.608774440243      -2.563857049767      -0.003032939805
C      -3.419976071276      -1.392792757962      -0.001463550802
C      -2.845768768455      -0.153448200641       0.000066274512
C      -0.782537133625       1.224880028723       0.001138425272
C       0.610290107704       1.289686800603       0.001070875701
C       1.338977657417       0.044552167083       0.000626427478
C      -0.603233193799      -1.196168020821      -0.000942251188
C      -1.426804176436      -0.011657605096       0.000286313134
H       0.780558693987       3.449636611739       0.001886338761
H       3.246565696816       3.454081843303       0.001756906362
H       4.506627513392       1.312954259059       0.001661317696
H       3.291508070195      -0.854513369898       0.001059136541
H      -0.608343865765      -3.345771268280      -0.003853169451
H      -3.085826450975      -3.537109836630      -0.004501476644
H      -4.498893941179      -1.493734471709      -0.001604166262
H      -3.458149693578       0.741898051179       0.000819443652
H      -1.366979025867       2.140047051807       0.001799246246
N       0.736039688823      -1.152128653237      -0.000456190488
"""

MULTI_ASSEMBLY_CIF_STRING = """data_multi_assembly_methanol
_cell_length_a 10.0
_cell_length_b 10.0
_cell_length_c 10.0
_cell_angle_alpha 90.0
_cell_angle_beta 90.0
_cell_angle_gamma 90.0
_space_group_name_H-M_alt 'P 1'
_space_group_IT_number 1

loop_
_symmetry_equiv_pos_as_xyz
'x, y, z'

loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
_atom_site_disorder_assembly
_atom_site_disorder_group
C1    C  0.500  0.500  0.500  1.0  .  .
O1    O  0.640  0.500  0.500  1.0  .  .
H1    H  0.450  0.590  0.500  1.0  .  .
H2    H  0.450  0.410  0.500  1.0  .  .
H3A   H  0.500  0.500  0.600  0.5  A  1
H3B   H  0.500  0.500  0.400  0.5  A  2
H4A   H  0.720  0.560  0.500  0.5  B  1
H4B   H  0.720  0.440  0.500  0.5  B  2
"""


CspyDisordRunner = Callable[
    [Sequence[str | Path], Path],
    None,
]


@pytest.fixture
def cspy_disord_runner() -> CspyDisordRunner:

    def run(
        arguments: Sequence[str | Path],
        cwd: Path,
    ) -> None:
        original_directory = Path.cwd()

        try:
            os.chdir(cwd)
            main([str(argument) for argument in arguments])
        finally:
            os.chdir(original_directory)

    return run


def read_json_comments(path, prefix):
    records = []

    for line in Path(path).read_text(
        errors="replace"
    ).splitlines():
        stripped = line.strip()

        if not stripped.startswith(prefix):
            continue

        payload = stripped.removeprefix(prefix).strip()
        records.append(json.loads(payload))

    return records


def read_database_rows(database):
    with sqlite3.connect(database) as connection:
        rows = connection.execute(
            """
            select C.id, C.energy, T.metadata
            from crystal C
            join trial_structure T on T.id = C.id
            order by C.id
            """
        ).fetchall()

    return [
        {
            "id": structure_id,
            "energy": energy,
            "metadata": (
                json.loads(metadata_text)
                if metadata_text
                else {}
            ),
        }
        for structure_id, energy, metadata_text in rows
    ]


def prepare_host_guest(
    tmp_path: Path,
    cspy_disord_runner: CspyDisordRunner,
) -> Path:
    host_res = tmp_path / "ANT.res"
    guest_xyz = tmp_path / "ACR.xyz"
    prepared_cif = tmp_path / "ACR_in_ANT.cif"

    host_res.write_text(ANT_RES_STRING)
    guest_xyz.write_text(ACR_XYZ_STRING)

    cspy_disord_runner(
        [
            "prepare-cif",
            host_res,
            "--component",
            "A1",
            "host",
            "--component",
            "A2",
            guest_xyz,
            "--overlay-method",
            "mapped-kabsch",
            "--all-isomorphic-overlays",
            "--atom-match-mode",
            "heavy",
            "--output",
            prepared_cif,
        ],
        tmp_path,
    )

    return prepared_cif


def prepare_multi_assembly(
    tmp_path: Path,
    cspy_disord_runner: CspyDisordRunner,
) -> Path:
    input_cif = tmp_path / "multi_assembly_input.cif"
    prepared_cif = tmp_path / "multi_assembly_prepared.cif"

    input_cif.write_text(MULTI_ASSEMBLY_CIF_STRING)

    cspy_disord_runner(
        [
            "prepare-cif",
            input_cif,
            "--existing-disorder",
            "--multi-assembly-mode",
            "explicit",
            "--joint-state",
            "A:1,B:1",
            "--joint-state",
            "A:2,B:2",
            "--existing-component",
            "A1",
            "--output",
            prepared_cif,
        ],
        tmp_path,
    )

    return prepared_cif


def enumerate_host_guest(
    tmp_path: Path,
    cspy_disord_runner: CspyDisordRunner,
    n_samples: int = 1,
) -> dict[str, Path]:
    
    prepared_cif = prepare_host_guest(
        tmp_path,
        cspy_disord_runner,
    )
    ratio_csv = tmp_path / "ratios.csv"
    output_db = tmp_path / "ACR_in_ANT.db"
    prefix = "ACR_in_ANT"

    ratio_csv.write_text(
        "label,n_A1,n_A2,n_samples\n"
        f"equimolar,1,1,{int(n_samples)}\n"
    )

    cspy_disord_runner(
        [
            "enumerate",
            prepared_cif,
            "--supercell",
            "1",
            "1",
            "1",
            "--prefix",
            prefix,
            "--output-db",
            output_db,
            "--ratio-counts-file",
            ratio_csv,
            "--samples-from",
            "csv",
            "--no-pure-components",
            "--maxiters",
            str(n_samples),
            "--seed",
            "123",
            "--keep-orgdisord-outputs",
        ],
        tmp_path,
    )

    return {
        "prepared_cif": prepared_cif,
        "output_db": output_db,
        "results_directory": tmp_path / f"{prefix}-results",
    }


def enumerate_multi_assembly(
    tmp_path: Path,
    cspy_disord_runner: CspyDisordRunner,
    n_samples: int = 2,
) -> dict[str, Path]:
    prepared_cif = prepare_multi_assembly(
        tmp_path,
        cspy_disord_runner,
    )
    output_db = tmp_path / "multi_assembly.db"
    prefix = "multi_states"

    cspy_disord_runner(
        [
            "enumerate",
            prepared_cif,
            "--supercell",
            "1",
            "1",
            "1",
            "--prefix",
            prefix,
            "--output-db",
            output_db,
            "--maxiters",
            str(n_samples),
            "--no-pure-components",
            "--seed",
            "17",
            "--keep-orgdisord-outputs",
        ],
        tmp_path,
    )

    return {
        "prepared_cif": prepared_cif,
        "output_db": output_db,
        "results_directory": tmp_path / f"{prefix}-results",
    }


def write_test_optimised_database(path):
    metadata = {
        "components": {
            "A1": 1,
            "A2": 1,
        },
        "ratio": {
            "A1": 0.5,
            "A2": 0.5,
        },
        "multiplicity": 1,
        "orientation_degeneracies": {
            "A1": 1.0,
            "A2": 1.0,
        },
    }

    crystal_rows = {
        "id": [
            "structure_1",
            "structure_2",
        ],
        "spacegroup": [1, 1],
        "energy": [-100.0, -99.0],
        "density": [1.2, 1.2],
        "molecule_id": ["", ""],
        "file_content": ["", ""],
    }

    trial_rows = {
        "id": [
            "structure_1",
            "structure_2",
        ],
        "minimization_step": [1, 1],
        "trial_number": [0, 0],
        "valid": [True, True],
        "minimization_time": [1.0, 1.0],
        "metadata": [
            json.dumps(metadata),
            json.dumps(metadata),
        ],
    }

    database = CspDataStore(str(path))

    try:
        database.insert_many(
            "crystal",
            crystal_rows,
            replace=True,
        )
        database.insert_many(
            "trial_structure",
            trial_rows,
            replace=True,
        )
    finally:
        database.close()


@pytest.mark.binaries
@requires_orgdisord
def test_prepare_cif_compatibility(
    tmp_path: Path,
    cspy_disord_runner: CspyDisordRunner,
) -> None:
    
    host_guest_cif = prepare_host_guest(
        tmp_path,
        cspy_disord_runner,
    )
    assert host_guest_cif.is_file()

    host_guest_text = host_guest_cif.read_text()

    assert "# cspy_disord_group_map " in host_guest_text
    assert (
        "# cspy_disord_orientation_degeneracies "
        in host_guest_text
    )

    host_guest_groups = read_json_comments(
        host_guest_cif,
        "# cspy_disord_group_map ",
    )

    assert {
        record["component"]
        for record in host_guest_groups
    } == {"A1", "A2"}

    prepared_existing_cif = prepare_multi_assembly(
        tmp_path,
        cspy_disord_runner,
    )

    assert prepared_existing_cif.is_file()

    group_maps = read_json_comments(
        prepared_existing_cif,
        "# cspy_disord_group_map ",
    )
    site_maps = read_json_comments(
        prepared_existing_cif,
        "# cspy_disord_site_map ",
    )
    orientation_records = read_json_comments(
        prepared_existing_cif,
        "# cspy_disord_orientation_degeneracies ",
    )

    assert len(group_maps) == 4
    assert len(site_maps) == 1
    assert orientation_records == [{"A1": 2}]

    site = site_maps[0]

    assert site["component"] == "A1"
    assert site["state_count"] == 2
    assert len(site["assemblies"]) == 2

    allowed_states = {
        tuple(
            sorted(
                state["original_choices"].items()
            )
        )
        for state in site["allowed_states"]
    }

    assert allowed_states == {
        (("A", "1"), ("B", "1")),
        (("A", "2"), ("B", "2")),
    }


@pytest.mark.binaries
@requires_orgdisord
def test_enumerate_exact_composition(
    tmp_path: Path,
    cspy_disord_runner: CspyDisordRunner,
) -> None:
    
    enumeration = enumerate_host_guest(
        tmp_path,
        cspy_disord_runner,
        n_samples=1,
    )

    rows = read_database_rows(
        enumeration["output_db"]
    )

    assert len(rows) == 1

    metadata = rows[0]["metadata"]

    assert metadata["components"] == {
        "A1": 1,
        "A2": 1,
    }
    assert metadata["ratio"] == {
        "A1": 0.5,
        "A2": 0.5,
    }
    assert metadata["composition_label"] == "equimolar"
    assert len(metadata["state_choices"]) == 2
    assert metadata["multiplicity"] == 1
    assert "orientation_degeneracies" in metadata
    assert (
        metadata["number_of_possible_state_assignments"]
        >= 1
    )


@pytest.mark.binaries
@requires_orgdisord
def test_enumerated_states_keep_atom_order(
    tmp_path: Path,
    cspy_disord_runner: CspyDisordRunner,
) -> None:
    
    enumeration = enumerate_multi_assembly(
        tmp_path,
        cspy_disord_runner,
        n_samples=2,
    )

    rows = read_database_rows(
        enumeration["output_db"]
    )

    assert len(rows) == 2

    sampled_choices = {
        tuple(row["metadata"]["state_choices"])
        for row in rows
    }

    assert sampled_choices == {
        (0,),
        (1,),
    }

    joint_states = {
        tuple(
            sorted(
                row["metadata"]["site_states"][0][
                    "original_joint_state"
                ].items()
            )
        )
        for row in rows
    }

    assert joint_states == {
        (("A", "1"), ("B", "1")),
        (("A", "2"), ("B", "2")),
    }

    generated_cifs = sorted(
        enumeration["results_directory"].glob("*.cif")
    )

    assert len(generated_cifs) == 2

    atom_orders = []

    for generated_cif in generated_cifs:
        crystal = Crystal.load(str(generated_cif))
        molecules = crystal.unit_cell_molecules()

        assert len(molecules) == 1

        atomic_numbers = tuple(
            int(number)
            for number in molecules[0].atomic_numbers
        )

        assert len(atomic_numbers) == 6
        assert sorted(atomic_numbers) == [
            1,
            1,
            1,
            1,
            6,
            8,
        ]

        atom_orders.append(atomic_numbers)

    assert atom_orders[0] == atom_orders[1]


def test_build_cspy_opt_command():
    args = Namespace(
        log_level="INFO",
        calculation_type="dmacrys",
        potential="fit",
        basis_set=None,
        method=None,
        gaussian_cpus=4,
        cutoff=12.0,
        dma_switch=None,
        reorder_atoms_if_high_rmsd=True,
        reorder_method="molecular_axis",
        single_point=False,
        mace_model="mace_mp",
    )

    automatic_dma_command = build_cspy_opt_command(
        "structure.res",
        args,
        multipole_arg=None,
        axis_arg=None,
    )

    potential_index = automatic_dma_command.index("-p")

    assert automatic_dma_command[
        potential_index + 1
    ] == "fit"
    assert "--multipole" not in automatic_dma_command
    assert "--cutoff" in automatic_dma_command
    assert "12.0" in automatic_dma_command
    assert "--reorder-atoms-if-high-rmsd" in automatic_dma_command
    assert "--reorder-method" in automatic_dma_command
    assert "--outputs" in automatic_dma_command
    assert "--no-cleanup" in automatic_dma_command

    supplied_dma_command = build_cspy_opt_command(
        "structure.res",
        args,
        multipole_arg="molecule.dma",
        axis_arg="molecule.axis",
    )

    assert "--multipole" in supplied_dma_command
    assert "molecule.dma" in supplied_dma_command
    assert "--axis" in supplied_dma_command
    assert "molecule.axis" in supplied_dma_command


def test_calc_ensemble_and_write_csv(tmp_path):
    group = [
        {
            "energy": -100.0,
            "multiplicity": 1,
            "components": {
                "A1": 1,
                "A2": 1,
            },
        },
        {
            "energy": -99.0,
            "multiplicity": 1,
            "components": {
                "A1": 1,
                "A2": 1,
            },
        },
    ]

    result = calculate_ensemble_for_group(
        group,
        temperature=298.15,
        energy_input="per_site",
        orientation_degeneracies={
            "A1": 1.0,
            "A2": 1.0,
        },
    )

    assert result["n_samples"] == 2
    assert result["W"] == pytest.approx(2.0)
    assert result["W_sampled"] == pytest.approx(2.0)
    assert result["sample_fraction"] == pytest.approx(1.0)
    assert result["energy_min"] == pytest.approx(-100.0)
    assert result["energy_mean"] == pytest.approx(-99.5)
    assert np.isfinite(result["G"])
    assert result["G"] <= result["energy_min"]

    input_db = tmp_path / "optimised.db"
    output_csv = tmp_path / "ensemble.csv"

    write_test_optimised_database(input_db)

    args = Namespace(
        database=input_db,
        output_csv=output_csv,
        temperature=298.15,
        bootstrap=20,
        ci=95.0,
        energy_input="per_site",
        reference_energy=[
            ("A1", "-100.0"),
            ("A2", "-100.0"),
        ],
        orientation_degeneracy=[],
        seed=123,
    )

    dataframe = run_cspy_disord_calc(args)

    assert output_csv.is_file()
    assert len(dataframe) == 1

    expected_columns = {
        "n_sites",
        "n_A1",
        "n_A2",
        "x_A1",
        "x_A2",
        "W",
        "W_sampled",
        "sample_fraction",
        "n_samples",
        "energy_min",
        "energy_mean",
        "G_ensemble",
        "G_bootstrap_low",
        "G_bootstrap_high",
        "reference_weighted_sum",
        "delta_G",
    }

    assert expected_columns.issubset(
        dataframe.columns
    )

    row = dataframe.iloc[0]

    assert row["n_sites"] == 2
    assert row["n_A1"] == 1
    assert row["n_A2"] == 1
    assert row["x_A1"] == pytest.approx(0.5)
    assert row["x_A2"] == pytest.approx(0.5)
    assert row["n_samples"] == 2
    assert row["energy_min"] == pytest.approx(-100.0)
    assert row["energy_mean"] == pytest.approx(-99.5)
    assert row["reference_weighted_sum"] == pytest.approx(-100.0)
    assert np.isfinite(row["G_ensemble"])
    assert np.isfinite(row["G_bootstrap_low"])
    assert np.isfinite(row["G_bootstrap_high"])
    assert np.isfinite(row["delta_G"])
    assert row["G_bootstrap_low"] <= row["G_bootstrap_high"]