from pathlib import Path
from cspy.threshold.threshold_setup import main, load_crystals, check_multipoles, check_bonding, check_molecular_axis
from cspy import Molecule
import pytest


FILE_DIR = Path(__file__).parent
DATA_DIR = FILE_DIR / "thresh_setup_data"
PYRENE_I = DATA_DIR / "pyrene_I_replaced.res"
PYRENE_II = DATA_DIR / "pyrene_II_replaced.res"
PYRENE_III = DATA_DIR / "pyrene_III_replaced.res"
PYRENE_IV = DATA_DIR / "pyrene_IV_replaced.res"
PYRENE_V = DATA_DIR / "pyrene_V_replaced.res"
CRYSTALS = [
    PYRENE_I,
    PYRENE_II,
    PYRENE_III,
    PYRENE_IV,
    PYRENE_V
]


def test_app(capsys) -> None:
    with pytest.raises(SystemExit) as sys_exit:
        # dumb test to catch that the app actually runs
        main(["--help"])
    assert sys_exit.value.code == 0

    expected = "Program that sets up the crystals to carry out a cspy-threshold job."
    screen_output = capsys.readouterr().out
    assert expected in screen_output


def test_app_crystals(caplog) -> None:
    caplog.set_level("INFO")

    # Check it reads the crystals
    main([str(c) for c in CRYSTALS] + ["--dry"])

    screen_output = caplog.text
    expecteds = [
        "Read 5 crystals",
        "Valid SuperCell(n=1, m=1, l=1) found for pyrene_I_replaced",
        "Valid SuperCell(n=1, m=1, l=1) found for pyrene_II_replaced",
        "Valid SuperCell(n=1, m=2, l=1) found for pyrene_III_replaced",
        "Valid SuperCell(n=1, m=1, l=1) found for pyrene_IV_replaced",
        "Valid SuperCell(n=1, m=1, l=1) found for pyrene_V_replaced",
    ]
    for expected in expecteds:
        assert expected in screen_output

    caplog.clear()

    # supercell with 8 molecules
    main([str(c) for c in CRYSTALS] + ["--dry", "--method", "double_shortest", "-z", "8"])

    screen_output = caplog.text
    expecteds = [
        "Read 5 crystals",
        "Valid SuperCell(n=2, m=1, l=1) found for pyrene_I_replaced",
        "Valid SuperCell(n=2, m=1, l=1) found for pyrene_II_replaced",
        "Valid SuperCell(n=1, m=4, l=1) found for pyrene_III_replaced",
        "Valid SuperCell(n=2, m=1, l=1) found for pyrene_IV_replaced",
        "Valid SuperCell(n=1, m=2, l=1) found for pyrene_V_replaced",
    ]
    for expected in expecteds:
        assert expected in screen_output
    

def test_load_crystals():
    loaded_c = load_crystals(CRYSTALS)
    assert len(loaded_c) == 5

    loaded_c = load_crystals(CRYSTALS, excluded_spgs=[2])
    assert len(loaded_c) == 4


def test_check_multipoles():
    mults_file = DATA_DIR / "pyrenex4.dma"
    loaded_c = load_crystals(CRYSTALS)
    assert check_multipoles(loaded_c, mults_file)


def test_check_bonding():
    xyz_file = DATA_DIR / "pyrene.xyz"
    loaded_c = load_crystals(CRYSTALS)
    assert check_bonding(loaded_c, [Molecule.from_xyz_file(xyz_file)])


def test_molecular_axis():
    axis_file = DATA_DIR / "pyrenex4.mols"
    loaded_c = load_crystals(CRYSTALS)
    assert check_molecular_axis(loaded_c, axis_file) 
    