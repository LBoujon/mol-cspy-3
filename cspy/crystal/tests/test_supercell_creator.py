from pathlib import Path
from cspy.crystal.supercell_creator import GrowShortestMethod, DoubleShortestMethod, create_supercells, SuperCell
from cspy import Crystal 
from cspy.ml.descriptors import PowderPattern
import pytest
from typing import List
from functools import partial
import numpy as np


FILE_DIR = Path(__file__).parent.parent.parent / "threshold" / "test"
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


def get_crystal(filename: str) -> Crystal:
    return Crystal.load(DATA_DIR / filename)


def check_no_change_pxrd(crystal: Crystal, original_pxrd: PowderPattern, supercell: List[int]) -> None:
    new_crystal = crystal.as_P1_supercell(supercell)
    new_crystal_pxrd = new_crystal.calculate_powder_pattern()
    if new_crystal_pxrd is None:
        raise ValueError("pxrd pattern not available")
    new = new_crystal_pxrd.pattern
    original = original_pxrd.pattern
    diff = np.abs(new-original)
    max_diff = max(diff)
    print(f"Max diff between patterns: {max_diff}")

    assert np.allclose(original, new, rtol=1e-2, atol=1e-3)


@pytest.mark.external_binaries
def test_grow_shortest_method() -> None:
    # This test uses the `platon` external app to calculate
    # pXRD patters. If it is not installed it will fail
    pyrene_I = get_crystal("pyrene_I_replaced.res")
    pyrene_I_pxrd = pyrene_I.calculate_powder_pattern()
    if pyrene_I_pxrd is None:
        raise ValueError("pxrd pattern not available")
    check_no_change_pxrd_pyrene_I = partial(check_no_change_pxrd, pyrene_I, pyrene_I_pxrd)
    
    gsm = GrowShortestMethod([pyrene_I], True)
    assert gsm.check_target(8) == [SuperCell(2, 1, 1)]
    check_no_change_pxrd_pyrene_I(gsm.supercells[0])
    assert gsm.check_target(16) == [SuperCell(2, 2, 1)]
    check_no_change_pxrd_pyrene_I(gsm.supercells[0])


    pyrene_III = get_crystal("pyrene_III_replaced.res")
    pyrene_III_pxrd = pyrene_III.calculate_powder_pattern()
    if pyrene_III_pxrd is None:
        raise ValueError("pxrd pattern not available")
    check_no_change_pxrd_pyrene_III = partial(check_no_change_pxrd, pyrene_III, pyrene_III_pxrd)
    gsm = GrowShortestMethod([pyrene_III], False)
    assert gsm.check_target(4) == [SuperCell(1, 2, 1)]
    check_no_change_pxrd_pyrene_III(gsm.supercells[0])
    assert gsm.check_target(6) == [SuperCell(1, 3, 1)]
    check_no_change_pxrd_pyrene_III(gsm.supercells[0])
    assert gsm.check_target(8) == None


@pytest.mark.external_binaries
def test_double_shortest_method() -> None:
    pyrene_I = get_crystal("pyrene_I_replaced.res")
    pyrene_I_pxrd = pyrene_I.calculate_powder_pattern()
    if pyrene_I_pxrd is None:
        raise ValueError("pxrd pattern not available")
    check_no_change_pxrd_pyrene_I = partial(check_no_change_pxrd, pyrene_I, pyrene_I_pxrd)

    dsm = DoubleShortestMethod([pyrene_I], False)
    assert dsm.check_target(8) == [SuperCell(2, 1, 1)]
    check_no_change_pxrd_pyrene_I(dsm.supercells[0])
    assert dsm.check_target(16) == [SuperCell(2, 2, 1)]
    check_no_change_pxrd_pyrene_I(dsm.supercells[0])
    assert dsm.check_target(32) == [SuperCell(2, 2, 2)]
    check_no_change_pxrd_pyrene_I(dsm.supercells[0])

    pyrene_III = get_crystal("pyrene_III_replaced.res")
    pyrene_III = get_crystal("pyrene_III_replaced.res")
    pyrene_III_pxrd = pyrene_III.calculate_powder_pattern()
    if pyrene_III_pxrd is None:
        raise ValueError("pxrd pattern not available")
    check_no_change_pxrd_pyrene_III = partial(check_no_change_pxrd, pyrene_III, pyrene_III_pxrd)
    dsm = DoubleShortestMethod([pyrene_III], False)
    assert dsm.check_target(4) == [SuperCell(1, 2, 1)]
    check_no_change_pxrd_pyrene_III(dsm.supercells[0])
    assert dsm.check_target(8) == [SuperCell(1, 4, 1)]
    check_no_change_pxrd_pyrene_III(dsm.supercells[0])
    assert dsm.check_target(6) == None
    

def test_create_supercells():
    crystals = [Crystal.load(str(p)) for p in CRYSTALS]
    supercells = create_supercells(crystals)
    assert isinstance(supercells, list)
    supercells = create_supercells(crystals, niggli=True)
    assert isinstance(supercells, list)
    for crystal in supercells:
        assert len(crystal.unit_cell_molecules()) == 4
    supercells = create_supercells(crystals, niggli=True, target=8)
    assert isinstance(supercells, list)
    for crystal in supercells:
        assert len(crystal.unit_cell_molecules()) == 8
    supercells = create_supercells(crystals, niggli=True, target=8, creation_method=GrowShortestMethod)
    assert supercells is None
