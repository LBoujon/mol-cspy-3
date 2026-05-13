import logging
import os
from typing import Union
import numpy as np

from cspy.crystal import Crystal

LOG = logging.getLogger(__name__)

# TODO: This whole file can be replaced with ASE tools, but we keep it for now

def _kpoints_per_klength(
    klength: float,
    kspacing: float = 0.05,
) -> int:
    """
    Calculate the number of k-points per reciprocal lattice vector length.
    Must have at least 1 k-point. Rounds up as kspacing is maximum spacing desired.

    Parameters
    ----------
    klength : float
        The length of the reciprocal lattice vector.
    kspacing : float, optional
        The desired k-point spacing. Default is 0.05.

    Returns
    -------
    int
        The number of k-points per reciprocal lattice vector length, rounded up.
    """
    nk_raw = max(1, klength / kspacing + 0.5)
    return int(np.ceil(nk_raw))


def make_poscar(
    crystal: Crystal,
    elem_set: set,
    fractional: bool = True,
    scaling: str = "1.000000",
    comment: str = "Made using CSPy",
) -> list:
    """
    Create a POSCAR file for VASP.

    Parameters
    ----------
    crystal : Crystal
        The crystal structure for which to create the POSCAR file.
    elem_set : set
        A set of elements present in the crystal structure.
    fractional : bool, optional
        If True, coordinates will be in fractional format; if False, in Cartesian format.
        Default is True.
    scaling : str, optional
        Scaling factor for the lattice vectors. Default is "1.000000".
    comment : str, optional
        A comment line for the POSCAR file. Default is "Made using CSPy".

    Returns
    -------
    list
        A list of strings representing the POSCAR file content.
    """
    elem_list = crystal.asymmetric_unit.elements
    elem_counts = [elem_list.count(el) for el in elem_set]
    poscar = []
    latt_vecs = crystal.unit_cell.lattice
    poscar.append(comment)
    poscar.append(scaling)
    for i in range(3):
        poscar.append(
            "  " + "{:12.6f} {:12.6f} {:12.6f}".format(*latt_vecs[i])
        )
    poscar.append(
        "  " + ("{:>4s}" * len(elem_set)).format(*[str(el) for el in elem_set])
    )
    poscar.append(
        "  " + ("{:4d}" * len(elem_set)).format(*elem_counts)
    )

    uc_atoms = crystal.unit_cell_atoms()
    ordered_uc_atoms = []
    for el in elem_set:
        ordered_uc_atoms += [
            i for i, e in zip(uc_atoms["asym_atom"], uc_atoms["element"])
            if e == el.atomic_number
        ]

    ordered_coords = []
    if fractional:
        poscar.append("Direct")
        for i in ordered_uc_atoms:
            ordered_coords.append(uc_atoms["frac_pos"][i])
    else:
        poscar.append("Cartesian")
        for i in ordered_uc_atoms:
            ordered_coords.append(uc_atoms["cart_pos"][i])
    LOG.debug("Ordered coordinates in make_poscar are:")
    LOG.debug("\n".join([str(c) for c in ordered_coords]))
    for coord in ordered_coords:
        poscar.append(
            "  " + "{:12.6f} {:12.6f} {:12.6f}".format(*coord)
        )
    return poscar


def make_kpoints(
    crystal: Crystal,
    numk: Union[int, None] = None,
    kspacing: Union[float, None] = None,
) -> list:
    """
    Create a KPOINTS file for VASP.

    Parameters
    ----------
    crystal : Crystal
        The crystal structure for which to create the KPOINTS file.
    numk : int or list of int, optional
        The number of k-points in each direction. If a single int is provided, it
        applies to all three directions. If a list of three ints is provided, it
        specifies the number of k-points in the x, y, and z directions respectively.
    kspacing : float, optional
        The desired k-point spacing. If specified, it overrides the `numk` parameter.

    Returns
    -------
    list
        A list of strings representing the KPOINTS file content.
    """
    kpoints = []
    if numk and kspacing:
        raise Exception(
            "Must specify ONE of A) number of k-points (a single int or 3 ints), or B) a k-spacing"
        )
    kpoints.append(f"K-point grid for {crystal.titl}")
    kpoints.append("0")
    if kspacing is not None:
        kpoints.append("Gamma")
        k_vec_norms = [
            np.linalg.norm(kvec) for kvec in crystal.unit_cell.reciprocal_lattice
        ]
        nkvec = [_kpoints_per_klength(knorm, kspacing) for knorm in k_vec_norms]
        kpoints.append("{:4d} {:4d} {:4d}".format(*nkvec))
    elif isinstance(numk, int):
        kpoints.append("Auto")
        kpoints.append(str(numk))
    elif isinstance(numk, list) and len(numk) == 3:
        kpoints.append("Gamma")
        kpoints.append("  ".join(str(k) for k in numk))
    return kpoints


def write_potcar(
    working_directory: str,
    elements: set,
    potcar_path: str = "POTCAR"
) -> None:
    """
    Write the POTCAR file for VASP by concatenating the POTCAR files for each element.

    Parameters
    ----------
    working_directory : str
        The directory where the POTCAR file will be written.
    elements : set
        A set of elements for which the POTCAR files will be concatenated.
    potcar_path : str
        The path to the directory containing the POTCAR files for each element.
    """
    potcar_file = os.path.join(working_directory, "POTCAR")
    with open(potcar_file, "w+") as final_potcar:
        for elem in elements:
            elem_potcar_path = os.path.join(potcar_path, str(elem), "POTCAR")
            with open(elem_potcar_path, "r") as elem_potcar:
                for line in elem_potcar:
                    final_potcar.write(line)


def write_incar(
    working_directory: str,
    incar_keywords: dict
) -> None:
    """
    Write the INCAR file for VASP with the given keywords.

    Parameters
    ----------
    working_directory : str
        The directory where the INCAR file will be written.
    incar_keywords : dict
        A dictionary of INCAR keywords and their values.
    """
    incar_path = os.path.join(working_directory, "INCAR")
    with open(incar_path, "w") as f:
        for key, val in incar_keywords.items():
            f.write(f"{key} = {val}\n")


def create_vasp_inputs(
    input_crystal: Crystal,
    working_directory: str,
    settings: dict
) -> None:
    """
    Create VASP input files (POSCAR, KPOINTS, POTCAR, INCAR) for the given crystal.
    
    Parameters
    ----------
    input_crystal : Crystal
        The crystal structure for which to create the VASP input files.
    working_directory : str
        The directory where the VASP input files will be written.
    settings : dict
        A dictionary containing settings for the VASP input files, including:
        - 'potcar_path': Path to the directory containing POTCAR files.
        - 'kspacing': K-point spacing for the KPOINTS file.
        - 'incar_settings': A dictionary of INCAR keywords and their values.
    """

    if not working_directory:
        raise TypeError(
            f'working_directory is not correctly set. Current value: {working_directory}'
        )
    potcar_path = settings['potcar_path']
    kspacing = settings['kspacing']
    try:
        crys = input_crystal.standardized()
        crys = crys.as_P1()
    except Exception:
        crys = input_crystal.as_P1()

    basename = crys.titl
    elem_set = set(crys.asymmetric_unit.elements)

    with open(os.path.join(working_directory, "POSCAR"), "w+") as f:
        f.write(
            "\n".join(
                make_poscar(
                    crys,
                    elem_set=elem_set,
                    comment=basename + " made using CSPy"
                )
            )
        )
    with open(os.path.join(working_directory, "KPOINTS"), "w+") as f:
        f.write(
            "\n".join(
                make_kpoints(
                    crys,
                    kspacing=kspacing
                )
            )
        )
    write_potcar(
        working_directory=working_directory,
        elements=elem_set,
        potcar_path=potcar_path
    )
    write_incar(
        working_directory=working_directory,
        incar_keywords=settings['incar_settings']
    )

    LOG.info(
        "Finished writing POSCAR, KPOINTS, and POTCAR for " + crys.titl
    )

