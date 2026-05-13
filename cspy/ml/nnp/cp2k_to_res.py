from cspy.crystal.space_group import SpaceGroup
from cspy.chem import Molecule
from cspy.crystal import UnitCell
from cspy import Crystal
from numpy import pi
from os import remove
from typing import Tuple, Dict, List
import logging

LOG = logging.getLogger(__name__)


def lattice_reader(cell_fname: str) -> Tuple[Dict[str, List[float]], Dict[str, List[float]]]:
    """
    Takes concatenated cells information output of CP2K and converts it to
    lengths and angles dictionaries.

    Args:
        cell_fname (str): Path to the concatenated cell file.

    Returns:
        Tuple[Dict[str, List[float]], Dict[str, List[float]]]: A tuple containing two dictionaries:
            - lengths: Dictionary with keys 'a', 'b', 'c' and their corresponding values as lists of floats.
            - angles: Dictionary with keys 'alpha', 'beta', 'gamma' and their corresponding values as lists of floats in radians.
    The angles are converted from degrees to radians.
    """
    lengths = {'a': [], 'b': [], 'c': []}
    angles = {'alpha': [], 'beta': [], 'gamma': []}
    with open(cell_fname, 'r') as f:
        for line in f:
            parts = line.split()
            if len(parts) > 2 and parts[2] in lengths:
                lengths[parts[2]].append(float(parts[-1]))
            if len(parts) > 3 and parts[3] in angles:
                angles[parts[3]].append(float(parts[-1]) / 180 * pi)
    return lengths, angles


def separate_xyz(pos_fname: str) -> None:
    """
    Converts concatenated XYZ coordinates of the CP2K output to separated numbered XYZ files.

    Args:
        pos_fname (str): Path to the concatenated XYZ file.

    Returns:
        None: The function creates separate XYZ files named '0000.xyz', '0001.xyz', etc., in the current directory.
    Each file contains the coordinates of atoms for a specific time step.
    The files are created in the same directory as the input file.
    """
    counter = -1
    with open(pos_fname, 'r') as f:
        for _, line in enumerate(f):
            parts = line.split()
            if len(parts) == 1:
                counter += 1
            with open(f'{counter:04d}.xyz', 'a') as g:
                g.write(line)


def xyz_to_res(
    xyz_fname: str,
    lv_lengths: List[float],
    lv_angles: List[float],
    sg_number: int = 1
) -> None:
    """
    This function converts a single lattice to a res file.
    Take lattice lengths and angles, space group number, and coordinates of atoms in XYZ format and returns a res file

    Args:
        xyz_fname (str): Path to the XYZ file containing atomic coordinates.
        lv_lengths (List[float]): List of lattice vector lengths [a, b, c].
        lv_angles (List[float]): List of lattice vector angles [alpha, beta, gamma].
        sg_number (int, optional): Space group number. Defaults to 1.

    Returns:
        None: The function saves the converted data to a RES file with the same name as the XYZ file but with a '.res' extension.
    The XYZ file is removed after conversion.
    """
    mol = Molecule.from_xyz_file(xyz_fname)
    sg = SpaceGroup(sg_number)
    uc = UnitCell.from_lengths_and_angles(
        lengths=lv_lengths,
        angles=lv_angles
    )

    cryst = Crystal.from_unit_cell_sg_molecules(
        unit_cell=uc,
        space_group=sg,
        molecules=[mol]
    )
    try:
        cryst.save(f'{xyz_fname[:-4]}.res')
        remove(xyz_fname)
    except Exception as e:
        LOG.error(f'XYZ to RES Conversion failed: {e}')


def cp2k_to_res(cell_vectors_fname: str, all_xyz_fname: str) -> None:
    """
    Takes cp2k cell information file and positions file and returns separated res files

    Args:
        cell_vectors_fname (str): Path to the CP2K cell vectors file.
        all_xyz_fname (str): Path to the CP2K concatenated XYZ coordinates file.

    Returns:
        None: The function creates separate RES files for each time step in the current directory.
    """
    lengths, angles = lattice_reader(cell_vectors_fname)
    LOG.info(f'Lattice lengths: {lengths}')
    separate_xyz(all_xyz_fname)

    for i in range(len(lengths['a'])):
        lv_lengths = [lengths['a'][i], lengths['b'][i], lengths['c'][i]]
        lv_angles = [angles['alpha'][i], angles['beta'][i], angles['gamma'][i]]
        xyz_fname = f'{i:04d}.xyz'
        xyz_to_res(xyz_fname, lv_lengths, lv_angles, sg_number=1)


def main() -> None:
    import argparse
    from cspy.util.logging_config import FORMATS, DATEFMT

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--latticevectors',
        metavar='',
        help='Lattice vectors filename',
        required=True
    )

    parser.add_argument(
        '--xyzs',
        metavar='',
        help='Concatenated XYZ position from the output of CP2K filename',
        required=True
    )
    parser.add_argument(
        '--log-level',
        metavar='',
        help='Logging level',
        default='INFO'
    )

    args = parser.parse_args()
    logging.basicConfig(
        level=args.log_level,
        format=FORMATS.get(args.log_level, FORMATS['INFO']),
        datefmt=DATEFMT
    )
    cp2k_to_res(args.latticevectors, args.xyzs)


if __name__ == '__main__':
    main()
