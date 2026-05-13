import logging
from os import remove
from typing import Dict

import numpy as np

from cspy.chem import Molecule
from cspy.crystal import UnitCell
from cspy.crystal.space_group import SpaceGroup
from cspy import Crystal
from numpy import pi

LOG = logging.getLogger(__name__)


def lattice_reader(cell_fname: str) -> Dict[float, np.ndarray]:
    """
    Takes cell_vectors file and returns a dictionary with keys=timestep and
    values=corresponding lattice vectors as np.arrays.

    Parameters
    ----------
    cell_fname : str
        Filename of the CP2K cell vectors file.

    Returns
    -------
    Dict[float, np.ndarray]
        Dictionary with keys as timesteps and values as lattice vectors.
    """
    lattice_vectors_dict = {}
    with open(cell_fname, 'r') as f:
        for line in f.readlines():
            if line[0] == '#':
                continue
            parts = line.split()
            timestep = float(parts[0])
            a = [float(parts[2]), float(parts[3]), float(parts[4])]
            b = [float(parts[5]), float(parts[6]), float(parts[7])]
            c = [float(parts[8]), float(parts[9]), float(parts[10])]

            lattice_vectors_dict[timestep] = np.array([a, b, c])

    return lattice_vectors_dict


def separate_xyz(pos_fname: str) -> None:
    """
    Converts concatenated XYZ coordinates of the CP2K output to separated numbered XYZ files.

    Parameters
    ----------
    pos_fname : str
        Filename of the CP2K positions file containing concatenated XYZ coordinates.
    """
    counter = -1
    with open(pos_fname, 'r') as f:
        for i, line in enumerate(f.readlines()):
            parts = line.split()
            if len(parts) == 1:
                counter += 1
            with open(f'{counter:04d}.xyz', 'a') as g:
                g.write(line)


def xyz_to_res(xyz_fname: str, lattice_vectors: np.ndarray, sg_number: int = 1) -> None:
    """
    Converts a single lattice to a res file.
    Takes lattice vectors, space group number, and coordinates of atoms in XYZ format and returns a res file.

    Parameters
    ----------
    xyz_fname : str
        Filename of the XYZ file containing atomic coordinates.
    lattice_vectors : np.ndarray
        Lattice vectors as a 3x3 numpy array.
    sg_number : int, optional
        Space group number, by default 1 (P1).
    """
    mol = Molecule.from_xyz_file(xyz_fname)
    sg = SpaceGroup(sg_number)
    uc = UnitCell(lattice_vectors)

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
    Takes cp2k cell information file and positions file and returns separated res files.

    Parameters
    ----------
    cell_vectors_fname : str
        Filename of the CP2K cell vectors file.
    all_xyz_fname : str
        Filename of the CP2K positions file containing concatenated XYZ coordinates.
    """
    lattice_vectors_dict = lattice_reader(cell_vectors_fname)
    separate_xyz(all_xyz_fname)

    for i in range(len(lattice_vectors_dict.keys())):
        xyz_fname = f'{i:04d}.xyz'
        with open(xyz_fname, 'r') as f:
            second_line = f.readlines()[1]
            parts = second_line.split(',')
            subparts = parts[0].split()
            timestep = float(subparts[-1])
        try:
            xyz_to_res(xyz_fname, lattice_vectors_dict[timestep], sg_number=1)
            with open('all.res', 'a') as f:
                with open(f'{i:04d}.res', 'r') as g:
                    for line in g.readlines():
                        f.write(line)
                f.write('\nEND\n')
            remove(f'{i:04d}.res')
        except Exception as e:
            LOG.error(f'Something wrong about {xyz_fname}: {e}')


def main():
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
        default='INFO',
        choices=FORMATS.keys()
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
