from cp2k_to_res2 import lattice_reader
import numpy as np
from atom import Atom
from snapshot import Snapshot
from typing import Dict


def read_xyz(pos_fname: str) -> Dict[float, Dict[str, list]]:
    """
    Converts concatenated XYZ coordinates of the CP2K output to a dictionary of XYZ coordinates.
    Keys of the dictionary are the time step.

    Args:
        pos_fname (str): Path to the concatenated XYZ file.

    Returns:
        Dict[float, Dict[str, list]]: A dictionary where keys are time steps and
                                      values are dictionaries with 'elements' and 'positions' keys.
                                      'elements' is a list of element symbols and 'positions' is a list of positions
                                      corresponding to the elements.
    """
    atoms = {}
    with open(pos_fname, 'r') as f:
        reset = True
        for i, line in enumerate(f.readlines()):
            parts = line.split()
            if len(parts) == 1:
                try:
                    atoms[timestep] = {
                        'elements': elements,
                        'positions': positions
                    }
                except Exception:
                    pass
                reset = True
                continue
            if reset:
                parts = line.split(',')
                timestep = float(parts[0].split()[-1])
                reset = False
                elements = []
                positions = []
                continue
            elements.append(parts[0])
            positions.append([
                float(parts[1]),
                float(parts[2]),
                float(parts[3])
            ])
        # Store the last snapshot
        atoms[timestep] = {
            'elements': elements,
            'positions': positions
        }
    return atoms


def cp2k_to_n2p2(cell_fname: str, pos_fname: str, t_min: float, t_max: float,
                 output_fname: str = 'input.data', label_prefix: str = None) -> None:
    """
    Converts CP2K output files to N2P2 input format.

    Args:
        cell_fname (str): Path to the CP2K cell vectors file.
        pos_fname (str): Path to the CP2K concatenated XYZ coordinates file.
        t_min (float): Minimum time step to consider.
        t_max (float): Maximum time step to consider.
        output_fname (str): Name of the output N2P2 input file.
        label_prefix (str): Prefix for the structure label in the output file.

    Returns:
        None: Writes the N2P2 input data to the specified output file.
    """
    lattice_vectors_dict = lattice_reader(cell_fname)
    atoms_dict = read_xyz(pos_fname)
    for timestep in lattice_vectors_dict.keys():
        if timestep < t_min or timestep > t_max:
            continue

        # skip if the timestep doesn't exist in atoms dict
        try:
            atoms_dict[timestep]['elements']
        except Exception:
            continue
        atoms = []
        for element, pos in zip(atoms_dict[timestep]['elements'], atoms_dict[timestep]['positions']):
            atoms.append(
                Atom(
                    element,
                    pos,
                    [0, 0, 0],
                    0
                )
            )
        lattice_vectors = lattice_vectors_dict[timestep].tolist()

        if label_prefix:
            label = f'{label_prefix}_{timestep}'
        else:
            label = timestep
        s = Snapshot(
            label=label,
            atoms=atoms,
            energy=0,
            lattice_vectors=lattice_vectors
        )

        s.write(output_fname)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description='Takes concatenated snapshots xyz coordinate and lattice vectors file and returns a concatenated input.data '
                    'file for n2p2'
    )

    parser.add_argument(
        '-o', '--output',
        metavar='',
        help='Output filename',
        default='input.data'
    )

    parser.add_argument(
        '-xyz', '--xyzfile',
        metavar='',
        help='Snapshots xyz filename (typically xxxxx-pos-1.xyz)'
    )

    parser.add_argument(
        '-lv', '--cellfile',
        metavar='',
        help='Lattice vectors filename (usually cell_vectors)'
    )

    parser.add_argument(
        '--tmax',
        type=float,
        metavar='',
        help='A threshold after which snapshots will be ignored. Use this option when the system '
             'explodes after a specific time',
        default=np.inf
    )

    parser.add_argument(
        '--tmin',
        type=float,
        metavar='',
        help='A threshold before which snapshots will be ignored. Use this option when the system '
             'needs relaxation',
        default=0
    )

    parser.add_argument(
        '--label',
        type=str,
        metavar='',
        help='label prefix (i.e. structure ID)',
        default=None
    )

    args = parser.parse_args()

    cp2k_to_n2p2(
        args.cellfile,
        args.xyzfile,
        args.tmin,
        args.tmax,
        output_fname=args.output,
        label_prefix=args.label
    )


if __name__ == '__main__':
    main()
