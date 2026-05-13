from snapshot import Snapshot
from atom import Atom, atoms_dict
from cspy import Crystal

def res_to_n2p2(input_fname):
    """
    Takes a .res file and returns a Snapshot object
    Args:
        input_fname: str
            input file name of the .res file.

    Returns:
        structure: Snapshot Object

    """
    c = Crystal.load(input_fname)
    atoms = []
    for el, pos in zip(c.unit_cell_atoms()['element'], c.unit_cell_atoms()['cart_pos']):
        symbol = atoms_dict[el]
        atom_ = Atom(
            element=symbol,
            position=pos,
            force=[0, 0, 0],
            charge=0
        )
        atoms.append(atom_)

    structure = Snapshot(
        label=c.titl,
        atoms=atoms,
        lattice_vectors=c.unit_cell.lattice,
        energy=0
    )
    return structure


if __name__ == '__main__':

    import argparse
    import os

    parser = argparse.ArgumentParser(
        description='Takes res files and writes n2p2 data file.'
    )

    parser.add_argument(
            "files",
            type=str,
            nargs='+',
            help='res files to convert',
    )

    parser.add_argument('-o', '--output',
                        metavar='',
                        help='Output filename. If more that one res file is converted, it writes all the converted'
                             'file into the same output file.',
                        default='input.data')

    args = parser.parse_args()

    for file in args.files:
        s = res_to_n2p2(file)
        s.write(args.output)

