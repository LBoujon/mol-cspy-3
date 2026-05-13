from snapshot import Snapshot
from atom import Atom


class Structure(Snapshot):
    """An inherited class for conversion file formats between RuNNer and VASP packages."""

    def __init__(self, label=None, atoms=None, energy=None, box_bounds=None, lattice_vectors=None):
        Snapshot.__init__(self, label, atoms, energy, box_bounds=box_bounds, lattice_vectors=lattice_vectors)

    def read_POSCAR(self, fname='POSCAR'):
        """
        Read POSCAR file and returns the label of structure,
        atoms elements, lattice vectors, and atomic positions.
        Based on the POSCAR, inputs can be Direct or Fractional.
        """
        with open(fname, 'r') as f:
            lines = f.readlines()
            label = lines[0].split('/')[-1].split()[0]
            if label[-4:] == '.res':
                label = label[:-4]
            elements = [s_ for s_ in lines[5].split()]
            counts = [int(c) for c in lines[6].split()]
            elements_list = []
            for i in range(len(elements)):
                for j in range(counts[i]):
                    elements_list.append(elements[i])

            lattice_vectors = []
            for line in lines[2:5]:
                parts = line.split()
                lattice_vectors.append([float(v) for v in parts])

            frac_positions = []
            for line in lines[8:]:
                parts = line.split()
                frac_positions.append([float(p) for p in parts])

            if len(elements_list) != len(frac_positions):
                print('len(elements_list) != len(frac_positions)')
                return 
        return label, elements_list, lattice_vectors, frac_positions

    def read_OUTCAR(self, fname='OUTCAR'):
        """
        Reads OUTCAR and returns forces and total energy in VASP units.
        """
        with open(fname, 'r') as in_file:
            # loop over lines in file
            for line in in_file:
                # read the force section
                if "POSITION" in line:
                    next(in_file)
                    # write data into first sample in the data set (assuming having only one sample)
                    forces = []
                    while True:
                        line = next(in_file)
                        parts = line.split()
                        if len(parts) != 6:
                            break
                        forces.append([float(frc) for frc in parts[3:]])

                # read total energy
                if "  free  energy   TOTEN" in line:
                    total_energy = float(line.split()[-2])
                    break

        return forces, total_energy

    def read_structure(self, poscar, outcar, fractioanl=False):
        label, elements_list, lattice_vectors, frac_positions = self.read_POSCAR(fname=poscar)
        forces, total_energy = self.read_OUTCAR(fname=outcar)
        self.name = label
        self.lattice_vectors = lattice_vectors
        self.energy = total_energy
        atoms = []
        for i in range(len(elements_list)):
            pos = frac_positions[i]
            if fractioanl:
                pos = self.to_direct(frac_positions[i])
            atoms.append(Atom(elements_list[i],
                              pos,
                              forces[i],
                              0
                              )
                         )
        self.atoms = atoms


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(
        description='Takes VASP POSCAR and OUTCAR files and writes n2p2 data file.'
                    'It assumes that POSCAR coordinates are fractional. If not, code'
                    'needs to be modified.'
        )

    parser.add_argument('-o', '--output',
                        metavar='',
                        help='Output filename',
                        default='input.data')

    parser.add_argument('--poscar',
                        metavar='',
                        help='POSCAR filename')

    parser.add_argument('--outcar',
                        metavar='',
                        help='OUTCAR filename')

    args = parser.parse_args()

    s = Structure()
    s.read_structure(poscar=args.poscar, outcar=args.outcar)
    s.write(args.output)
