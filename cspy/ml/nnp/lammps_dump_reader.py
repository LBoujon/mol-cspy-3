from atom import Atom
from snapshot import Snapshot
from collections import OrderedDict
import numpy as np

lammps_dict = {
    1: 'O',
    2: 'H',
    3: 'C'
}

def dump_file_reader(fname):
    snapshots = OrderedDict()
    with open(fname, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if 'ITEM: TIMESTEP' in line:
                timestep = int(lines[i+1].split()[0])

            if 'ITEM: NUMBER OF ATOMS' in line:
                n_atoms = int(lines[i+1].split()[0])

            if 'ITEM: BOX BOUNDS' in line:
                bbox = []
                for j in range(i+1, i+4):
                    parts = lines[j].split()
                    bbox_temp = [float(p) for p in parts]
                    bbox.append(bbox_temp)
                bbox = np.array(bbox)

                # Orthorhombic box
                if bbox.shape[1] == 2:

                    xlo, xhi = bbox[0]
                    ylo, yhi = bbox[1]
                    zlo, zhi = bbox[2]

                    lx, ly, lz = xhi - xlo, yhi - ylo, zhi - zlo

                    lattice_vectors = [[lx, 0, 0],
                                       [0, ly, 0],
                                       [0, 0, lz]]

                # Triclinic box
                elif bbox.shape[1]==3:

                    xlo_bound, xhi_bound, xy = bbox[0]
                    ylo_bound, yhi_bound, xz = bbox[1]
                    zlo_bound, zhi_bound, yz = bbox[2]

                    xlo = xlo_bound - min(0.0, xy, xz, xy + xz)
                    xhi = xhi_bound - max(0.0, xy, xz, xy + xz)
                    ylo = ylo_bound - min(0.0, yz)
                    yhi = yhi_bound - max(0.0, yz)
                    zlo = zlo_bound
                    zhi = zhi_bound

                    lx, ly, lz = xhi - xlo, yhi - ylo, zhi - zlo
                    lattice_vectors = [[lx, 0, 0],
                                       [xy, ly, 0],
                                       [xz, yz, lz]]

            if 'ITEM: ATOMS' in line:
                atoms = []
                for j in range(i+1, i+n_atoms):
                    parts = lines[j].split()
                    element = lammps_dict[int(parts[1])]
                    pos = [float(parts[2]), float(parts[3]), float(parts[4])]
                    force = [0, 0, 0]
                    charge = 0
                    atoms.append(Atom(element=element, position=pos, force=force, charge=charge))

                snapshots[timestep] = Snapshot(
                    label=timestep,
                    atoms=atoms,
                    energy=0,
                    lattice_vectors=lattice_vectors
                )

                for atom in snapshots[timestep].atoms:
                    atom.position = snapshots[timestep].to_direct(atom.position)
    return snapshots




