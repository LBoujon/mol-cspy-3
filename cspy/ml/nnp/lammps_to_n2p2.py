from atom import Atom
from snapshot import Snapshot
import LAMMPSThermoExtractor as le
from pathlib import Path
import numpy as np

def lammps_log_reader(fname):
    l = le.LAMMPSLogFile(fname)
    l.parse()
    block = l.thermodata_blocks[-1]
    names_list = block.names
    values_list = block.values

    values_dict = {}
    for v in values_list:
        values_dict[v[0]] = {names_list[1]: v[1],
                             names_list[2]: v[2],
                             names_list[3]: v[3],
                             names_list[4]: v[4]}
    return values_dict


def coulomb_thermo_reader(fname):
    values_dict = {}
    with open(fname) as f:
        lines = f.readlines()
        names_list = lines[0].split()
        for line in lines[1:]:
            v = line.split()
            values_dict[int(v[0])] = {names_list[i]: v[i]
                                     for i in range(1, len(v))}
    return values_dict


def snapshot_file_reader(fname, values_dict = None):
    with open(fname, 'r') as f:
        lines = f.readlines()

        snapshot_timestep = int(lines[1].split()[0])

        if values_dict == None:
            epair = None
        elif snapshot_timestep in values_dict.keys():
            epair = values_dict[snapshot_timestep]['E_pair']
        else:
            epair = None

        no_of_atoms = int(lines[3].split()[0])

        box_bounds = []
        for line in lines[5:8]:
            parts = line.split()
            bb_temp = []
            for part in parts:
                bb_temp.append(float(part))
            box_bounds.append(bb_temp)
        box_bounds = np.array(box_bounds)

        # lattice_vectors = []
        # for line in lines[5:8]:
        #     parts = line.split()
        #     lattice_vectors.append([float(parts[0]),
        #                            float(parts[1]),
        #                            float(parts[2])])
        atoms_in_box = []
        for line in lines[9:]:
            parts = line.split()
            atoms_in_box.append(Atom(str(parts[0]),
                                     [float(parts[1]), float(parts[2]), float(parts[3])],
                                     [float(parts[4]), float(parts[5]), float(parts[6])],
                                     float(parts[7])))

        if len(atoms_in_box) != no_of_atoms:
            print('Error! The number of imported atoms ({}) is different from the '
                  'header of the snapshot file ({}).'.format(len(atoms_in_box), no_of_atoms))
            return None

    return Snapshot(snapshot_timestep, box_bounds, atoms_in_box, epair)


def n2p2_data_writer_batch(values_dict, snapshot_dir, output):

    snapshot_files = Path(snapshot_dir).glob('snapshot*.dat')

    with open(output, 'w') as f:
        for sf in snapshot_files:
            snapshot = snapshot_file_reader(sf, values_dict)
            f.write('begin\n')
            f.write('comment step {:d}\n'.format(snapshot.name))
            for lv in snapshot.lattice_vectors:
                f.write('lattice  {:>14.10f}  {:>14.10f}  {:>14.10f}\n'.format(*lv))
            for atom in snapshot.atoms:
                f.write('atom  {:>10.6f}  {:>10.6f}  {:>10.6f}  {:2s}  {:>+8.6f} '
                        ' {}  {:>10.6f}  {:>10.6f}  {:>10.6f}\n'.format(
                        atom.position[0], atom.position[1], atom.position[2],
                        atom.element, atom.charge, 0,
                        atom.force[0], atom.force[1], atom.force[2])
                )
            f.write('energy  {}\n'.format(snapshot.energy))
            f.write('charge  {}\n'.format(snapshot.net_charge))
            f.write('end\n')


def n2p2_data_writer_single(values_dict, snapshot_file, output):

    with open(output, 'w') as f:
        snapshot = snapshot_file_reader(snapshot_file, values_dict)
        f.write('begin\n')
        f.write('comment step {:d}\n'.format(snapshot.name))
        for lv in snapshot.lattice_vectors:
            f.write('lattice  {:>14.10f}  {:>14.10f}  {:>14.10f}\n'.format(*lv))
        for atom in snapshot.atoms:
            f.write('atom  {:>10.6f}  {:>10.6f}  {:>10.6f}  {:2s}  {:>+8.6f} '
                    ' {}  {:>10.6f}  {:>10.6f}  {:>10.6f}\n'.format(
                atom.position[0], atom.position[1], atom.position[2],
                atom.element, atom.charge, 0,
                atom.force[0], atom.force[1], atom.force[2])
            )
        f.write('energy  {}\n'.format(snapshot.energy))
        f.write('charge  {}\n'.format(snapshot.net_charge))
        f.write('end\n')


if __name__ == '__main__':
    
    import argparse
    import os
    
    parser = argparse.ArgumentParser(
        description='Takes snapshot and lammps log files as input and writes '
                    'n2p2 data file')

    parser.add_argument('-o', '--output',
                        metavar='',
                        help='Output filename',
                        default='input.data')

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-l', '--logfile',
                        metavar='',
                        help='lammps log file')

    group.add_argument('-tf', '--thermofile',
                       metavar='',
                       help='lammps customized thermo file')

    group2 = parser.add_mutually_exclusive_group(required=True)
    group2.add_argument('-sd', '--snapdir',
                        metavar='',
                        help='Path to to the folder where the snapshots are')

    group2.add_argument('-sf', '--snapfile',
                        metavar='',
                        help='Path to to the snapshot file')

    args = parser.parse_args()


    if args.logfile:
        values_dict = lammps_log_reader(args.logfile)
    elif args.thermofile:
        values_dict = coulomb_thermo_reader(args.thermofile)

    if args.snapdir:
        n2p2_data_writer_batch(values_dict, args.snapdir, args.output)
    elif args.snapfile:
        n2p2_data_writer_single(values_dict, args.snapfile, args.output)
