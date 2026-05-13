from snapshot import Snapshot
from atom import Atom


def input_data_reader(fname):
    snapshots = {}
    with open(fname, 'r') as f:
        lattice_vectors = []
        atoms = []
        for line in f.readlines():
            parts = line.split()

            if parts[0] == 'atom':
                atoms.append(
                    Atom(
                        parts[4],  # element
                        [float(parts[1]), float(parts[2]), float(parts[3])],  # pos
                        [float(parts[7]), float(parts[8]), float(parts[9])],  # force
                        float(parts[5]),  # charge
                    )
                )
                continue

            if parts[0] == 'lattice':
                lattice_vectors.append(
                    [
                        float(parts[1]),
                        float(parts[2]),
                        float(parts[3])
                    ]
                )
                continue
            if parts[0] == 'comment':
                try:
                    label = parts[1]
                except:
                    print('Error! Label is missing')
                continue
            if parts[0] == 'energy':
                energy = float(parts[1])
                continue

            # Lines below are not needed because the total charge is calculated as the sum of the atomic charges!
            # if parts[0] == 'charge':
            #     charge = parts[1]
            #     continue

            if parts[0] == 'end':
                # Save snapshot and reset the lists
                snapshots[label] = Snapshot(
                    label=label,
                    atoms=atoms,
                    energy=energy,
                    lattice_vectors=lattice_vectors
                )

                lattice_vectors = []
                atoms = []
                continue

    return snapshots

def get_ids_dict(fname):
    ids_dict = {}
    with open(fname, 'r') as f:
        counter = 0
        for line in f.readlines():
            parts = line.split()
            if parts[0] == 'comment':
                ids_dict[counter] = parts[1]
                counter += 1
    return ids_dict

def read_energy_prediction(fname):
    predicted_energy_dict = {}
    with open(fname, 'r') as f:
        for line in f.readlines():
            if line[0] == '#':
                continue
            parts = line.split()
            predicted_energy_dict[int(parts[0])] = {
                'ref': float(parts[2]),
                'nnp': float(parts[3])
            }
    return predicted_energy_dict

def read_force_predictions(fname):
    predicted_force_dict = {}
    with open(fname, 'r') as f:
        for line in f.readlines():
            if line[0] == '#':
                continue
            parts = line.split()
            if int(parts[0]) not in predicted_force_dict.keys():
                predicted_force_dict[int(parts[0])] = {
                    'ref' : [],
                    'nnp' : []
                }
            predicted_force_dict[int(parts[0])]['ref'].append(float(parts[2]))
            predicted_force_dict[int(parts[0])]['nnp'].append(float(parts[3]))
    return predicted_force_dict

def input_data_to_res(fname, output_fname='all.res'):
    from os import remove
    snapshots = input_data_reader(fname)
    for key in snapshots.keys():
        s = snapshots[key]
        s.write_res()
        with open(output_fname, 'a') as f:
            with open(f'{s.name}.res', 'r') as g:
                for line in g.readlines():
                    f.write(line)
            f.write('\nEND\n')
        remove(f'{s.name}.res')

