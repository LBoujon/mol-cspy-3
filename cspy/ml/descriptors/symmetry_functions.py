import argparse
import io
import logging
import sqlite3
import time
from collections import defaultdict
from itertools import combinations_with_replacement
from multiprocessing import Pool

import numpy as np
from cspy import Crystal
from cspy.db import CspDataStore
from scipy.spatial import cKDTree as KDTree

LOG = logging.getLogger("CSPy")
np.set_printoptions(threshold=np.inf)

DEFAULT_PARAMETERS = {
    "eta_r": 50,
    "eta_a": 5,
    "cutoff_r": 9.3,
    "cutoff_a": 9.3,
    "zeta": 70,
    "rs_r": np.linspace(1, 9.3, 32),
    "rs_a": np.linspace(1, 9.3, 8),
    "theta": np.linspace(0, np.pi, 8),
    "only_intermolecular": False,
    "separate_radial": True,
    "separate_angular": True,
}

global_var_dict = {}

def set_global_descriptors(descriptors):
    global_var_dict['descriptors'] = descriptors


def get_global_descriptors():
    return global_var_dict['descriptors']


def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out, allow_pickle=True)


def adapt_array(arr):
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())

class SymmetryFunctions:
    def __init__(self, labels, n_radial, n_angular, n_theta):
        self.labels = labels
        self.unique_atoms = np.unique(labels).tolist()
        self.pairs = list(
            tuple(sorted(x))
            for x in combinations_with_replacement(self.unique_atoms, 2)
        )
        self.radial = np.zeros(
            (self.n_atoms, self.n_atom_types * n_radial), dtype=np.float64
        )
        self.angular = np.zeros(
            (self.n_atoms, self.n_pairs * n_angular * n_theta), dtype=np.float64
        )
        self.nangular = n_angular * n_theta
        self.nradial = n_radial

    @property
    def n_atoms(self):
        return len(self.labels)

    @property
    def n_atom_types(self):
        return len(self.unique_atoms)

    @property
    def n_pairs(self):
        return len(self.pairs)

    def set_radial(self, i, arr):
        self.radial[i, :] = arr[:]

    def set_angular(self, i, arr):
        self.angular[i, :] = arr[:]

    def get_angular(self, i, pair):
        idx = self.pairs.index(tuple(sorted(pair)))
        l, u = idx * self.nangular, (idx + 1) * self.nangular
        return self.angular[i, l:u]

    def get_radial(self, i, atom):
        idx = self.unique_atoms.index(atom)
        l, u = idx * self.nradial, (idx + 1) * self.nradial
        return self.radial[i, l:u]

    def as_flat_matrix(self):
        return np.hstack((self.radial, self.angular))

    def __repr__(self):
        return f"<SymmetryFunctions: natom={self.n_atoms}>"

    @classmethod
    def from_crystal(cls, crys, **kwargs):
        #if len(crys.asym_mols()) > 1:
        #    raise NotImplementedError("Only implemented for Z'=1 crystals")
        args = DEFAULT_PARAMETERS.copy()
        args.update(kwargs)

        assert len(args["rs_a"]) == len(
            args["theta"]
        ), "Angular R_s and Theta lists must be the same length."

        # Generate a finite cluster of molecules to the largest cutoff given
        # This is done by centroid-centroid distances and so should be sufficient
        nearest_atoms = atoms_in_radius(crys, max((args["cutoff_r"], args["cutoff_a"])))

        # Get xyz data
        cm_xyz = []
        numbers = []
        labels = []
        #mol = crys.asym_mols()[0]
        for mol in crys.asym_mols():
            cm_xyz += list(mol.positions.copy())
            numbers += list(mol.atomic_numbers)
            labels += [e.symbol for e in mol.elements]

        unique_atoms = np.unique(numbers)

        if not args["separate_radial"]:
            raise NotImplementedError
        if not args["separate_angular"]:
            raise NotImplementedError

        symf = cls(labels, len(args["rs_r"]), len(args["rs_a"]), len(args["theta"]))

        for idx, cm_atom_xyz in enumerate(cm_xyz):
            symf.set_radial(
                idx,
                calc_radial_function(
                    cm_xyz,
                    unique_atoms,
                    nearest_atoms,
                    cm_atom_xyz,
                    args["cutoff_r"],
                    args["rs_r"],
                    args["eta_r"],
                    args["only_intermolecular"],
                    args["separate_radial"],
                ),
            )
            symf.set_angular(
                idx,
                calc_angular_function(
                    cm_xyz,
                    unique_atoms,
                    nearest_atoms,
                    cm_atom_xyz,
                    args["cutoff_a"],
                    args["rs_a"],
                    args["theta"],
                    args["eta_a"],
                    args["zeta"],
                    args["only_intermolecular"],
                    args["separate_angular"],
                ),
            )
        return symf


class NearestAtoms:
    def __init__(self, atomic_numbers, atomic_positions):
        self.nums = atomic_numbers
        self.positions = atomic_positions
        self.tree = KDTree(self.positions)

    def neighbours(self, pt, r):
        idx = self.tree.query_ball_point(pt, r)
        return self.nums[idx], self.positions[idx]


def atoms_in_radius(c, radius):
    atoms_dict = c.atoms_in_radius(radius)
    return NearestAtoms(atoms_dict["element"], atoms_dict["cart_pos"])


def calc_radial_function(
    cm_xyz,
    unique_atoms,
    nearest_atoms,
    cm_atom_xyz,
    cutoff,
    r_s_lst,
    eta,
    only_intermolecular=False,
    separate_radial=True,
):

    """
    Radial function (eq (3) of Smith et al. (2017))
    For a given atom returns a list of radial functions of length r_s

    mol_xyz : list of xyz of all atoms in a finite cluster larger than cutoff not in the central molecule
    cm_atom_xyz : xyz of a given atom in the central molecule
    cutoff : float, max L1 distance where atom contributions are included
    r_s : list of hyperparameters controlling position of each gaussian
    eta : float, hyperparameter controlling width of each gaussian
    """
    els, pos = nearest_atoms.neighbours(cm_atom_xyz, cutoff)
    centered = pos - cm_atom_xyz

    distances = np.linalg.norm(centered, axis=1)
    center_atom = distances < 1e-3
    fc = 0.5 * np.cos(distances * np.pi / cutoff) + 0.5
    fc[center_atom] = 0

    if only_intermolecular:
        for cm_atom_xyz in cm_xyz:
            fc[(np.linalg.norm(pos - cm_atom_xyz, axis=1) < 1e-3)] = 0

    g = np.square(distances[:, np.newaxis] - r_s_lst)

    #unique_atoms = np.unique(els)

    if separate_radial:
        G1_list = []
        for atom in unique_atoms:
            fc_temp = np.copy(fc)
            fc_temp[els != atom] = 0
            #        G1_list.append(np.sum(np.exp(g * -eta) * fc_temp[:, np.newaxis], axis=0))
            G1_list.append(fc_temp.dot(np.exp(g * -eta)))
        G1_list = np.array(G1_list)
        return np.ravel(G1_list)

    G1 = fc.dot(np.exp(g * -eta))

    return np.ravel(G1)


def calculate_symmetry_functions(res_file, **kwargs):
    """
    Based on descriptor described in: ANI-1: an extensible neural network
    potential with DFT accuracy at force field computational cost J.S Smith, O.
    Uslayev and A. E. Roitberg (2017)

    res_file : string, path to .res file
    """

    crys = Crystal.load(res_file)

    return SymmetryFunctions.from_crystal(crys)


def calc_angular_function(
    cm_xyz,
    unique_atoms,
    nearest_atoms,
    cm_atom_xyz,
    cutoff,
    r_s_lst,
    theta_lst,
    eta,
    zeta,
    only_intermolecular=False,
    separate_angular=True,
):
    """
    Angular function (eq (4) of Smith et al. (2017))
    For a given atom returns a list of angular functions of length r_s (and theta - they must be the same length)

    mol_xyz : list of xyz of all atoms in a finite cluster larger than cutoff not in the central molecule
    cm_atom_xyz : xyz of a given atom in the central molecule
    cutoff : float, max L1 distance where atom contributions are included
    r_s : list of hyperparameters controlling position of each gaussian
    theta: list of hyperparameters contrlling the angular position of each gaussian
    eta : float, hyperparameter controlling width of each gaussian
    zeta : float, hyperparameter controlling the width of each gaussian
    """

    front_factor = 2 ** (1 - zeta)

    els, pos = nearest_atoms.neighbours(cm_atom_xyz, cutoff)
    centered = pos - cm_atom_xyz
    distances = np.linalg.norm(centered, axis=1)

    if np.min(distances) > 0.001:
        raise Exception(
            """
        Central Atom is probabaly not at (0, 0, 0). This means that atoms in 
        the original molecule cannot be found in the cluster (ball).
        Check line 37 if "direct" should be replaced by "direct.T"
        """
        )

    center_atom = distances < 1e-3

    dims = distances.shape[0]

    fc = 0.5 * np.cos(distances * np.pi / cutoff) + 0.5
    fc[center_atom] = 0
    distances[center_atom] = 1

    if only_intermolecular:
        for cm_atom_xyz in cm_xyz:
            fc[(np.linalg.norm(pos - cm_atom_xyz, axis=1) < 1e-3)] = 0

    normed = centered[:] / distances[:, np.newaxis]
    angles = np.arccos(np.clip(np.matmul(normed, normed.T), -1, 1))
    angle_diff = (1 + np.cos(angles[:, :, np.newaxis] - np.array(theta_lst))) ** zeta
    r1r2 = 0.5 * (distances[:, np.newaxis] + distances)
    r1r2 = np.exp(-eta * (r1r2[:, :, np.newaxis] - r_s_lst) ** 2)

    fjfk = np.outer(fc, fc)

    # Diagonal terms correspond to the angle of an atom with itself.
    # Pushed them to zero by the following line

    fjfk = np.triu(fjfk, k=1)

    pairs_in_cluster = np.outer(els, els)
    #unique_atoms = np.unique(els)
    atoms_pairs = np.triu(np.outer(unique_atoms, unique_atoms))

    if separate_angular == True:
        G2_list = []
        for pair in atoms_pairs[atoms_pairs != 0]:
            fjfk_temp = np.copy(fjfk)
            fjfk_temp[pairs_in_cluster != pair] = 0
            G2 = front_factor * np.einsum("ij,ijk,ijl->kl", fjfk_temp, r1r2, angle_diff)
            G2_list.append(G2)

        return np.ravel(G2_list)

    G2 = front_factor * np.einsum("ij,ijk,ijl->kl", fjfk, r1r2, angle_diff)

    return np.ravel(G2)


def read_db(db_name):
    """Read crystals from database file

    Args:
        db_name(string): Name of the threshold clustered database file

    Returns:
        ids(list): list of ids.
        jobs(list): list of file_contents.
    """
    ds = CspDataStore(db_name)

    ids = []
    jobs = []
    max_min_step = ds.query(
        'select max(minimization_step) from trial_structure'
    ).fetchone()[0]
    for id, file_content in ds.query(
        "select id, file_content from crystal "
        "join trial_structure using(id) where minimization_step={} "
        #"limit 5"
        .format(max_min_step)
    ).fetchall():
        LOG.debug("Reading id %s", id)
        ids.append(id)
        jobs.append(file_content)

    return ids, jobs


def read_thre_db(db_name):
    """
    Read crystals from database file

    Args:
        db_name(string): Name of the threshold clustered database file

    Returns:
        ids(list): list of ids.
        jobs(list): list of file_contents.
    """
    ds = CspDataStore(db_name)

    ids = []
    jobs = []
    for id, file_content in ds.query(
        "select id, file_content from thre_crystal"
    ).fetchall():
        LOG.debug("Reading id %s", id)
        ids.append(id)
        jobs.append(file_content)

    ds.disconnect()

    return ids, jobs


def read_descriptors(db_name, des_type="symf"):
    """
    Read crystals from database file

    Args:
        db_name(string): Name of the database file with calcualted soap
        descriptors

    Returns:
        ids(list): list of ids.
        descriptors(list): list of soaps.
    """
    ds = CspDataStore(db_name)

    ids = []
    descriptors = []
    for id, radial, angular in ds.query(
        f"select id, value, metadata from descriptor where name='{des_type}' "
        #"limit 5"
    ).fetchall():
        ids.append(id)
        descriptors.append(
            {
                "radial": convert_array(radial),
                "angular": convert_array(angular),
            }
        )

    ds.disconnect()

    return ids, descriptors


def worker(file_content):
    c = Crystal.from_shelx_string(file_content)
    symf = c.calculate_symmetry_functions()
    return {"radial": symf.radial, "angular": symf.angular}

def calculate_descriptors(jobs, nprocs, **kwargs):
    with Pool(processes=nprocs) as pool:
        chunksize = min(10000, int(len(jobs) / nprocs / nprocs) + 1)
        results = pool.map(worker, jobs, chunksize=chunksize)

    return results


def precondition(descriptors, method):
    if method == "None":
        return descriptors

    elif method == "average":
        for i, descriptor in enumerate(descriptors):
            ave_radial = np.average(descriptor["radial"], axis=0)
            ave_angular = np.average(descriptor["angular"], axis=0)
            descriptor["radial"] -= ave_radial
            descriptor["angular"] -= ave_angular
            descriptors[i] = descriptor

    elif method == "normalise":
        for i, descriptor in enumerate(descriptors):
            for part, atoms in descriptor.items():
                for j, atom in enumerate(atoms):
                    max_atom = max(atom)
                    min_atom = min(atom)
                    atom = (atom - min_atom) / (max_atom - min_atom)
                    descriptors[i][part][j] = atom
    else:
        LOG.warning("No such precondition method: %s, use original descriptors", method)

    return descriptors


def euclidean_distance(descriptor_i, descriptor_j):
    a = np.concatenate((descriptor_i["radial"], descriptor_i["angular"]), axis=None)
    b = np.concatenate((descriptor_j["radial"], descriptor_j["angular"]), axis=None)
    return np.linalg.norm(a - b, ord=2)


def infinity_euclidean_distance(descriptor_i, descriptor_j):
    a = np.concatenate((descriptor_i["radial"], descriptor_i["angular"]), axis=None)
    b = np.concatenate((descriptor_j["radial"], descriptor_j["angular"]), axis=None)
    return np.linalg.norm(a - b, ord=np.inf)


def max_atom_distance(descriptor_i, descriptor_j):
    dis_radial = np.linalg.norm(
        descriptor_i["radial"] - descriptor_j["radial"], axis=1,
    )
    dis_angular = np.linalg.norm(
        descriptor_i["angular"] - descriptor_j["angular"], axis=1,
    )
    distance = np.sqrt(dis_radial * dis_radial + dis_angular * dis_angular)
    return max(distance)


def get_j(bulk):
    return int((1 + np.sqrt(1 + bulk * 8)) / 2)


def chunk_distance(index, chunksize, max_j, method):
    """
    Computes number of chunksize distances. This function is for
    parallelization.

    Args:
        index(int): The indexth chunk to be calculated.
        chunksize(int): The size of the chunk.
        max_j(int): Maximum that j could reach, equals the number of
        descriptors, i.e., the number of structures.

    Returns:
        output_list(list): List that contain index i, j, and distance between i
        and j i(int): Index of first structure.
    """
    descriptors = get_global_descriptors()
    bulk = index * chunksize
    start_j = get_j(bulk)
    start_i = bulk - int(start_j * (start_j - 1) / 2)
    end_j = min(max_j, get_j(bulk + chunksize) + 1)
    if method == "euclidean":
        pair_distance = euclidean_distance
    elif method == "infinity_euclidean":
        pair_distance = infinity_euclidean_distance
    elif method == "max_atom":
        pair_distance = max_atom_distance

    output_list = []
    num = 0
    j = start_j
    for i in range(start_i, start_j):
        output_list.append(
            [
                i, j, 
                pair_distance(descriptors[i], descriptors[j])
            ]
        )
        num += 1
        if num >= chunksize:
            return output_list

    for j in range(start_j + 1, end_j):
        for i in range(j):
            output_list.append(
                [
                    i, j, 
                    pair_distance(descriptors[i], descriptors[j])
                ]
            )
            num += 1
            if num >= chunksize:
                return output_list

    return output_list


def calculate_distance(descriptors, nprocs, method="euclidean"):
    from functools import partial
    length = len(descriptors)
    set_global_descriptors(descriptors)

    time1 = time.time()
    input_length = int(length * (length - 1) / 2)
    chunksize = min(10000, int(input_length / nprocs / nprocs) + 1)
    if input_length % chunksize == 0:
        chunknum = int(input_length / chunksize)
    else:
        chunknum = int(input_length / chunksize) + 1
    LOG.info(
        "Input length %s chunksize %s chunknum %s", 
        input_length, chunksize, chunknum
    )

    with Pool(processes=nprocs) as pool:
        f = partial(
            chunk_distance, 
            chunksize=chunksize,
            max_j=length,
            method=method,
        )
        results = pool.map(f, [i for i in range(chunknum)], chunksize=chunksize)
    
    LOG.info("Pool joined by %s seconds", time.time()-time1)

    D = np.zeros((length, length))

    for result in results:
        for i, j, d_ij in result: #.get():
            D[i, j] = d_ij
            D[j, i] = d_ij
            if j == i:
                LOG.error("Diagonal ones %s should not be calculated again!", i)

    for i in range(length):
        D[i, i] = 0.0

    return D


def output(ids, descriptors, db_name, des_type="symf"):
    """
    Update descriptors to input database
    """
    ds = CspDataStore(db_name)
    ds.query(f"delete from descriptor where name='{des_type}'")

    descriptors_tmp = defaultdict(list)
    for i, id in enumerate(ids):
        descriptors_tmp["id"].append(id)
        descriptors_tmp["value"].append(adapt_array(descriptors[i]["radial"]))
        descriptors_tmp["name"].append(des_type)
        descriptors_tmp["metadata"].append(adapt_array(descriptors[i]["angular"]))

    ds.insert_many("descriptor", descriptors_tmp)
    ds.add_metadata("added {} descriptors".format(len(descriptors)))
    ds.commit()

    ds.disconnect()


def output_seperate(ids, descriptors, db_name):
    """
    Dump descriptors to an individual database
    """
    db_conn = sqlite3.connect(db_name)
    db_cursor = db_conn.cursor()
    db_cursor.execute("CREATE TABLE descriptor (id PRIMARY KEY, name, value, metadata)")

    for i, id in enumerate(ids):
        db_cursor.execute(
            "INSERT INTO descriptor VALUES (?, ?, ?, ?)", 
            (
                id, 
                "symf", 
                adapt_array(descriptors[i]["radial"]), 
                adapt_array(descriptors[i]["angular"]),
            )
        )

    db_conn.commit()
    db_conn.close()


def main():
    parser = argparse.ArgumentParser(
        description="Calculate symmetry functions and distance matrix from database",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "input_db", 
        type=str,
        help="Input clustered database file"
    )
    parser.add_argument(
        "-t", 
        "--threshold-database", 
        action="store_true",
        default=False,
        help="Read jobs from clustered threshold database containing "
        "thre_crystal column"
    )
    parser.add_argument(
        "-e", 
        "--exist-descriptors", 
        action="store_true",
        default=False,
        help="Descriptors already existed, directly calculate distance matrix"
    )
    parser.add_argument(
        "--distance", 
        type=str, 
        choices=["euclidean", "infinity_euclidean", "max_atom"],
        default="euclidean",
        help="Type of distance"
    )
    parser.add_argument(
        "--precondition", 
        type=str, 
        choices=["None", "average", "normalise"],
        default="None",
        help="Type of preconditioning of symmetry function"
    )
    parser.add_argument(
        "-dm",
        "--distance-matrix", 
        action="store_true",
        default=False,
        help="Calculate and dump distance matrix"
    )
    parser.add_argument(
        "--seperate-database", 
        action="store_true",
        default=False,
        help="Store descriptors not in input database but in a seperate one"
    )
    parser.add_argument(
        "-np", 
        "--number-process", 
        type=int, 
        default=1,
        help="Number of processes to parrellel"
    )
    parser.add_argument(
        "-o", 
        "--output", 
        type=str, 
        default=None,
        help="Name for the output file"
    )
    parser.add_argument(
        "-ll",
        "--log_level", 
        default="INFO",
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help="Set the logging level"
    )

    args = parser.parse_args()
    
    if args.output is None:
        out_name = args.input_db.split('.')[0]
    else:
        out_name = args.output
    
    logging.basicConfig(level=args.log_level) 
    LOG.info("Reading database %s", args.input_db)
    #Read crystals from database file

    if args.exist_descriptors:
        args.distance_matrix = True
        ids, descriptors = read_descriptors(args.input_db)
    else:
        if args.threshold_database:
            args.seperate_database = True
            ids, jobs = read_thre_db(args.input_db)
        else:
            ids, jobs = read_db(args.input_db)

        LOG.info("Start calculating descriptores for %s structures", len(jobs))
        time_1 = time.time()
        descriptors = calculate_descriptors(
            jobs, 
            #args.symf_part, 
            args.number_process,
        )

        time_run = time.time() - time_1
        LOG.info("Analyze time %.3f writing to input database", time_run)
        if args.seperate_database:
            output_seperate(ids, descriptors, f"{out_name}_symf.db")
        else:
            output(ids, descriptors, args.input_db)

    if args.distance_matrix:
        descriptors = precondition(descriptors, args.precondition)
        LOG.info("Distance matrix calculating for %s descriptors", len(descriptors))
        time_1 = time.time()
        distance = calculate_distance(
            descriptors, 
            args.number_process,
            args.distance,
        )
        LOG.info(
            "Distance matrix calculated by %s seconds, dumping to %s",
            time.time() - time_1, f"{out_name}_{args.distance}.npy"
        )
        np.save(f"{out_name}_{args.distance}.npy", distance)

    return

if __name__ == "__main__":
    main()
