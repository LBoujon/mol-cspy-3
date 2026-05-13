#!/usr/bin/env python
import argparse
import io
import sys
import logging
import sqlite3
import time
import numpy as np
import json
import hdbscan
from multiprocessing import Pool
from collections import namedtuple, defaultdict
from cspy.db import CspDataStore
from cspy.crystal import Crystal
from cspy.db.key import CspDatabaseId
from dscribe.descriptors import SOAP
from ase import Atoms

LOG = logging.getLogger("CSPy")
np.set_printoptions(threshold=np.inf)

species = [0]

params = {
    "gamma": 2.0,
    "threshold": 1e-6
}

global_var_dict = {}

def get_global_soap_calculator():
    return global_var_dict['soaper']

def get_global_soaps():
    return global_var_dict['soaps']

def get_global_diagonal():
    return global_var_dict['diagonal']

def set_global_soap_calculator(soaper):
    global_var_dict['soaper'] = soaper

def set_global_soaps(soaps):
    global_var_dict['soaps'] = soaps

def set_global_diagonal(diagonal):
    global_var_dict['diagonal'] = diagonal

def get_j(bulk):
    return int((1 + np.sqrt(1 + bulk * 8)) / 2)

def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out, allow_pickle=True)

def adapt_array(arr):
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())

def rematch_global_similarity(localkernel, gamma, threshold):
    """
    Computes the REMatch similarity between two structures A and B.

    Args:
        localkernel(np.ndarray): NxM matrix of local similarities between
            structures A and B, with N and M atoms respectively.
        gamma(float): Parameter for rematch kernel weighting entropy
        threshold(float): Threshold for convergence

    Returns:
        glosim(float): REMatch similarity between the structures A and B.
    """
    n, m = localkernel.shape
    K = np.exp(-(1 - localkernel) / gamma)

    # initialisation
    u = np.ones((n,)) / n
    v = np.ones((m,)) / m

    en = np.ones((n,)) / float(n)
    em = np.ones((m,)) / float(m)

    # converge balancing vectors u and v
    itercount = 0
    error = 1
    while (error > threshold):
        uprev = u
        vprev = v
        v = np.divide(em, np.dot(K.T, u))
        u = np.divide(en, np.dot(K, v))

        # determine error every now and then
        if itercount % 5:
            error = np.sum((u - uprev) ** 2) / np.sum((u) ** 2) + np.sum((v - vprev) ** 2) / np.sum((v) ** 2)
        itercount += 1

    # using Tr(X.T Y) = Sum[ij](Xij * Yij)
    # P.T * C
    # P_ij = u_i * v_j * K_ij
    pity = np.multiply( np.multiply(K, u.reshape((-1, 1))), v)

    glosim = np.sum( np.multiply( pity, localkernel))

    return glosim

def diagonal_pair_similarity(i):
    """
    Computes the diagonal components of REMatch similarity matrix.

    Args:
        i(int): Index of diagonal component to be calculated.

    Returns:
        (float): The ith diagonal component of similarity matrix.
    """
    x = get_global_soaps()[i]
    local_kernel = np.dot(x, x.T)
    gamma = params["gamma"]
    threshold = params["threshold"]
    global_kernel = rematch_global_similarity(local_kernel, gamma, threshold)
    return np.sqrt(global_kernel)

#def rematch_pair_similarity(input_data):
def rematch_pair_similarity(i, j, x, y, xx, yy):
    """
    Computes linear kernel and REMatch kernel with diagonal components to
    normalize similarity.

    Args:
        i(int): Index of first structure.
        j(int): Index of second structure.
        x(np.ndarray): Soap descriptor of first structure.
        y(np.ndarray): Soap descriptor of second structure.
        xx(float): The ith diagonal component as normalization factor.
        yy(float): The jth diagonal component as normalization factor.

    Returns:
        i(int): Index of first structure.
        j(int): Index of second structure.
        global_kernel(float): REMatch similarity between the structures i and j.
    """
    #i, j, x, y, xx, yy = input_data

    time1 = time.time()
    local_kernel = np.dot(x, y.T)
    #normalise = xx * yy
    #local_kernel /= normalise
    #LOG.info("Get normalized local kernel by %s seconds", time.time()-time1)

    time1 = time.time()
    gamma = params["gamma"]
    threshold = params["threshold"]
    global_kernel = rematch_global_similarity(local_kernel, gamma, threshold)
    #LOG.info("Get global kernel by %s seconds", time.time()-time1)
    normalise = xx * yy
    global_kernel /= normalise
    if global_kernel > 1.0:
        LOG.debug(
            "Global_kernel bigger than 1.0 for pair (%s, %s) with %s",
            i, j, global_kernel
        )
        global_kernel = 1.0

    return [i, j, global_kernel]

def chunk_rematch_pair_similarity(
    index, chunksize, max_j
):
    """
    Computes number of chunksize REMatch similarity. This function is for
    parallelization.

    Args:
        index(int): The indexth chunk to be calculated.
        chunksize(int): The size of the chunk.
        max_j(int): Maximum that j could reach, equals the number of SOAPs,
        i.e., the number of structures.

    Returns:
        output_list(list): List that contain index i, j, and rematch similarity
        between i and j i(int): Index of first structure.
    """
    soaps = get_global_soaps()
    diagonal = get_global_diagonal()
    bulk = index * chunksize
    start_j = get_j(bulk)
    start_i = bulk - int(start_j * (start_j - 1) / 2)
    end_j = min(max_j, get_j(bulk + chunksize) + 1)

    output_list = []
    num = 0
    j = start_j
    for i in range(start_i, start_j):
        output_list.append(
            rematch_pair_similarity(
                i, j,
                soaps[i],
                soaps[j],
                diagonal[i],
                diagonal[j],
            )
        )
        num += 1
        if num >= chunksize:
            return output_list

    for j in range(start_j + 1, end_j):
        for i in range(j):
            output_list.append(
                rematch_pair_similarity(
                    i, j,
                    soaps[i],
                    soaps[j],
                    diagonal[i],
                    diagonal[j],
                )
            )
            num += 1
            if num >= chunksize:
                return output_list

    return output_list

def read_db(db_name):
    """
    Read crystals from database file

    Args:
        db_name(string): Name of the database file

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
        #"limit 50"
        .format(max_min_step)
        #"join equivalent_to on crystal.id = equivalent_to.unique_id "
    ).fetchall():
        LOG.debug("Reading id %s", id)
        ids.append(id)
        jobs.append(file_content)

    ds.disconnect()

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

def read_descriptors(db_name):
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
    for id, descriptor in ds.query(
        "select id, value from descriptor where name='soap'"
    ).fetchall():
        ids.append(id)
        descriptors.append(convert_array(descriptor))

    ds.disconnect()

    return ids, descriptors

def calculate_single_soap(file_content):
    """
    Calculate soap descriptor of a crystal structure.

    Args:
        file_content(string): Res file string of crystal structure to be
        calculated.

    Returns:
        (np.ndarray): SOAP descriptor of the crystal structure.
    """
    soaper = get_global_soap_calculator()
    c = Crystal.from_shelx_string(file_content)
    slab = c.unit_cell_atoms()
    atom = Atoms(
        positions=slab['cart_pos'], cell=c.unit_cell.lattice, pbc=[1,1,1]
    )
    return soaper.create(atom)

def calculate_soaps(
    jobs, 
    cutoff, 
    num_radial, 
    num_spherical, 
    sigma, 
    nprocs,
    periodic=True,
):
    """
    Loop to calculate all the soap descriptors of crystal structures.

    Args:
        jobs(list): List of res strings.
        cutoff(float): Distance cutoff defining neighbour atoms.
        num_radial(int): Number of radial basis functions expanding atomic
        neighbour density.
        num_spherical(int): Number of spherical harmonics expanding atomic
        neighbour density.
        sigma(float): Width of gaussian function presenting atomic denstiy.
        nprocs(int): Number of processes to parallel.
        periodic(boolean): Whether to treat the structure as periodic boundary.

    Returns:
        (np.ndarray): SOAP descriptor of the crystal structure.
    """
    soaper = SOAP(
        cutoff, 
        num_radial, 
        num_spherical, 
        species=species, 
        sigma=sigma, 
        periodic=periodic,
    )

    time1 = time.time()
    chunksize = min(10000, int(len(jobs) / nprocs / nprocs))
    pool = Pool(
        processes=nprocs,
        initializer=set_global_soap_calculator,
        initargs=(soaper,),
    )
    soaps = pool.map(calculate_single_soap, jobs, chunksize=chunksize)
    pool.close()
    pool.join()
    del pool
    LOG.info(
        "Calculated %s soaps by %s seconds", 
        len(soaps), time.time()-time1
    )

    return soaps

def calculate_rematch_similarity(soaps, gamma, nprocs):
    """
    Loop to calculate distance matrix from REMatch similarity matrix.

    Args:
        soaps(list): List of SOAP descriptors.
        gamma(float): Parameter for rematch kernel weighting entropy
        nprocs(int): Number of processes to parallel.

    Returns:
        distance(np.ndarray): Distance matrix from REMatch similarity.
    """
    length = len(soaps)

    LOG.info(
        "Rematch kernel calculating similarity matrix for %s soaps "
        "with each %s dimension by %s processes", 
        length, 
        soaps[0].shape, 
        nprocs,
    )

    params_tmp = {
        "gamma": gamma,
    }

    for para in params_tmp:
        if para in params:
            params[para] = params_tmp[para]

    time1 = time.time()
    chunksize = min(10000, int(length / nprocs / nprocs))
    indices = [x for x in range(length)]
    set_global_soaps(soaps)
    pool = Pool(processes=nprocs)
    diagonal = pool.map(diagonal_pair_similarity, indices)
    pool.close()
    pool.join()
    del pool
    LOG.info(
        "Get diagonal components for normalization by %s seconds", 
        time.time()-time1
    )

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
    results = []
    pool = Pool(
        processes=nprocs,
        initializer=set_global_diagonal,
        initargs=(diagonal,),
    )
    for i in range(chunknum):
        #LOG.info("Submit tasks index %s", i)
        results.append(
            pool.apply_async(
                func=chunk_rematch_pair_similarity,
                args=(i, chunksize, length,),
            )
        )
    pool.close()
    pool.join()
    del pool
    LOG.info("Pool joined by %s seconds", time.time()-time1)
    
    K_ij = np.zeros((length, length))

    for result in results:
        for i, j, k_ij in result.get():
            K_ij[i, j] = k_ij
            K_ij[j, i] = k_ij
            if j == i:
                LOG.error("Diagonal ones %s should not be calculated again!", i)

    for i in range(length):
        K_ij[i, i] = 1.0
        
    # Enforce kernel normalization.
    #time1 = time.time()
    #k_ii = np.diagonal(K_ij)
    #x_k_ii_sqrt = np.sqrt(k_ii)
    #y_k_ii_sqrt = x_k_ii_sqrt
    #K_ij /= np.outer(x_k_ii_sqrt, y_k_ii_sqrt)
    #LOG.info("Kernel normaliziation by %s seconds", time.time()-time1)

    #LOG.info("Similarity matrix calculated for %s seconds", time.time()-time1)
    distance = np.sqrt( 2 - 2 * K_ij)
    return distance

def output(ids, soaps, db_name):
    """
    Update SOAP descriptor to input database
    """
    ds = CspDataStore(db_name)
    ds.query("delete from descriptor where name='soap'")

    descriptors = defaultdict(list)
    for i, id in enumerate(ids):
        descriptors["id"].append(id)
        descriptors["value"].append(adapt_array(soaps[i]))
        descriptors["name"].append("soap")
        descriptors["metadata"].append("")

    ds.insert_many("descriptor", descriptors)
    ds.add_metadata("added {} descriptors".format(len(soaps)))
    ds.commit()

    ds.disconnect()

def output_seperate(ids, soaps, db_name):
    """
    Dump SOAP descriptor to an individual database
    """
    db_conn = sqlite3.connect(db_name)
    db_cursor = db_conn.cursor()
    db_cursor.execute("CREATE TABLE descriptor (id PRIMARY KEY, name, value)")

    for i, id in enumerate(ids):
        db_cursor.execute(
            "INSERT INTO descriptor VALUES (?, ?, ?)", 
            (id, "soap", adapt_array(soaps[i]))
        )

    db_conn.commit()
    db_conn.close()


def main():
    parser = argparse.ArgumentParser(
        description="Calculate soap descriptor and distance matrix from database",
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
        help="Read jobs from clustered threshold database containing thre_crystal column"
    )
    parser.add_argument(
        "-e", 
        "--exist-descriptors", 
        action="store_true",
        default=False,
        help="Descriptors already existed, directly calculate distance matrix"
    )
    parser.add_argument(
        "-c", 
        "--cutoff", 
        type=float, 
        default=3.0,
        help="Cut-off of atomic distances"
    )
    parser.add_argument(
        "-g", 
        "--gamma", 
        type=float, 
        default=2.0,
        help="Gamma parameter for match kernel"
    )
    parser.add_argument(
        "-s", 
        "--sigma", 
        type=float, 
        default=0.3,
        help="Width of Gaussian functions in SOAP"
    )
    parser.add_argument(
        "-np", 
        "--number-process", 
        type=int, 
        default=1,
        help="Number of processes to parrellel"
    )
    parser.add_argument(
        "-n", 
        "--radial-num", 
        type=int, 
        default=9,
        help="Number of radial functions"
    )
    parser.add_argument(
        "-l", 
        "--spherical-num", 
        type=int, 
        default=9,
        help="Number of spherical harmonics"
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
        "-o", 
        "--output", 
        type=str, 
        default=None,
        help="Name for the output file"
    )
    parser.add_argument(
        "-ll",
        "--log-level", 
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
        ids, soaps = read_descriptors(args.input_db)
    else:

        if args.threshold_database:
            args.seperate_database = True
            ids, jobs = read_thre_db(args.input_db)
        else:
            ids, jobs = read_db(args.input_db)

        time_1 = time.time()
        soaps = calculate_soaps(
            jobs, 
            args.cutoff, 
            args.radial_num, 
            args.spherical_num, 
            args.sigma,
            args.number_process,
        )

        time_run = time.time() - time_1
        LOG.info("Analyze time %.3f writing soaps to input database", time_run)
        if args.seperate_database:
            output_seperate(ids, soaps, f"{out_name}_soap.db")
        else:
            output(ids, soaps, args.input_db)
   
    if args.distance_matrix:
        LOG.info("Distance matrix calculating for %s soaps", len(soaps))
        time_1 = time.time()
        distance = calculate_rematch_similarity(
            soaps,
            args.gamma,
            args.number_process,
        )
        LOG.info(
            "Distance matrix calculated by %s seconds, dumping to %s",
            time.time() - time_1, f"{out_name}_distance.npy"
        )
        np.save(f"{out_name}_distance.npy", distance)

    return

if __name__ == "__main__":
    main()
