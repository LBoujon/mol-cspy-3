#!/usr/bin/env python
import argparse
import logging
import sqlite3
import time
import numpy as np
import json
import scipy
from collections import namedtuple, defaultdict
from cspy.db import CspDataStore
from cspy.crystal import Crystal
from cspy.db.key import CspDatabaseId
from sklearn.decomposition import PCA
from cspy.ml.descriptors.soap import convert_array, adapt_array

LOG = logging.getLogger("CSPy")
np.set_printoptions(threshold=np.inf)

def read_db(db_name, destype):
    """
    Read crystals from database file

    Args:
        db_name: string, the name of clustered database
        destype: string, type of descriptor to read

    Return:
        ids: list of id of structures
        a numpy array containing the descriptor as the same order of ids
    """
    ds = CspDataStore(db_name)

    ids = []
    matrix = []
    max_min_step = ds.query(
        'select max(minimization_step) from trial_structure'
    ).fetchone()[0]

    for id, descriptor, meta in ds.query(
        "select id, value, descriptor.metadata from descriptor join trial_structure using(id) "
        "where minimization_step={} and name='{}'"
        .format(max_min_step, destype)
    ).fetchall():
        LOG.debug("Reading id %s", id)
        ids.append(id)
        if destype == "symf":
            #if type of descriptor is symmetry function, the radial and angular
            #part will be combined to one vector
            vector = np.concatenate(
                (convert_array(descriptor).flatten(), convert_array(meta).flatten()), 
                axis=None,
            )
        else:
            vector = convert_array(descriptor).flatten()
        #vector = vector / np.linalg.norm(vector)
        matrix.append(vector)

    return ids, np.array(matrix)

def principal_component_analysis(matrix, pca_num):
    """
    Use single value decomposition to calculate principal axes and mapped to
    principal components
    
    Args:
        matrix: numpy array, containing the descriptors
        pca_num: int, the number of principal components to be calcualted and
        dumped

    Return:
        matrix_pca: numpy array, calculated principal components, N*pca_num
        matrix, N is the number of structures
    """
    #To use svd to calculate pca, the descriptors need to be centered at 0
    matrix -= np.mean(matrix, axis=0)

    time1 = time.time()
    u, s, vh = np.linalg.svd(matrix, full_matrices=False)
    LOG.info("SVD for %s seconds", time.time()-time1)
    #pca = PCA(n_components=min(len(matrix), len(matrix[0])))
    #matrix_pca = pca.fit_transform(matrix)
    #LOG.info("%i +10 biggest eigenvalues: %s", pca_num, pca.explained_variance_[:pca_num+10])
    eigen = s ** 2 / (len(matrix) - 1)
    LOG.info("%i +10 biggest eigenvalues: %s", pca_num, s[:pca_num+10])
    time1 = time.time()
    matrix_pca = np.dot(matrix, vh[:pca_num].T)
    LOG.info("Multiply for %s seconds", time.time()-time1)
    return matrix_pca

def output(ids, matrix, pca_num, descriptor, out_name):
    out_file = open(f"{out_name}-{descriptor}.pca", 'w')
    format_str =  ' '.join("{:<12.6f}" for x in range(pca_num))
    format_str = "{:<36} " + format_str + '\n'
    out_file.write("{:<36} {:<24}\n".format("id", "components"))
    for i, id in enumerate(ids):
        out_file.write(
            format_str.format(
                id, *(matrix[i][:pca_num])
            )
        )
    out_file.close()

def main():
    parser = argparse.ArgumentParser(
        description="Dimension reduction by Principle Component Analysis (PCA)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input_db", 
        type=str,
        help="Input clustered database file"
    )
    parser.add_argument(
        "-t", 
        "--descriptor", 
        type=str, 
        choices=["symf", "soap", "sht"],
        help="Type of descriptor used for PCA"
    )
    parser.add_argument(
        "-p", 
        "--pca_type", 
        type=str, 
        choices=["exact", "incremental", "kernel"],
        default="exact",
        help="Type of PCA method to use"
    )
    parser.add_argument(
        "-k", 
        "--kernel", 
        type=str, 
        default="linear",
        choices=["linear", "poly", "rbf", "sigmoid", "cosine"],
        help="Type of kernel for kernel PCA"
    )
    parser.add_argument(
        "-d", 
        "--dimension", 
        type=int, 
        default=2,
        help="Number of dimensions to dump"
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
    
    logging.basicConfig(level=args.log_level) 
    LOG.info("Reading database %s", args.input_db)
    #Read crystals from database file
    time_1 = time.time()

    ids, matrix = read_db(args.input_db, args.descriptor)
    LOG.info(
        "Calculating pca for %s %s descriptors in %s dimensions", 
        len(ids), args.descriptor, len(matrix[0])
    )
    matrix_pca = principal_component_analysis(matrix, args.dimension)

    time_run = time.time() - time_1
    if args.output is None:
        out_name = args.input_db.split('.')[0]
    else:
        out_name = args.output
    LOG.info("Analyze time %.3f writing to output name %s", time_run, out_name)
    output(ids, matrix_pca, args.dimension, args.descriptor, out_name)

    return

if __name__ == "__main__":
    main()
