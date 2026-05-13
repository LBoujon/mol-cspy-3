#!/usr/bin/env python
import argparse
import sys
import logging
import sqlite3
import time
import numpy as np
import json
#import hdbscan
from hdbscan import HDBSCAN
from multiprocessing import Pool
from collections import namedtuple, defaultdict
from cspy.db import CspDataStore
from cspy.crystal import Crystal

LOG = logging.getLogger("CSPy")
np.set_printoptions(threshold=np.inf)

def similarity_to_distance(S):
    x = len(S)
    D = np.zeros((x, x))
    
    for i in range(x):
        D[i, i] = 0
        for j in range(i):
            d_ij = np.sqrt(max(0, S[i, i] + S[j, j] - 2 * S[i, j]))
            D[i, j] = d_ij
            D[j, i] = d_ij

    return D

def read_db(db_name, ids=[]):
    """Read crystals from database file"""
    ds = CspDataStore(db_name)

    output = []
    if ids == []:
        max_min_step = ds.query(
            'select max(minimization_step) from trial_structure'
        ).fetchone()[0]
        for id, density, energy in ds.query(
            "select id, density, energy from crystal "
            "join trial_structure using(id) where minimization_step={} "
            #"limit 50"
            .format(max_min_step)
            #"join equivalent_to on crystal.id = equivalent_to.unique_id "
        ).fetchall():
            LOG.debug("Reading id %s", id)
            output.append([id, density, energy])
    else:
        for id in ids:
            density, energy = ds.query(
                "select density, energy from crystal where id='{}'"
                .format(id)
            ).fetchone()
            LOG.debug("Reading id %s", id)
            output.append([id, density, energy])
            
    return output

def read_thre_db(db_name, ids=[]):
    """Read crystals from database file"""
    ds = CspDataStore(db_name)

    output = []
    if ids == []:
        for id, density, energy in ds.query(
            "select id, min_density, min_energy from thre_crystal "
            #"join equivalent_to on crystal.id = equivalent_to.unique_id "
        ).fetchall():
            LOG.debug("Reading id %s", id)
            ids.append([id, density, energy])
    else:
        for id in ids:
            density, energy = ds.query(
                "select min_density, min_energy from thre_crystal where id='{}'"
                .format(id)
            ).fetchone()
            LOG.debug("Reading id %s", id)
            output.append([id, density, energy])
            

    return output

def clustering(distance, min_cluster_size, nprocs):
    LOG.info("HDBscan started for matrix %s with %s cores", distance.shape, nprocs)
    time1 = time.time()

    clusterer = HDBSCAN(
        metric="precomputed", 
        min_cluster_size=min_cluster_size,
        nprocs=nprocs,
    )

    clusterer.fit(distance)
    LOG.info("Cluster completed %s seconds", time.time()-time1)
    return clusterer.labels_

def output(ids, labels, out_name):
    out_file = open(f"{out_name}.hdbscan", 'w')
    out_file.write(
        "{:<36} {:<12} {:<12} {:<12}\n".format("id", "density", "energy", "cluster")
    )
    for i, value in enumerate(ids):
        id, density, energy = value
        out_file.write(
            "{:<36} {:<12.6f} {:<12.6f} {:<12}\n".format(id, density, energy, labels[i])
        )
    out_file.close()

def main():
    parser = argparse.ArgumentParser(
        description="Cluster by hdbscan from database and distance matrix",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "input_db", 
        type=str,
        help="Input clustered database file"
    )
    parser.add_argument(
        "-t", 
        "--threshold", 
        action="store_true",
        default=False,
        help="Read input clustered threshold database file"
    )
    parser.add_argument(
        "--id", 
        type=str,
        default=None,
        help="Id file"
    )
    parser.add_argument(
        "-sm",
        "--similarity-matrix", 
        type=str,
        default=None,
        help="Similarity matrix numpy file"
    )
    parser.add_argument(
        "-dm",
        "--distance-matrix", 
        type=str,
        default=None,
        help="Distance matrix numpy file"
    )
    parser.add_argument(
        "-m",
        "--min-cluster-size", 
        type=int, 
        default=5,
        help="Minimum number of cluster size for hdbscan"
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
    time_1 = time.time()

    ids = []
    if args.id is not None:
        with open(args.id, 'r') as f:
            ids = f.read().splitlines()

    if args.threshold:
        ids = read_thre_db(args.input_db, ids=ids)
    else:
        ids = read_db(args.input_db, ids=ids)

    if args.distance_matrix is not None:
        LOG.info("Reading distance matrix from %s", args.distance_matrix)
        distance = np.load(args.distance_matrix)
    elif args.similarity_matrix is not None:
        LOG.info("Reading similarity matrix from %s", args.similarity_matrix)
        similarity = np.load(args.similarity_matrix)
        distance = similarity_to_distance(similarity)
        del similarity
    else:
        LOG.error("No similarity or distance matrix")
        exit()

    labels = clustering(
        distance, 
        args.min_cluster_size,
        args.number_process,
    )

    time_run = time.time() - time_1
    LOG.info("Analyze time %.3f writing to output name %s", time_run, out_name)
    output(ids, labels, out_name)

    return

if __name__ == "__main__":
    main()
