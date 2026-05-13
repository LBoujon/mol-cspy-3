#!/usr/bin/env python
import argparse
import sys
import logging
import sqlite3
import time
import numpy as np
import json
from sklearn.cluster import AffinityPropagation
from multiprocessing import Pool
from collections import namedtuple, defaultdict
from cspy.db import CspDataStore

LOG = logging.getLogger("CSPy")
np.set_printoptions(threshold=np.inf)

def read_db(db_name):
    """Read crystals from database file"""
    ds = CspDataStore(db_name)

    ids = []
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
        ids.append([id, density, energy])

    return ids

def read_thre_db(db_name):
    """Read crystals from database file"""
    ds = CspDataStore(db_name)

    ids = []
    for id, density, energy in ds.query(
        "select id, min_density, min_energy from thre_crystal "
        #"join equivalent_to on crystal.id = equivalent_to.unique_id "
    ).fetchall():
        LOG.debug("Reading id %s", id)
        ids.append([id, density, energy])

    return ids

def distance_to_similarity(distance):
    return (2 - distance * distance) / 2

def clustering(distance, damping, max_iter, convergence_iter):
    LOG.info("Affinity Propagation started for matrix %s", distance.shape)
    time1 = time.time()

    clusterer = AffinityPropagation(
        affinity="precomputed", 
        damping=damping,
        max_iter=max_iter,
        convergence_iter=convergence_iter,
        random_state=0,
    )

    clusterer.fit(distance)
    LOG.info("Cluster completed %s seconds", time.time()-time1)
    return clusterer.labels_

def output(ids, labels, out_name):
    out_file = open(f"{out_name}.ap", 'w')
    out_file.write(
        "{:<24} {:<12} {:<12} {:<12}\n".format("id", "density", "energy", "cluster")
    )
    for i, value in enumerate(ids):
        id, density, energy = value
        out_file.write(
            "{:<24} {:<12.6f} {:<12.6f} {:<12}\n".format(id, density, energy, labels[i])
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
        "-d",
        "--damping", 
        type=float, 
        default=0.5,
        help="Damping factor"
    )
    parser.add_argument(
        "-mi",
        "--max-iter", 
        type=int, 
        default=200,
        help="Maximum number of iterations"
    )
    parser.add_argument(
        "-ci", 
        "--convergence-iter", 
        type=int, 
        default=15,
        help="Number of no changed iterations considered as convergence"
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

    if args.threshold:
        ids = read_thre_db(args.input_db)
    else:
        ids = read_db(args.input_db)

    if args.distance_matrix is not None:
        LOG.info("Reading distance matrix from %s", args.distance_matrix)
        distance = np.load(args.distance_matrix)
        similarity = distance_to_similarity(distance)
        del distance
    elif args.similarity_matrix is not None:
        LOG.info("Reading similarity matrix from %s", args.similarity_matrix)
        similarity = np.load(args.similarity_matrix)
    else:
        LOG.error("No similarity or distance matrix")
        exit()

    labels = clustering(
        similarity, 
        args.damping,
        args.max_iter,
        args.convergence_iter,
    )

    time_run = time.time() - time_1
    LOG.info("Analyze time %.3f writing to output name %s", time_run, out_name)
    output(ids, labels, out_name)

    return

if __name__ == "__main__":
    main()
