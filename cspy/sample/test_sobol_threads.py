from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
from collections import defaultdict
import numpy as np
import sys
import logging
from cspy.sample.sobol import sobol_vector

LOG = logging.getLogger(__name__)


def random_vector(seed, dims):
    vector = sobol_vector(seed + 1, dims)
    return seed, dims, vector


def test_sobol_multithread():
    nvals = 100
    dims = list(range(2, 5))
    correct = {}
    for d in dims:
        correct[d] = {i: random_vector(i, d)[2] for i in range(1, nvals + 1)}
    results = {d: {} for d in dims}

    for nthreads in range(2, 5):
        with ThreadPoolExecutor(nthreads) as e:
            futures = []
            for i in range(1, nvals + 1):
                for d in dims:
                    futures.append(e.submit(random_vector, i, d))
            for f in as_completed(futures):
                i, d, v = f.result()
                results[d][i] = v
        LOG.info("Threads = %d", nthreads)
        for d in dims:
            for i in range(2, nvals + 1):
                np.testing.assert_allclose(correct[d][i], results[d][i])
        LOG.info("Threads = %d correct", nthreads)
