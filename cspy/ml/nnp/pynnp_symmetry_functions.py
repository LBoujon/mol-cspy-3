import logging
import time
from functools import partial
from multiprocessing import Pool
from typing import List, Union

import numpy as np
import pynnp

from cspy.crystal import Crystal
from cspy.ml.nnp.format.parse_n2p2_data import to_n2p2_input_data_str

LOG = logging.getLogger(__name__)


def calculate_descriptors(
    jobs: List[str],
    inputnn_fpath: str,
    scaling_fpath: str = None,
    nprocs: int = 1,
    **kwargs
) -> List[np.ndarray]:
    """Calculate symmetry functions for cspy crystal using n2p2

    Args:
        jobs - list of res contents
        inputnn_fpath (str): path to input.nn defining symmetry functions
        scaling_fpath (str): path to symmetry function scaling file, if None, raw values
        nprocs (int): number of processes to use for parallelization

    Returns:
        List[np.ndarray]: list of symmetry function descriptors for each crystal
    """
    from cspy import Crystal

    crystals = [Crystal.from_shelx_string(c) for c in jobs]

    return calculate_symmetry_functions(
        crystals,
        inputnn_fpath=inputnn_fpath,
        scaling_fpath=scaling_fpath,
        nprocs=nprocs
    )


def calculate_symmetry_functions(
    crystals: List[Crystal],
    inputnn_fpath: str,
    scaling_fpath: str = None,
    nprocs: int = 1,
    **kwargs
) -> List[np.ndarray]:
    """Calculate symmetry functions for cspy crystal using n2p2

    Args:
        crystal List[Crystal]: crystal objects to calculate symmetry functions,
                                without symmetry, i.e. P1
        input_nn_path (string): path to input.nn defining symmetry functions
        scaling_fpath (str): path to symmetry function scaling file, if None, raw values
        nprocs (int): number of processes to use for parallelization

    Returns:
        List[np.ndarray]: list of symmetry function descriptors for each crystal
    """
    with Pool(processes=nprocs) as pool:
        chunksize = min(10000, int(len(crystals) / nprocs / nprocs) + 1)
        f = partial(
            calculate_single_symmetry_functions,
            inputnn_fpath=inputnn_fpath,
            scaling_fpath=scaling_fpath
        )
        results = pool.map(
            f,
            crystals,
            chunksize=chunksize,
        )

    return results


def calculate_single_symmetry_functions(
    crystal: Crystal,
    inputnn_fpath: str,
    scaling_fpath: str = None
) -> np.ndarray:
    """calculate symmetry functions using pynnp for single cspy crystal structure

    Args:
        crystal (Crystal): cspy crystal object to calculate symmetry functions,
                           without symmetry, i.e. P1
        inputnn_fpath (str): path to input.nn defining symmetry functions
        scaling_fpath (str): path to symmetry function scaling file, if None, raw values

    Returns:
        np.ndarray: symmetry function descriptors for the crystal
    """
    # setup model
    m = pynnp.Mode()
    m.log.writeToStdout = False
    m.initialize()
    m.loadSettingsFile(inputnn_fpath)
    m.setupElementMap()
    m.setupElements()
    m.setupCutoff()
    m.setupSymmetryFunctions()
    m.setupSymmetryFunctionMemory()  # comment out when compiled with N2P2_FULL_SFD_MEMORY
    m.setupSymmetryFunctionCache()   # comment out when compiled with N2P2_NO_SF_CACHE
    m.setupSymmetryFunctionGroups()
    if scaling_fpath is not None:
        # Either use symmetry function scaling...
        m.setupSymmetryFunctionScaling("scaling.data")
        m.setupSymmetryFunctionStatistics(False, False, True, False)
    else:
        # ... or calculate raw values.
        m.setupSymmetryFunctionScalingNone()

    # get num molecules per asym unit
    nmols = len(crystal.asym_mols())

    # convert to P1 if not already, calc num molecules
    if crystal.space_group.centering == "primitive":
        crystal = crystal.as_P1()
    else:
        crystal = crystal.as_primitive_P1()

    ratio = len(crystal.asym_mols()) / nmols

    # convert to input data string
    n2p2_data_str = to_n2p2_input_data_str(crystal)

    # calculate symmetry functions
    s = pynnp.Structure()
    s.setElementMap(m.elementMap)
    s.readFromLines(n2p2_data_str.split('\n'))

    # Retrieve cutoff radius form NNP setup.
    cutoff_radius = m.getMaxCutoffRadius()

    # Calculate neighbor list.
    s.calculateNeighborList(cutoff_radius)

    # Calculate symmetry functions for all atoms (use groups).
    # m.calculateSymmetryFunctions(s, False)
    m.calculateSymmetryFunctionGroups(s, False)

    symm_funcs = [atom.G for atom in s.atoms[:int(len(s.atoms) / ratio)]]

    return symm_funcs


def euclidean_distance(descriptor_i: np.ndarray, descriptor_j: np.ndarray) -> float:
    """
    Calculate the Euclidean distance between two symmetry function descriptors.

    Args:
        descriptor_i (np.ndarray): The first symmetry function descriptor.
        descriptor_j (np.ndarray): The second symmetry function descriptor.

    Returns:
        float: The Euclidean distance between the two descriptors.
    """
    a = np.concatenate((descriptor_i), axis=None)
    b = np.concatenate((descriptor_j), axis=None)
    return np.linalg.norm(a - b, ord=2)


def infinity_euclidean_distance(descriptor_i: np.ndarray, descriptor_j: np.ndarray) -> float:
    """
    Calculate the infinity norm (max) distance between two symmetry function descriptors.

    Args:
        descriptor_i (np.ndarray): The first symmetry function descriptor.
        descriptor_j (np.ndarray): The second symmetry function descriptor.

    Returns:
        float: The infinity norm distance between the two descriptors.
    """
    a = np.concatenate((descriptor_i), axis=None)
    b = np.concatenate((descriptor_j), axis=None)
    return np.linalg.norm(a - b, ord=np.inf)


global_var_dict = {}


def set_global_descriptors(descriptors: List[np.ndarray]) -> None:
    """
    Set global variable for descriptors.

    Args:
        descriptors (List[np.ndarray]): List of symmetry function descriptors to be set as global variable.
    """
    global_var_dict['descriptors'] = descriptors


def get_global_descriptors() -> List[np.ndarray]:
    """
    Get global variable for descriptors.

    Returns:
        List[np.ndarray]: List of symmetry function descriptors stored in the global variable.
    """
    return global_var_dict['descriptors']


def get_j(bulk: int) -> int:
    return int((1 + np.sqrt(1 + bulk * 8)) / 2)


def chunk_distance(
    index: int,
    chunksize: int,
    max_j: int,
    method: str
) -> List[List[Union[int, float]]]:
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
    else:
        raise ValueError(
            f"Unknown method {method} for distance calculation. Use 'euclidean' or 'infinity_euclidean'."
        )

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


def calculate_distance(
    descriptors: np.array,
    nprocs: int,
    method: str = "euclidean"
) -> np.ndarray:
    """
    Calculate the distance matrix for a list of symmetry function descriptors using multiprocessing.

    Args:
        descriptors (np.ndarray): Array of symmetry function descriptors.
        nprocs (int): Number of processes to use for parallelization.
        method (str): Method to use for distance calculation, options are "euclidean" or "infinity_euclidean".

    Returns:
        np.ndarray: Distance matrix where D[i, j] is the distance between descriptors i and j.
    """
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
    LOG.debug(
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
    LOG.debug("Pool joined by %s seconds", time.time() - time1)

    D = np.zeros((length, length))

    for result in results:
        for i, j, d_ij in result:  # .get():
            D[i, j] = d_ij
            D[j, i] = d_ij
            if j == i:
                LOG.error("Diagonal ones %s should not be calculated again!", i)

    for i in range(length):
        D[i, i] = 0.0

    return D
