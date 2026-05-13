import numpy as np
from .kabsch import rmsd_points, iterative_closest_points


def cartesian_product(*arrays):
    """Form the cartesian product of a number of input arrays
    i.e. A x B x C etc. in set theory.

    Parameters
    ----------
    *arrays : array_like
        N arrays of values from which to form the product.

    Returns
    -------
    :obj: `np.ndarray`
        (M, N) array of all combinations in all orders

    """
    arrays = [np.asarray(a) for a in arrays]
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[..., i] = a
    return arr.reshape(-1, la)


def transform_tensor(T, tensor):
    """Transform `tensor` by the provided covariant transformation
    matrix `T`.

    Parameters
    ----------
    T : :obj:`np.ndarray`
        The transformation from the current basis to the new basis
    tensor : :obj:`np.ndarray`
        The tensor to be transformed.

    Returns
    -------
    :obj:`np.ndarray`
        tensor transformed into the new basis
    """
    degree = tensor.ndim
    if degree > 4:
        raise NotImplementedError("transform_tensor only implemented for ndim < 4")
    if degree == 1:
        return np.dot(T, tensor)
    elif degree == 2:
        return np.einsum("ik,jl,kl->ij", T, T, tensor)
    elif degree == 3:
        return np.einsum("il,jm,kn,lmn->ijk", T, T, T, tensor)
    elif degree == 4:
        return np.einsum("ai,bj,ck,dl,ijkl->abcd", T, T, T, T, tensor)


def plane_of_best_fit(points, return_projection=False):
    """Calculate the best plane of best fit to a set of points.

    Parameters
    ----------
    points : array
        (N, 3) array of points in cartesian space.
    return_projection : bool, optional
        Whether or not to return the projected points on the new plane
        as a (N, 2) array

    Returns
    -------
    tuple
        tuple of plane normal, magnitude, rmsd and (optionally) projected points
    """
    c = np.mean(points, axis=0)
    xyz = points.copy()
    xyz -= c
    u, s, vh = np.linalg.svd(xyz.T)
    normal = u[:, 2]
    d = s[2]
    d = -1 * normal.dot(c)
    rmsd = np.sqrt((xyz.dot(normal) ** 2).sum()) / (len(xyz))
    if return_projection:
        along_a = np.dot(xyz, u[:, 0])
        along_b = np.dot(xyz, u[:, 1])
        amin, amax = np.min(along_a), np.max(along_a)
        bmin, bmax = np.min(along_b), np.max(along_b)
        P = np.asarray([amin * u[:, 0], amax * u[:, 0], bmin * u[:, 1], bmax * u[:, 1]])
        return normal, d, rmsd, P
    return normal, d, rmsd


__all__ = [
    "cartesian_product",
    "rmsd_points",
    "iterative_closest_points",
    "transform_tensor",
    "plane_of_best_fit",
]
