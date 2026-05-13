import numpy as np
from typing import Callable, Union, Tuple

try:
    from chainer import cuda
except ImportError:
    raise ImportError(
        "cspy.util.farthest_point_sampling requires chainer to be installed. "
        "Install it with `pip install chainer`."
    )

def l2_norm(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Calculate l2 norm (distance) of `x` and `y`.
    Args:
        x (numpy.ndarray or cupy): (batch_size, num_point, coord_dim)
        y (numpy.ndarray): (batch_size, num_point, coord_dim)
    Returns (numpy.ndarray): (batch_size, num_point,)
    """
    return ((x - y) ** 2).sum(axis=2)


def farthest_point_sampling(
    pts: np.ndarray,
    k: int,
    initial_idx: Union[int, None] = None,
    metrics: Callable[[np.ndarray, np.ndarray], np.ndarray] = l2_norm,
    skip_initial: bool = False,
    indices_dtype: np.dtype = np.int32,
    distances_dtype: np.dtype = np.float32
):
    """Batch operation of farthest point sampling.
    Code referenced from below link by @Graipher
    https://codereview.stackexchange.com/questions/179561/farthest-point-algorithm-in-python

    Args:
        pts (numpy.ndarray or cupy.ndarray): 2-dim array (num_point, coord_dim)
            or 3-dim array (batch_size, num_point, coord_dim)
            When input is 2-dim array, it is treated as 3-dim array with
            `batch_size=1`.
        k (int): number of points to sample
        initial_idx (int): initial index to start farthest point sampling.
            `None` indicates to sample from random index,
            in this case the returned value is not deterministic.
        metrics (callable): metrics function, indicates how to calc distance.
        skip_initial (bool): If True, initial point is skipped to store as
            farthest point. It stabilizes the function output.
        indices_dtype (): dtype of output `indices`
        distances_dtype (): dtype of output `distances`

    Returns:
        tuple: `indices` and `distances`.
            indices (numpy.ndarray or cupy.ndarray): 2-dim array (batch_size, k)
                indices of sampled farthest points.
                `pts[indices[i, j]]` represents `i-th` batch element of `j-th`
                farthest point.
            distances (numpy.ndarray or cupy.ndarray): 3-dim array
                (batch_size, k, num_point)
    """
    if pts.ndim == 2:
        # Insert batch_size axis
        pts = pts[None, ...]
    assert pts.ndim == 3

    xp = cuda.get_array_module(pts)
    batch_size, num_point, coord_dim = pts.shape
    indices = xp.zeros((batch_size, k), dtype=indices_dtype)

    # distances[bs, i, j] is distance between i-th farthest point `pts[bs, i]`
    # and j-th input point `pts[bs, j]`.
    distances = xp.zeros((batch_size, k, num_point), dtype=distances_dtype)

    if initial_idx is None:
        indices[:, 0] = xp.random.randint(len(pts))
    else:
        indices[:, 0] = initial_idx

    batch_indices = xp.arange(batch_size)
    farthest_point = pts[batch_indices, indices[:, 0]]

    # Minimum distances to the sampled farthest point
    try:
        min_distances = metrics(farthest_point[:, None, :], pts)
    except Exception:
        raise ValueError(
            "min_distances calculation failed. "
            "Check the shape of `pts` and the `metrics` function."
        )

    if skip_initial:
        # Override 0-th `indices` by the farthest point of `initial_idx`
        indices[:, 0] = xp.argmax(min_distances, axis=1)
        farthest_point = pts[batch_indices, indices[:, 0]]
        min_distances = metrics(farthest_point[:, None, :], pts)

    distances[:, 0, :] = min_distances
    for i in range(1, k):
        indices[:, i] = xp.argmax(min_distances, axis=1)
        farthest_point = pts[batch_indices, indices[:, i]]
        dist = metrics(farthest_point[:, None, :], pts)
        distances[:, i, :] = dist
        min_distances = xp.minimum(min_distances, dist)

    return indices, distances
