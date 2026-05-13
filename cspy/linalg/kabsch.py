import numpy as np
from sklearn.neighbors import NearestNeighbors
from typing import Optional

def centroid(A):
    """Calculate the centroid of a set of vectors

    Parameters
    ----------
    A : array
        (N,D) matrix where N is the number of vectors
        and D is the dimension of each vector
    """
    return A.mean(axis=0)


def rmsd_points(A, B, reorient="kabsch"):
    """Rotate the points in `A` onto `B` and calculate
    their RMSD

    Parameters
    ----------
    A : array
        (N,D) matrix where N is the number of vectors and D
        is the dimension of each vector
    B : array
        (N,D) matrix where N is the number of vectors and D
        is the dimension of each vector
    Returns
    -------
    rmsd : float
        root mean squared deviation
    """
    if reorient:
        A = reorient_points(A, B, method=reorient)
    diff = B - A
    return np.sqrt(np.vdot(diff, diff) / diff.shape[0])


def reorient_points(A, B, method="kabsch"):
    """Rotate the points in `A` onto `B`

    Parameters
    ----------
    A : array
        (N,D) matrix where N is the number of vectors and D
        is the dimension of each vector
    B : array
        (N,D) matrix where N is the number of vectors and D
        is the dimension of each vector
    Returns
    -------
    A : array
        (N,D) matrix where N is the number of vectors and D
        is the dimension of each vector, now rotated to fit
        align with B
    """
    if method != "kabsch":
        raise NotImplementedError("Only kabsch algorithm is currently implemented")
    R = kabsch_rotation_matrix(A, B)
    A = np.dot(A, R)
    return A


def kabsch_rotation_matrix(A : np.array, B : np.array, half : bool=False) -> np.array:
    """ Calculate the optimal rotation matrix `R` to rotate
    `A` onto `B`, minimising root-mean-square deviation so that
    this may be then calculated.

    https://en.wikipedia.org/wiki/Kabsch_algorithm

    Reference: Kabsch, W. Acta Cryst. A, 32, 922-923, (1976)
    DOI: http://dx.doi.org/10.1107/S0567739476001873

    Parameters
    ----------
    A : array
        (N,D) matrix where N is the number of vectors and D
        is the dimension of each vector
    B : array
        (N,D) matrix where N is the number of vectors and D
        is the dimension of each vector
    half : bool
        If True, get the rotation matrix that takes you only 
        half way
    Returns
    -------
    R : array
        (D,D) rotation matrix where D is the dimension of each
        vector
    """

    # Calculate the covariance matrix
    cov = np.dot(np.transpose(A), B)

    # Use singular value decomposition to calculate
    # the optimal rotation matrix
    v, s, w = np.linalg.svd(cov)

    if half:
        s = np.sqrt(s)

    # check the determinant to ensure a right-handed
    # coordinate system
    if (np.linalg.det(v) * np.linalg.det(w)) < 0.0:
        s[-1] = -s[-1]
        v[:, -1] = -v[:, -1]

    if half:
        R = v @ np.diag(s) @ w
    R = np.dot(v, w)
    return R


def best_fit_transform_homogeneous(A, B):
    """
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    Parameters
    ----------
      A : array
            Nxm numpy array of corresponding points
      B : array
            Nxm numpy array of corresponding points
    Returns
    -------
      T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
      R: mxm rotation matrix
      t: mx1 translation vector
    """

    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
        Vt[m - 1, :] *= -1
        R = np.dot(Vt.T, U.T)

    # translation
    t = centroid_B.T - np.dot(R, centroid_A.T)

    # homogeneous transformation
    T = np.identity(m + 1)
    T[:m, :m] = R
    T[:m, m] = t

    return T, R, t


def nearest_neighbours(src, dst):
    """
    Find the nearest (Euclidean) neighbor in dst for each point in src
    Input:
        src: Nxm array of points
        dst: Nxm array of points
    Output:
        distances: Euclidean distances of the nearest neighbor
        indices: dst indices of the nearest neighbor
    """

    assert src.shape == dst.shape

    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)
    return distances.ravel(), indices.ravel()


def iterative_closest_points(A, B, init_pose=None, max_iterations=20, tolerance=0.001):
    """ The Iterative Closest Point method: finds
    best-fit transform that maps points A on to points B
    Parameters:
        A: Nxm numpy array of source mD points
        B: Nxm numpy array of destination mD point
        init_pose: (m+1)x(m+1) homogeneous transformation
        max_iterations: exit algorithm after max_iterations
        tolerance: convergence criteria
    Returns:
        T: final homogeneous transformation that maps A on to B
        rmsd: Root mean square deviation error between point clouds
        idx: indices of reordering
    """

    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # make points homogeneous, copy them to maintain the originals
    src = np.c_[A, np.ones(A.shape[0])].T
    dst = np.c_[B, np.ones(B.shape[0])].T

    # apply the initial pose estimation
    if init_pose is not None:
        src = np.dot(init_pose, src)

    prev_error = 0

    T, _, _ = best_fit_transform_homogeneous(src[:m, :].T, dst[:m, :].T)
    src = np.dot(T, src)
    for i in range(max_iterations):
        # find the nearest neighbors between the current source and destination points
        distances, indices = nearest_neighbours(src[:m, :].T, dst[:m, :].T)

        # compute the transformation between the current source and nearest destination points
        T, _, _ = best_fit_transform_homogeneous(src[:m, :].T, dst[:m, indices].T)

        # update the current source
        src = np.dot(T, src)

        # check error
        mean_error = np.mean(distances)
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    # calculate final transformation
    T, _, _ = best_fit_transform_homogeneous(A, src[:m, :].T)

    rmsd = np.sqrt(np.vdot(distances, distances) / distances.shape[0])
    return T, rmsd, indices
