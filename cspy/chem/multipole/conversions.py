import logging
import numpy as np
from cspy.linalg import cartesian_product
from itertools import permutations

LOG = logging.getLogger(__name__)


def traceless_to_cartesian(m):
    """Convert the set of traceless mutlipole moments in
    `m` to unique cartesian compoenents.

    Parameters
    ----------
    m : :obj:`np.ndarray`
        Array of traceless moments in GDMA ordering 

    Returns
    tuple
        Unique cartesian charge, dipole, quadrupole etc.
        components. Will determine which multipoles to return
        based on the size of `m`.
    """

    rank = int(np.sqrt(len(m))) - 1
    charge = m[0]
    if rank > 4:
        LOG.warn(
            "Truncation: maximum rank > 4 is not supported, "
            "only moments hexadecapole and below "
            " will be returned"
        )
    if rank < 1:
        return charge
    dipole = m[[2, 3, 1]]
    if rank < 2:
        return charge, dipole
    half_sqrt3 = 0.5 * np.sqrt(3)
    Q20, Q21c, Q21s, Q22c, Q22s = m[4:9]
    quadrupole = np.zeros(6)
    quadrupole[0] = -0.5 * Q20 + half_sqrt3 * Q22c  # xx
    quadrupole[1] = -0.5 * Q20 - half_sqrt3 * Q22c  # yy
    quadrupole[2] = Q20  # zz
    quadrupole[3] = half_sqrt3 * Q22s  # xy
    quadrupole[4] = half_sqrt3 * Q21c  # xz
    quadrupole[5] = half_sqrt3 * Q21s  # yz
    if rank < 3:
        return charge, dipole, quadrupole
    sqrt23 = np.sqrt(2 / 3)
    sqrt58 = np.sqrt(5 / 8)
    sqrt38 = np.sqrt(3 / 8)
    sqrt512 = np.sqrt(5 / 12)
    sqrt124 = np.sqrt(1 / 24)
    Q30, Q31c, Q31s, Q32c, Q32s, Q33c, Q33s = m[9:16]
    octapole = np.zeros(10)
    octapole[0] = sqrt58 * Q33c - sqrt38 * Q31c  # xxx
    octapole[1] = sqrt58 * Q33s - sqrt124 * Q31s  # xxy
    octapole[2] = -sqrt58 * Q33c - sqrt124 * Q31c  # xyy
    octapole[3] = -sqrt58 * Q33s - sqrt38 * Q31s  # yyy
    octapole[4] = sqrt512 * Q32c - 0.5 * Q30  # xxz
    octapole[5] = sqrt512 * Q32s  # xyz
    octapole[6] = -sqrt512 * Q32c - 0.5 * Q30  # yyz
    octapole[7] = sqrt23 * Q31c  # xzz
    octapole[8] = sqrt23 * Q31s  # yzz
    octapole[9] = Q30  # zzz
    if rank < 4:
        return charge, dipole, quadrupole, octapole

    sqrt5 = np.sqrt(5)
    sqrt35 = np.sqrt(35)
    sqrt10 = np.sqrt(10)
    sqrt70 = np.sqrt(70)
    Q40, Q41c, Q41s, Q42c, Q42s, Q43c, Q43s, Q44c, Q44s = m[16:25]
    hexadecapole = np.zeros(15)
    hexadecapole[0] = 3 / 8 * Q40 - 1 / 4 * sqrt5 * Q42c + 1 / 8 * sqrt35 * Q44c  # xxxx
    hexadecapole[1] = 1 / 8 * (-sqrt5 * Q42s + sqrt35 * Q44s)  # xxxy
    hexadecapole[2] = 1 / 8 * (Q40 - sqrt35 * Q44c)  # xxyy
    hexadecapole[3] = -1 / 8 * (sqrt5 * Q42s + sqrt35 * Q44s)  # xyyy
    hexadecapole[4] = 3 / 8 * Q40 + 1 / 4 * sqrt5 * Q42c + 1 / 8 * sqrt35 * Q44c  # yyyy
    hexadecapole[5] = 1 / 16 * (-3 * sqrt10 * Q41c + sqrt70 * Q43c)  # xxxz
    hexadecapole[6] = 1 / 16 * (-sqrt10 * Q41s + sqrt70 * Q43s)  # xxyz
    hexadecapole[7] = -1 / 16 * (sqrt10 * Q41c + sqrt70 * Q43c)  # xyyz
    hexadecapole[8] = -1 / 16 * (3 * sqrt10 * Q41s + sqrt70 * Q43s)  # yyyz
    hexadecapole[9] = -1 / 2 * Q40 + 1 / 4 * sqrt5 * Q42c  # xxzz
    hexadecapole[10] = 1 / 4 * sqrt5 * Q42s  # xyzz
    hexadecapole[11] = -1 / 2 * Q40 - 1 / 4 * sqrt5 * Q42c  # yyzz
    hexadecapole[12] = np.sqrt(5 / 8) * Q41c  # xzzz
    hexadecapole[13] = np.sqrt(5 / 8) * Q41s  # yzzz
    hexadecapole[14] = Q40  # zzzz
    return charge, dipole, quadrupole, octapole, hexadecapole


def expand_tensor(moments):
    """Expand a cartesian multipole tensor from
    the unique components.

    Parameters
    ----------
    moments : :obj:`np.ndarray`
        vector of length 6 (quadrupole), 10 (octapole), or 15 (hexadecapole)
        containing unique components of the multipole moment tensor.

    Returns
    -------
    :obj:`np.ndarray`
        Rank `l` cartesian tensor for the corresponding
        multipole moments
    """
    nmom = len(moments)
    result = None
    if nmom not in (6, 10, 15):
        raise ValueError(
            "Only unique moment vectors of length 6, 10 or 16 are supported"
        )
    if nmom == 6:
        quadrupole = np.zeros((3, 3))
        quadrupole[(0, 1, 2, 0, 0, 1), (0, 1, 2, 1, 2, 2)] = moments
        quadrupole[(1, 2, 2), (0, 0, 1)] = quadrupole[(0, 0, 1), (1, 2, 2)]
        result = quadrupole
    elif nmom == 10:
        octapole = np.zeros((3, 3, 3))
        idxs = (
            (0, 0, 0),  # xxx
            (0, 0, 1),  # xxy
            (0, 1, 1),  # xyy
            (1, 1, 1),  # yyy
            (0, 0, 2),  # xxz
            (0, 1, 2),  # xyz
            (1, 1, 2),  # yyz
            (0, 2, 2),  # xzz
            (1, 2, 2),  # yzz
            (2, 2, 2),  # zzz
        )
        for n, idx in enumerate(idxs):
            for i, j, k in permutations(idx):
                octapole[i, j, k] = moments[n]
        result = octapole
    elif nmom == 15:
        hexadecapole = np.zeros((3, 3, 3, 3))
        idxs = (
            (0, 0, 0, 0),  # xxxx
            (0, 0, 0, 1),  # xxxy
            (0, 0, 1, 1),  # xxyy
            (0, 1, 1, 1),  # xyyy
            (1, 1, 1, 1),  # yyyy
            (0, 0, 0, 2),  # xxxz
            (0, 0, 1, 2),  # xxyz
            (0, 1, 1, 2),  # xyyz
            (1, 1, 1, 2),  # yyyz
            (0, 0, 2, 2),  # xxzz
            (0, 1, 2, 2),  # xyzz
            (1, 1, 2, 2),  # yyzz
            (0, 2, 2, 2),  # xzzz
            (1, 2, 2, 2),  # yzzz
            (2, 2, 2, 2),  # zzzz
        )
        for n, idx in enumerate(idxs):
            for i, j, k, l in permutations(idx):
                hexadecapole[i, j, k, l] = moments[n]
        result = hexadecapole
    return result


def tensor_to_gdma(tensor):
    """Convert a cartesian multipole tensor to the format
    expported by GDMA i.e. traceless moments.

    Parameters
    ----------
    tensor : :obj:`np.ndarray`
        The cartesian multipole tensor from which to extract
        components. Must be shape (3, 3), (3, 3, 3) or (3, 3, 3, 3)
        i.e. quadrupole, octapole, hexadecapole in xyz ordered indices

    Returns
    -------
    :obj: `np.ndarray`
        Vector of traceless moments corresponding to the input cartesian
        tensor, in the order output by GDMA
    """
    if any(x != 3 for x in tensor.shape):
        raise ValueError("Invalid tensor shape")
    if tensor.ndim == 1:
        return tensor[[2, 0, 1]]
    elif tensor.ndim == 2:
        moments = np.zeros(5)
        s3 = np.sqrt(3)
        moments[0] = tensor[2, 2]
        moments[1] = 2 * tensor[0, 2] / s3
        moments[2] = 2 * tensor[1, 2] / s3
        moments[3] = (2 * tensor[0, 0] + tensor[2, 2]) / s3
        moments[4] = 2 * tensor[0, 1] / s3
        return moments
    elif tensor.ndim == 3:
        moments = np.zeros(7)
        s23 = np.sqrt(2 / 3)
        s512 = np.sqrt(5 / 12)
        s38 = np.sqrt(3 / 8)
        s58 = np.sqrt(5 / 8)
        moments[0] = tensor[2, 2, 2]  # Q30
        moments[1] = tensor[0, 2, 2] / s23  # Q31c
        moments[2] = tensor[1, 2, 2] / s23  # Q31s
        moments[3] = (-0.5 * moments[0] - tensor[1, 1, 2]) / s512  # Q32c
        moments[4] = tensor[0, 1, 2] / s512  # Q32s
        moments[5] = (tensor[0, 0, 0] + s38 * moments[1]) / s58  # Q33c
        moments[6] = (tensor[0, 0, 1] + np.sqrt(1 / 24) * moments[2]) / s58  # Q33s
        return moments
    elif tensor.ndim == 4:
        moments = np.zeros(9)
        s58 = np.sqrt(5 / 8)
        s70 = np.sqrt(70)
        s10 = np.sqrt(10)
        s35 = np.sqrt(35)
        s5 = np.sqrt(5)
        moments[0] = tensor[2, 2, 2, 2]  # Q40
        moments[1] = tensor[0, 2, 2, 2] / s58  # Q41c
        moments[2] = tensor[1, 2, 2, 2] / s58  # Q41s
        moments[3] = -4 / s5 * (tensor[1, 1, 2, 2] + 0.5 * moments[0])  # Q42c
        moments[4] = 4 / s5 * tensor[0, 1, 2, 2]  # Q42s
        moments[5] = (16 * tensor[0, 0, 0, 2] + 3 * s10 * moments[1]) / s70  # Q43c
        moments[6] = (-16 * tensor[1, 1, 1, 2] - 3 * s10 * moments[2]) / s70  # Q43s
        moments[7] = (moments[0] - 8 * tensor[0, 0, 1, 1]) / s35  # Q44c
        moments[8] = -(8 * tensor[0, 1, 1, 1] + s5 * moments[4]) / s35  # Q44s
        return moments
