# cython: language_level=3, boundscheck=False, wraparound=False
from pathlib import Path
from libc.math cimport ceil, log, pow
cimport numpy as np
import numpy as np

__all__ = ["sobol_vector"]

def _load_data():
    from . import __file__ as init_py_path
    return np.load((Path(init_py_path).parent / "soboldata.npz"))["poly"]

_SOBOL_DATA = _load_data()

cpdef sobol_vector(unsigned int N, unsigned int D):
    """Sobol sequence generation based on:
        S. Joe & F.Y. Kuo, ACM Trans. Math. Softw., 29, 49-57 (2003)
        S. Joe & F.Y. Kuo, SIAM J. Sci. Comput., 30, 2635-2654 (2008)

    Data in soboldata.npz is a modified version of polynomial coefficient table
    based on the D(6) criteria up to dimension 21201, where the $d$ value is
    implicit in the 1st index, the $a$ value is the at position 0 in the 2nd
    index and $s$ is implicit in the nonzero polynomial coefficients.

    Note that this sequence produces different values to the previously used
    code from Burkardt etc., but they are both Sobol type sequences.

    Coefficient data retrieved from:
        https://web.maths.unsw.edu.au/~fkuo/sobol/new-joe-kuo-6.21201
    Retrieval date:
        2019/10/25
    """
    assert N > 0, "input seed for sobol_vector must be > 0"
    cdef double base = 2.0
    cdef unsigned int bitwidth = 32
    cdef unsigned int L = <unsigned>(ceil(log(<double>(N))/log(base)))
    C_arr = np.empty(N, dtype=np.uint32)
    V_arr = np.empty(L+1, dtype=np.uint32)
    X_arr = np.empty(N, dtype=np.uint32)
    pts = np.zeros(D, dtype=np.float64)
    cdef unsigned int[::1] C = C_arr, V = V_arr, X = X_arr
    cdef double[::1] pts_view = pts
    cdef const unsigned int[:, ::1] poly = _SOBOL_DATA
    cdef const unsigned int[::1] m
    cdef unsigned int i, j, k
    cdef unsigned int value, s, a
    with nogil:
        # C[i] = index from the right of the first zero bit of i
        C[0] = 1
        for i in range(1, N):
            C[i] = 1
            value = i
            while ((value & 1U) != 0U):
                value >>= 1U
                C[i] += 1

        # Compute direction numbers V[1] to V[L], scaled by pow(2,32)
        for i in range(1, L + 1):
            V[i] = 1U << (bitwidth - i)  # all m's = 1

        # Evalulate X[0] to X[N-1], scaled by pow(2,32)
        X[0] = 0
        for i in range(1, N):
            X[i] = X[i-1] ^ V[C[i-1]];

        pts_view[0] = <double>(X[N-1]) / pow(base, bitwidth) 
        for j in range(1, D):
            m = poly[j+1]
            # Compute direction numbers V[1] to V[L], scaled by pow(2,32)
            a = m[0]
            for s in range(1, m.shape[0]):
                if m[s] == 0:
                    break
            s = s - 1
            if (L <= s):
                for i in range(1, L + 1):
                    V[i] = <unsigned int>(m[i]) << (bitwidth - i)
            else:
                for i in range(1, s + 1):
                    V[i] = <unsigned int>(m[i]) << (bitwidth - i)
                for i in range(s + 1, L + 1):
                    V[i] = V[i-s] ^ (V[i-s] >> s)
                    for k in range(1, s):
                        V[i] ^= (((a >> (s - 1U - k)) & 1U) * V[i-k]); 

            # Evalulate X[0] to X[N-1], scaled by pow(2,32)
            X[0] = 0
            for i in range(1, N):
                X[i] = X[i-1] ^ V[C[i-1]]

            pts_view[j] = <double>(X[N-1]) / pow(base, bitwidth)
    return pts
