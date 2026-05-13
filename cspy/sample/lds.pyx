cimport numpy as np
import numpy as np
from libc.math cimport pow


cdef double phi(const unsigned int d) nogil:
    cdef int iterations = 30
    cdef int i
    cdef double x = 2.0
    for i in range(iterations):
        x = pow(1 + x, 1.0 / (d + 1.0))
    return x


cdef void alpha(double[::1] a) nogil:
    cdef int dims = a.shape[0]
    cdef double g = phi(dims)
    cdef int i
    for i in range(dims):
        a[i] = pow(1 / g, i + 1) % 1


cpdef quasirandom_kgf(const unsigned int N, const unsigned int D):
    """ Generate an N dimensional Korobov type quasi-random vector
    based on the generalized Fibonacci sequence.

    Based on the R_1, R_2 sequences available here:
        https://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences/
    """
    cdef double offset = 0.5
    a = np.empty(D, dtype=np.float64)
    alpha(a)
    return (offset + a * (N + 1)) % 1

cpdef quasirandom_kgf_batch(const unsigned int L, const unsigned int U, const unsigned int D):
    """ Generate an N dimensional Korobov type quasi-random vector
    based on the generalized Fibonacci sequence.

    Based on the R_1, R_2 sequences available here:
        https://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences/
    """
    cdef double offset = 0.5
    cdef double[::1] a_view
    a = np.empty(D, dtype=np.float64)
    alpha(a)
    result = np.empty((U - L + 1, D), dtype=np.float64)
    N = np.arange(L, U + 1)
    return (offset + a[np.newaxis, :] * (N[:, np.newaxis] + 1)) % 1
