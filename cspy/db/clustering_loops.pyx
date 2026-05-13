# cython: language_level=3
cimport cython
cimport numpy as np
import numpy as np
from libc.math cimport fabs, sqrt
from libc.stdlib cimport malloc, free
from libc.stdio cimport printf
ctypedef np.uint8_t uint8
cdef extern from "math.h":
    double INFINITY


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline double l1_norm(const double[::1] arr1, const double[::1] arr2, const int seq_length) nogil:
    cdef double distance = 0
    cdef int i
    for i in range(seq_length):
        distance += fabs(arr1[i] - arr2[i])
    return distance


@cython.boundscheck(False)
@cython.wraparound(False)
cdef double cosine_similarity(const double[::1] x, const double[::1] y, const int n) nogil:
    cdef double total = 0.0, xmag = 0.0, ymag=0.0
    cdef int i
    for i in range(n):
        total += x[i] * y[i]
        xmag += x[i]*x[i]
        ymag += y[i]*y[i]
    return total / (sqrt(xmag) * sqrt(ymag))


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline double _cdtw_sakoe_chiba(const double[::1] arr1, const double[::1] arr2, const int seq_length, const int r) nogil:
    cdef double x1, x2;
    cdef int seq1_length, seq2_length;
    # For now, both sequences have to be of the same length.
    # For low values of r this restriction would be implied anyway.
    seq1_length = seq2_length = seq_length;

    # Note: To save a little memory, our 'column' is not a full column of the
    # DTW matrix, but only the part within r steps from the diagonal (although
    # if 2*r+1 > seq2_length, this approach actually costs more memory). This
    # makes the formulas a bit different than usual. For example,
    # curr_column[1+r] always represents the element on the diagonal.
    # Columns need r+1+r elements to cover the range of the allowed time skew
    # on both sides of the diagonal of the DTW matrix. Two extra elements are
    # fixed to infinity to simplify finding best_value further below.
    cdef int column_size = 2* r + 3
    cdef double *prev_column = <double *> malloc(column_size * sizeof(double))
    if not prev_column:
        return -1
    cdef double *curr_column = <double *> malloc(column_size * sizeof(double))
    if not curr_column:
        return -1
    cdef double *swap_temp = <double *> malloc(column_size * sizeof(double))
    if not swap_temp:
        return -1
    # In the DTW matrix the items left of the first column must be infinite, as
    # must values below the bottom row.
    cdef int i, j, col, row
    cdef int top_valid_row = 0
    cdef int bottom_valid_row
    for i in range(column_size):
        prev_column[i] = INFINITY
        curr_column[i] = INFINITY
    # We do of course need a place to start from. This forces the first matrix
    # element to become |arr1[0]-arr2[0]|
    prev_column[r + 1] = 0
    cdef double error, left_value, left_under_value, under_value, best_value
    for col in range(seq1_length):
        # Pick range of rows, ensuring 0 <= j < seq2_length
        bottom_valid_row = max(1, -col+r+1)
        top_valid_row = min(column_size-2, seq2_length-1-col+r+1)
        # Fill up this column
        for row in range(bottom_valid_row, top_valid_row + 1):
            # Compute mapping to the index in the full DTW matrix.
            i = col
            j = col - r + row - 1
            # Read values from the numpy array
            x1 = arr1[i]
            x2 = arr2[j]

            # Compute their absolute error
            error = fabs(x1-x2)

            # Determine via which path to get here with lowest cumulative error
            left_value = prev_column[row+1]
            left_under_value = prev_column[row]
            under_value = curr_column[row-1]
            best_value = min(left_under_value, min(left_value, under_value))
            # Remember the best possible cumulative error for this point
            curr_column[row] = error + best_value

        # Move to the next column, forget the previous one. (*curr_column is
        # only written to, so it need not be cleaned)
        for i in range(column_size):
            swap_temp[i] = prev_column[i]
            prev_column[i] = curr_column[i]
            curr_column[i] = swap_temp[i]

    free(swap_temp)
    free(prev_column)
    free(curr_column)
    # The result can be found in the upper right corner of the DTW matrix
    return prev_column[top_valid_row]


@cython.boundscheck(False)
@cython.wraparound(False)
cdef double cdtw_sakoe_chiba(const double[::1] arr1, const double[::1] arr2, const int seq_length, const int r) nogil:
    if r == 0:
        return l1_norm(arr1, arr2, seq_length)
    else:
        return _cdtw_sakoe_chiba(arr1, arr2, seq_length, r)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef int iterate_cdtw_sakoe_chiba(const double[::1] arr1, const double[::1] arr2, const int seq_length, int band_range, double threshold) nogil:
    cdef int r
    cdef double diff
    cdef double distance
    distance = l1_norm(arr1, arr2, seq_length)
    for r in range(1, band_range):
        diff = _cdtw_sakoe_chiba(arr1, arr2, seq_length, r)
        if diff == -1:
            return 1
        if diff < distance:
            distance = diff
        if distance <= threshold:
            break
    return distance <= threshold


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void find_duplicates_cdtw_cosine(
        const double[:,::1] ed, const double[:, ::1] xrds, int[::1] equivalent_to,
        const double ethresh, const double dthresh, const double cdtw_thresh, const double cos_thresh) nogil:
    cdef int N = ed.shape[0]
    cdef int seq_length = xrds.shape[1]
    cdef int i, j
    cdef double e1, d1, e2, d2
    for i in range(N):
        equivalent_to[i] = -1
    for i in range(N):
        if equivalent_to[i] > -1:
            continue
        e1 = ed[i, 0]
        d1 = ed[i, 1]
        for j in range(i + 1, N):
            if equivalent_to[j] > -1:
                continue
            e2 = ed[j, 0]
            if (fabs(e2 - e1) > ethresh): break
            d2 = ed[j, 1]
            if (fabs(d2 - d1) > dthresh): continue
            if cosine_similarity(xrds[i, :], xrds[j, :], seq_length) < cos_thresh:
                continue
            if iterate_cdtw_sakoe_chiba(xrds[i, :], xrds[j, :], seq_length, 10, cdtw_thresh):
                equivalent_to[j] = i


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void find_duplicates_cdtw(
        const double[:,::1] ed, const double[:, ::1] xrds, int[::1] equivalent_to,
        const double ethresh, const double dthresh, const double cdtw_thresh) nogil:
    cdef int N = ed.shape[0]
    cdef int seq_length = xrds.shape[1]
    cdef int i, j
    cdef double e1, d1, e2, d2
    for i in range(N):
        equivalent_to[i] = -1
    for i in range(N):
        if equivalent_to[i] > -1:
            continue
        e1 = ed[i, 0]
        d1 = ed[i, 1]
        for j in range(i + 1, N):
            if equivalent_to[j] > -1:
                continue
            e2 = ed[j, 0]
            if (fabs(e2 - e1) > ethresh): break
            d2 = ed[j, 1]
            if (fabs(d2 - d1) > dthresh): continue
            if iterate_cdtw_sakoe_chiba(xrds[i, :], xrds[j, :], seq_length, 10, cdtw_thresh):
                equivalent_to[j] = i


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void find_duplicates_cosine(
        const double[:,::1] ed, const double[:, ::1] xrds, int[::1] equivalent_to,
        const double ethresh, const double dthresh, const double cos_thresh) nogil:
    cdef int N = ed.shape[0]
    cdef int seq_length = xrds.shape[1]
    cdef int i, j
    cdef double e1, d1, e2, d2
    for i in range(N):
        equivalent_to[i] = -1
    for i in range(N):
        if equivalent_to[i] > -1:
            continue
        e1 = ed[i, 0]
        d1 = ed[i, 1]
        for j in range(i + 1, N):
            if equivalent_to[j] > -1:
                continue
            e2 = ed[j, 0]
            if (fabs(e2 - e1) > ethresh): break
            d2 = ed[j, 1]
            if (fabs(d2 - d1) > dthresh): continue
            if cosine_similarity(xrds[i, :], xrds[j, :], seq_length) > cos_thresh:
                equivalent_to[j] = i


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void find_duplicates_cdtw_cosine_between_list(
        const double[:,::1] ed_x, const double[:, ::1] xrds_x, 
        const double[:,::1] ed_y, const double[:, ::1] xrds_y, 
        int[::1] equivalent_to, const double ethresh, const double dthresh, 
        const double cdtw_thresh, const double cos_thresh
) nogil:
    cdef int N = ed_x.shape[0]
    cdef int M = ed_x.shape[0]
    cdef int seq_length = xrds_x.shape[1]
    cdef int i, j
    cdef double e1, d1, e2, d2
    for i in range(M):
        equivalent_to[i] = -1
    for i in range(N):
        e1 = ed_x[i, 0]
        d1 = ed_x[i, 1]
        for j in range(M):
            if equivalent_to[j] > -1:
                continue
            e2 = ed_y[j, 0]
            if (fabs(e2 - e1) > ethresh): break
            d2 = ed_y[j, 1]
            if (fabs(d2 - d1) > dthresh): continue
            if cosine_similarity(xrds_x[i, :], xrds_y[j, :], seq_length) < cos_thresh:
                continue
            if iterate_cdtw_sakoe_chiba(xrds_x[i, :], xrds_y[j, :], seq_length, 10, cdtw_thresh):
                equivalent_to[j] = i


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void find_duplicates_cdtw_between_list(
        const double[:,::1] ed_x, const double[:, ::1] xrds_x, 
        const double[:,::1] ed_y, const double[:, ::1] xrds_y, 
        int[::1] equivalent_to, const double ethresh, const double dthresh, 
        const double cdtw_thresh
) nogil:
    cdef int N = ed_x.shape[0]
    cdef int M = ed_x.shape[0]
    cdef int seq_length = xrds_x.shape[1]
    cdef int i, j
    cdef double e1, d1, e2, d2
    for i in range(M):
        equivalent_to[i] = -1
    for i in range(N):
        e1 = ed_x[i, 0]
        d1 = ed_x[i, 1]
        for j in range(M):
            if equivalent_to[j] > -1:
                continue
            e2 = ed_y[j, 0]
            if (fabs(e2 - e1) > ethresh): break
            d2 = ed_y[j, 1]
            if (fabs(d2 - d1) > dthresh): continue
            if iterate_cdtw_sakoe_chiba(xrds_x[i, :], xrds_y[j, :], seq_length, 10, cdtw_thresh):
                equivalent_to[j] = i


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void find_duplicates_cosine_between_list(
        const double[:,::1] ed_x, const double[:, ::1] xrds_x, 
        const double[:,::1] ed_y, const double[:, ::1] xrds_y, 
        int[::1] equivalent_to, const double ethresh, const double dthresh, 
        const double cos_thresh
) nogil:
    cdef int N = ed_x.shape[0]
    cdef int M = ed_x.shape[0]
    cdef int seq_length = xrds_x.shape[1]
    cdef int i, j
    cdef double e1, d1, e2, d2
    for i in range(M):
        equivalent_to[i] = -1
    for i in range(N):
        e1 = ed_x[i, 0]
        d1 = ed_x[i, 1]
        for j in range(M):
            if equivalent_to[j] > -1:
                continue
            e2 = ed_y[j, 0]
            if (fabs(e2 - e1) > ethresh): break
            d2 = ed_y[j, 1]
            if (fabs(d2 - d1) > dthresh): continue
            if cosine_similarity(xrds_x[i, :], xrds_y[j, :], seq_length) > cos_thresh:
                equivalent_to[j] = i


def find_duplicates(const double[:,::1] ed, const double[:, ::1] xrds,
        method, double ethresh, double dthresh, double cdtw_thresh, double cos_thresh):
    equivalent_to = - np.ones(ed.shape[0], dtype=np.int32)
    cdef int[::1] equivalent_to_view = equivalent_to
    if method == "cdtw_cos":
        with nogil:
            find_duplicates_cdtw_cosine(ed, xrds, equivalent_to_view, ethresh, dthresh, cdtw_thresh, cos_thresh)
    elif method == "cdtw":
        with nogil:
            find_duplicates_cdtw(ed, xrds, equivalent_to_view, ethresh, dthresh, cdtw_thresh)
    else:
        with nogil:
            find_duplicates_cosine(ed, xrds, equivalent_to_view, ethresh, dthresh, cos_thresh)

    return equivalent_to

def find_duplicates_between_list(
    const double[:,::1] ed_x, const double[:, ::1] xrds_x,
    const double[:,::1] ed_y, const double[:, ::1] xrds_y,
    method, double ethresh, double dthresh, double cdtw_thresh, double cos_thresh
):
    equivalent_to = - np.ones(ed_x.shape[0], dtype=np.int32)
    cdef int[::1] equivalent_to_view = equivalent_to
    if method == "cdtw_cos":
        with nogil:
            find_duplicates_cdtw_cosine_between_list(
                ed_x, xrds_x, ed_y, xrds_y, 
                equivalent_to_view, ethresh, dthresh, cdtw_thresh, cos_thresh
            )
    elif method == "cdtw":
        with nogil:
            find_duplicates_cdtw_between_list(
                ed_x, xrds_x, ed_y, xrds_y, 
                equivalent_to_view, ethresh, dthresh, cdtw_thresh
            )
    else:
        with nogil:
            find_duplicates_cosine_between_list(
                ed_x, xrds_x, ed_y, xrds_y, 
                equivalent_to_view, ethresh, dthresh, cos_thresh
            )

    return equivalent_to

def cosine_distance(double[::1] arr1, double[::1] arr2):
    assert arr1.shape[0] == arr2.shape[0], "arrays must be of same shape"
    cdef int seq_length = arr1.shape[0]
    return cosine_similarity(arr1, arr2, seq_length)


def cdtw_distance(double[::1] arr1, double[::1] arr2, int r):
    assert arr1.shape[0] == arr2.shape[0], "arrays must be of same shape"
    cdef int seq_length = arr1.shape[0]
    return cdtw_sakoe_chiba(arr1, arr2, seq_length, r)

def iterative_cdtw_distance(double[::1] arr1, double[::1] arr2, int r, double threshold):
    assert arr1.shape[0] == arr2.shape[0], "arrays must be of same shape"
    cdef int seq_length = arr1.shape[0]
    return iterate_cdtw_sakoe_chiba(arr1, arr2, seq_length, r, threshold)
