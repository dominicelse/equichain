from sage.matrix.matrix_dense cimport Matrix_dense
from sage.matrix.constructor import matrix
cimport numpy as np
from sage.rings.infinity import Infinity

def sage_matrix_from_numpy_int_array(ring, np.ndarray[np.int_t, ndim=2] A):
    cdef int nrows = A.shape[0]
    cdef int ncols = A.shape[1]
    Asage = matrix(ring,nrows,ncols)

    order_sage = ring.order()
    cdef bint should_quotient
    cdef int order
    if order_sage < Infinity:
        should_quotient = True
        order = order_sage
    else:
        should_quotient = False
        order = 0

    cdef int i
    cdef int j
    cdef int quotient
    for i in xrange(nrows):
        for j in xrange(ncols):
            if should_quotient:
                quotient = A[i,j] % order
            else:
                quotient = A[i,j]
            Asage.set_unsafe_int(i, j, quotient)

    return Asage
