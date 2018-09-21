from libcpp.string cimport string
from libc.string cimport strtok
from libc.stdlib cimport atoi
import numpy
cimport numpy as np

from sage.interfaces.magma import magma

cdef extern from "string" namespace "std":
    cdef string to_string(int)

def magma_to_numpy_int_matrix(A):
    cdef int nrows = int(magma.Nrows(A))
    cdef int ncols = int(magma.Ncols(A))

    out = numpy.empty( (nrows,ncols), dtype=int )
    cdef np.int_t [:,:] out_view = out

    magma_str_python = str(A)
    cdef char* magma_str = magma_str_python

    cdef char* current_token
    cdef int i,j

    cdef char* delims = " []\n"

    for i in xrange(nrows):
        for j in xrange(ncols):
            if i == 0 and j == 0:
                current_token = strtok(magma_str, delims)
            else:
                current_token = strtok(NULL, delims)

            if current_token is NULL:
                raise ValueError

            out_view[i,j] = atoi(current_token)

    return out

def numpy_int_matrix_to_magma(A,ring):
    Aflat = numpy.ravel(A)

    cdef np.int_t [:] Aview = Aflat
    cdef int n = len(Aflat)

    cdef int i

    cdef string s
    cdef string comma = ","

    s += <char*>("[")
    
    for i in xrange(n):
        s += to_string(Aview[i])
        if i < n-1:
            s += comma

    s += <char*>("]")

    return magma.Matrix(ring, A.shape[0], A.shape[1], magma(s))
