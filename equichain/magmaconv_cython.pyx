# Copyright (c) Dominic Else 2019
#
# This file is part of equichain.
# equichain is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# equichain is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with equichain.  If not, see <https://www.gnu.org/licenses/>.

from libcpp.string cimport string
from libc.string cimport strtok,strspn,strchr
from libc.stdlib cimport atoi,strtol
from scipy import sparse
import numpy
import sys
cimport numpy as np

from sage.interfaces.magma import magma

cdef extern from "string" namespace "std":
    cdef string to_string(int)

def numpy_matrix_from_magma(A):
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

def numpy_vector_from_magma(A):
    cdef int n = int(magma.Ncols(A))

    out = numpy.empty( n, dtype=int )
    cdef np.int_t [:] out_view = out

    magma_str_python = str(A)
    cdef char* magma_str = magma_str_python

    cdef char* current_token
    cdef int i

    cdef char* delims = " ()\n"

    for i in xrange(n):
        if i == 0:
            current_token = strtok(magma_str, delims)
        else:
            current_token = strtok(NULL, delims)

        if current_token is NULL:
            raise ValueError

        out_view[i] = atoi(current_token)

    return out

def numpy_int_matrix_to_magma(A, ring):
    assert len(A.shape) == 2
    return _numpy_int_matrix_or_vector_to_magma(A, ring)

def numpy_int_vector_to_magma(A, ring):
    assert len(A.shape) == 1
    return _numpy_int_matrix_or_vector_to_magma(A, ring)

def _numpy_int_matrix_or_vector_to_magma(A,ring):
    if len(A.shape) == 1:
        vector = True
    elif len(A.shape) == 2:
        vector = False
    else:
        raise ValueError

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

    if vector:
        return magma.Vector(ring, magma(s))
    else:
        return magma.Matrix(ring, A.shape[0], A.shape[1], magma(s))

cdef const char* check_strchr(const char* s, int c):
    cdef const char* ret = strchr(s,c)
    if ret == NULL:
        raise IndexError
    return ret

cdef long int check_strtol(const char* s):
    cdef char* endptr
    cdef int ret = strtol(s, &endptr, 10)
    if endptr == s:
        raise ValueError
    return ret

def scipy_sparse_matrix_from_magma(A, sparse_matrix_class=sparse.coo_matrix):
    seq = magma.ElementToSequence(A)
    length = int(magma('#' + seq.name()))
    sseq_py = str(seq)
    del seq
    cdef const char* sseq = sseq_py
    #cdef const char* sseq = sseq_py

    data = numpy.empty( length, dtype=int)
    i = numpy.empty( length, dtype=int)
    j = numpy.empty( length, dtype=int)

    cdef np.int_t [:] i_c = i
    cdef np.int_t [:] j_c = j
    cdef np.int_t [:] data_c = data

    cdef const char* curpos=sseq
    cdef int k=0

    for k in range(length):
        curpos = check_strchr(curpos, '<')
        curpos += 1
        i_c[k] = check_strtol(curpos)-1

        curpos = check_strchr(curpos, ',')
        curpos += 1
        curpos += strspn(curpos, " ")
        j_c[k] = check_strtol(curpos)-1

        curpos = check_strchr(curpos, ',')
        curpos += 1
        curpos += strspn(curpos, " ") 
        data_c[k] = check_strtol(curpos)

        k += 1

    del sseq_py

    assert k == length

    Acoo = sparse.coo_matrix((data, (i,j)), (int(magma.Nrows(A)), int(magma.Ncols(A))))

    if sparse_matrix_class is sparse.coo_matrix:
        A = Acoo
    else:
        A = sparse_matrix_class(Acoo)

    return A
