from sage.all import *
from scipy import sparse
import itertools
import numpy

import contextlib
import magmaconv_cython

@contextlib.contextmanager
def printoptions(*args, **kwargs):
    original = numpy.get_printoptions()
    numpy.set_printoptions(*args, **kwargs)
    try:
        yield
    finally: 
        numpy.set_printoptions(**original)

from sage.rings.finite_rings.integer_mod_ring import IntegerModRing_generic
def convert_sage_ring_to_magma(ring):
    if ring == ZZ:
        return magma.IntegerRing()
    elif isinstance(ring, IntegerModRing_generic):
        return magma.IntegerRing(ring.order())
    else:
        raise NotImplementedError

class AutoIncrList(object):
    def __init__(self, totallen):
        self.data = [None]*totallen
        self.totallen = totallen
        self.i = 0

    def append(self,val):
        print self.i
        self.data[self.i] = val
        self.i += 1

    def finished():
        assert self.i == self.totallen

def magma_sparse_matrix_from_scipy(A, ring):
    ring = convert_sage_ring_to_magma(ring)

    A = A.tocsr()

    magmadata = AutoIncrList(2*A.nnz + A.shape[0])
    for i in xrange(A.shape[0]):
        magmadata.append(A.indptr[i+1]-A.indptr[i])
        for indx,val in itertools.izip(
                A.indices[A.indptr[i]:A.indptr[i+1]]+1,
                A.data[A.indptr[i]:A.indptr[i+1]]):
            magmadata.append(indx)
            magmadata.append(val)

    return magma.SparseMatrix(ring, A.shape[0], A.shape[1], magmadata.data)

def magma_dense_matrix_from_numpy(A, ring):
    if ring != ZZ:
        raise NotImplementedError

    return magmaconv_cython.numpy_int_matrix_to_magma(A)

def scipy_sparse_matrix_from_magma(A, sparse_matrix_class=sparse.coo_matrix):
    magmadata = [ tuple(x) for x in magma.ElementToSequence(A) ]

    data = numpy.empty( len(magmadata), dtype=int)
    i = numpy.empty( len(magmadata), dtype=int)
    j = numpy.empty( len(magmadata), dtype=int)
    for k,tup in enumerate(magmadata):
        i[k] = tup[0]-1
        j[k] = tup[1]-1
        data[k] = tup[2]

    Acoo = sparse.coo_matrix((data, (i,j)), (magma.Nrows(A), magma.Ncols(A)))

    if sparse_matrix_class is sparse.coo_matrix:
        A = Acoo
    else:
        A = sparse_matrix_class(Acoo)

    return A
        
def numpy_dense_matrix_from_magma(A):
    return magmaconv_cython.magma_to_numpy_int_matrix(A)
