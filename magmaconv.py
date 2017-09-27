from sage.all import *
from scipy import sparse
import itertools
import numpy

def convert_sage_ring_to_magma(ring):
    if ring == ZZ:
        return magma.IntegerRing()
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

def magma_sparse_matrix(A, ring):
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

def scipy_sparse_matrix(A, dtype=int, sparse_matrix_class=sparse.coo_matrix):
    magmadata = [ tuple(x) for x in magma.ElementToSequence(A) ]

    data = numpy.empty( len(magmadata), dtype=dtype)
    i = numpy.empty( len(magmadata), dtype=int)
    j = numpy.empty( len(magmadata), dtype=int)
    for k,tup in enumerate(magmadata):
        i[k] = tup[0]-1
        j[k] = tup[1]-1
        data[k] = tup[2]

    Acoo = sparse.coo_matrix((data, (i,j)), (magma.Nrows(A), magma.Ncols(A)))

    if sparse_matrix_class is sparse.coo_matrix:
        return Acoo
    else:
        return sparse_matrix_class(Acoo)
