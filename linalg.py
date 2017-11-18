from __future__ import division
from sage.all import *
import numpy
from chaincplx import magma_fns
from scipy import sparse
import functools

try:
    magma('1')
    use_magma = True
except RuntimeError:
    warnings.warn("Could not load Magma. Falling back to Sage for all functionality.")
    use_magma = False

# Check if patched sage
A = matrix(ZZ, 1,1)
if hasattr(A, 'set_unsafe_from_numpy_int_array'):
    patched_sage = True
else:
    patched_sage = False
    print "WARNING: patched Sage not found; performance will be much slower"

def kernel_mod_image(d1,d2):
    # Returns the torsion coefficients of (ker d1)/(im d2), where d1 d2 = 0

    #image = column_space_matrix(d2,encoder)
    kernel = d1.right_kernel_matrix()

    assert d1.dot(d2).count_nonzero() == 0

    AB = d2.factory().bmat([[kernel,-d2]])
    k2 = AB.right_kernel_matrix()
    alpha = k2[0:kernel.shape[1],:]
    divisors = alpha.elementary_divisors()

    return [ divisor for divisor in divisors if divisor != 1 ]

def isiterable_or_slice(i):
    try:
        iter(i)
        return True
    except TypeError:
        return isinstance(i,slice)

right_kernel_use = 'sage'

class GenericMatrix(object):
    def __init__(self):
        raise NotImplementedError

    def __str__(self):
        return str(self.A)
    
    def __repr__(self):
        return repr(self.A)

    def to_magma():
        if self.density == 'dense':
            return self.to_magmadense()
        else:
            return self.to_magmasparse()

    def right_kernel_matrix(self):
        if right_kernel_use == 'sage':
            conv = self.to_sagedense()
        else:
            conv = self.to_magma()
        return self.convert_to_like_self(conv.right_kernel_matrix_())

    def pivots(self):
        return self.to_sagedense().pivots()

    def column_space_matrix(self):
        #print "column_space_matrix:", self.A.shape
        ret = self._constructor(self.A[:,self.pivots()])
        #print "column_space_matrix: done" 
        return ret

    def elementary_divisors(self):
        return self.to_sagedense().elementary_divisors()

    def solve_right(self, v):
        return v.convert_to_like_self(self.to_sagedense().solve_right(v.to_sagedense()))

    def __getitem__(self, i):
        if ( (isinstance(i, tuple) and any(isiterable_or_slice(x) for x in i)) or
                isiterable_or_slice(i) ):
            return self._constructor(self.A[i])
        else:
            return self.A[i]

    def __setitem__(self, i, x):
        if isinstance(x, GenericMatrix):
            self.A[i] = x.A
        elif isinstance(x, GenericVector):
            self.A[i] = x.v
        else:
            self.A[i] = x

    def __neg__(self):
        return self._constructor(-self.A)

class NumpyMatrixFactoryBase(object):
    def __init__(self, constructor, vector_constructor, numpy_dtype, base_module):
        self.constructor = constructor
        self.vector_constructor = vector_constructor
        self.numpy_dtype = numpy_dtype
        self.base_module = base_module

    def bmat(self, block):
        ring = None
        for i in xrange(len(block)):
            for j in xrange(len(block[i])):
                if block[i][j] is None:
                    continue

                if ring is None:
                    ring = block[i][j].ring
                else:
                    assert block[i][j].ring == ring

                block[i][j] = block[i][j].A
        return self.constructor(self.base_module.bmat(block))

    def eye(self, n):
        return self.constructor(self.base_module.eye(n, dtype=self.numpy_dtype))

    def zeros(self, shape):
        return self.constructor(self.base_module.zeros(shape, dtype=self.numpy_dtype))

    def zero_vector(self, n):
        return self.vector_constructor(numpy.zeros(n,
            dtype=self.numpy_dtype))

    def concatenate_vectors(self, a, b):
        return self.vector_constructor(numpy.concatenate((a.v,b.v)))

class ScipySparseMatrixFactory(NumpyMatrixFactoryBase):
    def __init__(self, constructor, vector_constructor, numpy_dtype):
        super(ScipySparseMatrixFactory,self).__init__(constructor,
                vector_constructor, numpy_dtype, sparse)

    def zeros(self, shape):
        return self.constructor(sparse.csc_matrix(shape, dtype=self.numpy_dtype))

class NumpyMatrixFactory(NumpyMatrixFactoryBase):
    def __init__(self, constructor, numpy_dtype):
        super(ScipySparseMatrixFactory,self).__init__(constructor, numpy_dtype, numpy)

class ScipyOrNumpyMatrixOverRingGeneric(GenericMatrix):
    def __init__(self):
        raise NotImplementedError

    def dot(self, b):
        if isinstance(b, GenericVector):
            return self._vector_constructor(self.A.dot(b.v))
        else:
            return self._constructor(self.A.dot(b.A))

    def nrows(self):
        return self.A.shape[0]

    def ncols(self):
        return self.A.shape[1]

    @property
    def shape(self):
        return self.A.shape

    def convert_to_like_self_preserve_density(self,a):
        if a.density == 'sparse':
            return a.to_scipysparse()
        else:
            return a.to_numpydense()

class ScipySparseMatrixOverRingGeneric(ScipyOrNumpyMatrixOverRingGeneric):
    def to_sagedense(self):
        return self.to_numpydense().to_sagedense()

    def to_numpydense(self):
        return NumpyMatrixOverRing(self.A.toarray(), self.ring)

    def to_scipysparse(self):
        return self

    def convert_to_like_self(self,a):
        return a.to_scipysparse()

    def factory(self):
        return ScipySparseMatrixFactory(numpy_dtype=self.A.dtype,
                constructor=self._constructor,
                vector_constructor=self._vector_constructor)

    def count_nonzero(self):
        return len(self.A.nonzero()[0])

    def equals(self, b):
        return self.shape == b.shape and len((self.A != b.A).nonzero()[0]) == 0

    @property
    def density(self):
        return "sparse"

class NumpyMatrixOverRingGeneric(ScipyOrNumpyMatrixOverRingGeneric):
    def __init__(self):
        raise NotImplementedError

    def factory(self):
        return NumpyMatrixFactory(numpy_dtype=self.A.dtype,
                constructor=self._constructor)

    def to_sagedense(self):
        if patched_sage:
            nrows,ncols = self.A.shape
            B = matrix(self.ring, nrows, ncols)
            B.set_unsafe_from_numpy_int_array(self.A)
            return SageDenseMatrix(B)
        else:
            return SageDenseMatrix(matrix(self.ring, self.A))

    def to_numpydense(self):
        return self

    def to_scipysparse(self):
        return ScipySparseMatrixOverRing(sparse.csc_matrix(self.A), ring=self.ring)

    def convert_to_like_self(self,a):
        return a.to_numpydense()

    def count_nonzero(self):
        return numpy.count_nonzero

    def equals(self, b):
        return numpy.array_equal(self.A, self.b.A)

    @property
    def density(self):
        return "dense"

from sage.rings.finite_rings.integer_mod_ring import IntegerModRing_generic

class NumpyMatrixOverZ(NumpyMatrixOverRingGeneric):
    def __init__(self, A):
        self.ring = ZZ
        self._constructor = NumpyMatrixOverZ
        self._vector_constructor = NumpyVectorOverZ
        self.A = numpy.asarray(A)

    def change_ring(self, new_ring):
        if isinstance(new_ring, IntegerModRing_generic):
            return self.NumpyMatrixOverZn(self.A, new_ring.order())
        else:
            raise NotImplementedError

class NumpyMatrixOverZn(NumpyMatrixOverRingGeneric):
    def __init__(self, A, n):
        self.ring = Integers(n)
        self.A = numpy.asarray(A) % n
        self.n = n
        self._constructor = functools.partial(NumpyMatrixOverZn, n=n)
        self._vector_constructor = functools.partial(NumpyVectorOverZn, n=n)

def _scipy_sparse_mod(A, n):
    return A - ((A/n).floor()*n).astype(int)

class ScipySparseMatrixOverZ(ScipySparseMatrixOverRingGeneric):
    def __init__(self, A):
        self.ring = ZZ
        self._constructor = ScipySparseMatrixOverZ
        self._vector_constructor = NumpyVectorOverZ
        if sparse.issparse(A):
            self.A = A
        else:
            self.A = sparse.csc_matrix(A)

    def change_ring(self, new_ring):
        if new_ring is ZZ:
            return self
        elif isinstance(new_ring, IntegerModRing_generic):
            return ScipySparseMatrixOverZn(self.A, new_ring.order())
        else:
            raise NotImplementedError


class ScipySparseMatrixOverZn(ScipySparseMatrixOverRingGeneric):
    def __init__(self, A, n):
        self.ring = Integers(n)
        self._constructor = functools.partial(ScipySparseMatrixOverZn, n=n)
        self._vector_constructor = functools.partial(NumpyVectorOverZn, n=n)
        if sparse.issparse(A):
            self.A = A
        else:
            self.A = sparse.csc_matrix(A)
        self.A = _scipy_sparse_mod(A,2)

class GenericVector(object):
    def __getitem__(self, i):
        if  isiterable_or_slice(i):
            return self._constructor(self.v[i])
        else:
            return self.v[i]

    def __setitem__(self, i, x):
        if isinstance(x, GenericVector):
            self.v[i] = x.v
        else:
            self.v[i] = x

    def __neg__(self):
        return self._constructor(-self.v)

    def __len__(self):
        return len(self.v)

class SageVector(GenericVector):
    def __init__(self, v):
        self.v = v

    def to_sagedense(self):
        return self

    def sageobj(self):
        return self.v

    def to_numpydense(self):
        return NumpyVectorOverRing(self.v.numpy(dtype=numpy_dtype_for_ring(self.ring)),
            ring=self.ring)

    @property
    def ring(self):
        return self.v.base_ring()

class NumpyVectorOverRingGeneric(GenericVector):
    def __init__(self):
        raise NotImplementedError

    def to_sagedense(self):
        return SageVector(vector(self.ring, self.v))

    def to_numpydense(self):
        return self

    def convert_to_like_self(self,a):
        return a.to_numpydense()

    def count_nonzero(self):
        return numpy.count_nonzero(self.v)

class NumpyVectorOverZ(NumpyVectorOverRingGeneric):
    def __init__(self, v):
        self.ring = ZZ
        self.v = numpy.array(v)
        self._constructor = NumpyVectorOverZ

class NumpyVectorOverZn(NumpyVectorOverRingGeneric):
    def __init__(self, v, n):
        self.ring = Integers(n)
        self.v = numpy.array(v) % n
        self.n = n
        self._constructor = functools.partial(NumpyVectorOverZn, n = n)

#class ScipyVectorOverRingTemplate(object):
#    def __init__(self, v, ring, numpy_dtype):
#        self.ring = ring
#        self.numpy_dtype = numpy_dtype
#        self.v = v
#
#    def tosage(self,A):
#        return vector(self.v, self.ring)
#
#    def toscipy(self,A):
#        return numpy.array(A.numpy(dtype=self.numpy_dtype))

def solve_matrix_equation_with_constraint(A, subs_indices, xconstr):
    """ Solves the equation Ax = 0, given the constraint that x[i] = xconstr[i]
    for i in subs_indices. """ 

    subs_indices_list = list(subs_indices)
    subs_indices_set = frozenset(subs_indices)

    notsubs_indices_set = frozenset(xrange(A.shape[1])) - subs_indices_set
    notsubs_indices_list = list(notsubs_indices_set)

    xconstr_reduced = xconstr[subs_indices_list]

    A_reduced_1 = A[:,notsubs_indices_list]
    A_reduced_2 = A[:,subs_indices_list]

    b = -A_reduced_2.dot(xconstr_reduced)

    sol = A_reduced_1.solve_right(b)

    sol_full = A.factory().zero_vector(A.shape[1])
    sol_full[notsubs_indices_list] = sol
    sol_full[subs_indices_list] = xconstr[subs_indices_list]

    assert(A.dot(sol_full).count_nonzero() == 0)
    return sol_full

from sage.matrix.matrix_dense import Matrix_dense
class SageDenseMatrix(GenericMatrix):
    def __init__(self,A):
        if not isinstance(A, Matrix_dense):
            raise TypeError, "Input to SageDenseMatrix constructor must be a sage dense matrix object."
        self.A = A
        self._constructor = SageDenseMatrix

    def to_numpydense(self):
        return NumpyMatrixOverRing(numpy.array(self.A.numpy(dtype=numpy_dtype_for_ring(self.ring))),
            ring=self.ring)

    def to_scipysparse(self):
        return self.to_numpydense().to_scipysparse()

    def to_sagedense(self):
        return self

    def convert_to_like_self(self):
        return self.to_sagedense()

    @property
    def ring(self):
        return self.A.base_ring()

    def solve_right(self, b):
        return SageVector(self.A.solve_right(b.v))

    def right_kernel_matrix_(self):
        #print "right_kernel_matrix:", (self.A.nrows(), self.A.ncols())
        
        # Flint is a lot faster than the default algorithm when working over the
        # integers.
        if self.ring is ZZ:
            kwargs = dict(algorithm='flint')
        else:
            kwargs = dict()

        ret = SageDenseMatrix(self.A.right_kernel_matrix(basis='computed', **kwargs).transpose())

        #print "right_kernel_matrix: done"
        return ret

    def pivots(self):
        return self.A.pivots()

    def elementary_divisors(self):
        return self.A.elementary_divisors()

class MagmaMatrix(GenericMatrix):
    def __init__(self):
        raise NotImplementedError

    def right_kernel_matrix_(self):
        return MagmaDenseMatrix( magma.KernelMatrix(self.A), self.ring)

class MagmaDenseMatrix(MagmaMatrix):
    def __init__(self,A,ring):
        self.A = A
        self.ring = ring
        self._constructor = functools.partial(MagmaDenseMatrix, ring=ring)

    def to_numpydense(self):
        return NumpyMatrixOverRing(magmaconv.numpy_dense_matrix_from_magma(self.A,self.ring),
            self.ring)

    def to_sagedense(self):
        return self.to_numpydense().to_sagedense()

    def convert_to_like_self(self,obj):
        return obj.to_magmadense()

    @property
    def density(self):
        return "dense"

class MagmaSparseMatrix(MagmaMatrix):
    def __init__(self,A,ring):
        self.A = A
        self.ring = ring
        self._constructor = functools.partial(MagmaSparseMatrix, ring=ring)

    def to_scipysparse(self):
        return ScipySparseMatrixOverRing(
                magmaconv.scipy_sparse_matrix_from_magma(self.A,self.ring),
                self.ring)

    @property
    def density(self):
        return "sparse"


def numpy_dtype_for_ring(ring):
    if ring is ZZ:
        return int
    elif isinstance(ring, IntegerModRing_generic):
        return int
    else:
        raise NotImplementedError

def NumpyMatrixOverRing(A, ring):
    if ring is ZZ:
        return NumpyMatrixOverZ(A)
    elif isinstance(ring, IntegerModRing_generic):
        return NumpyMatrixOverZn(A, ring.order())
    else:
        raise NotImplementedError

def NumpyVectorOverRing(v, ring):
    if ring is ZZ:
        return NumpyVectorOverZ(v)
    elif isinstance(ring, IntegerModRing_generic):
        return NumpyVectorOverZn(v, ring.order())
    else:
        raise NotImplementedError

def ScipySparseMatrixOverRing(A, ring):
    if ring is ZZ:
        return ScipySparseMatrixOverZ(A)
    elif isinstance(ring, IntegerModRing_generic):
        return ScipySparseMatrixOverZn(A, ring.order())
    else:
        raise NotImplementedError

#def column_space_intersection_with_other_space(B, other_space_basis_matrix, encoder):
#    W = numpy.bmat([[other_space_basis_matrix, -column_space_matrix(B,encoder)]])
#    Wsage = encoder.sage_matrix_from_numpy(W)
#    kernel_matrix = Wsage.right_kernel_matrix(basis='computed')
#
#    ret = [ encoder.numpy_matrix_multiply(
#                other_space_basis_matrix,
#                encoder.sage_matrix_to_numpy(kernel_matrix.transpose()[0:other_space_basis_matrix.shape[1],i]).flatten(),
#           ) for i in xrange(kernel_matrix.nrows()) ]
#
#    return ret

#def column_space_intersection_with_other_space(B, other_space_basis_matrix, field):
#    Bsage = matrix(field, B)
#    column_space = Bsage.column_space()
#    other_space = span([ vector(field, other_space_basis_matrix[:,i]) for i in
#        xrange(other_space_basis_matrix.shape[1]) ])
#    intersection = column_space.intersection(other_space)
#
#    return [ v for v in intersection.basis() ]

def image_of_constrained_subspace(A,B):
    """ Finds the basis matrix for the space of vectors v such that there exists
    x with Bx = 0 such that Ax = v. """

    if A.shape[1] != B.shape[1]:
        raise ValueError
    #if A.dtype != B.dtype:
    #    raise TypeError

    #AB = numpy.empty( (A.shape[0] + B.shape[0], A.shape[1]), A.dtype)
    #AB[0:A.shape[0],:] = A
    #AB[A.shape[0]:,:] = B

    #a = encoder.numpy_eye(A.shape[0])
    #b = encoder.numpy_zeros( (B.shape[0],A.shape[0]) )
    #target_space_basis = numpy.array(numpy.bmat([[a],
    #                                             [b]]))
    #return [ v[0:A.shape[0]] for v in column_space_intersection_with_other_space(AB, target_space_basis,
    #            encoder) ]

    K = B.right_kernel_matrix()
    AK = A.dot(K)
    return AK.column_space_matrix()
    #scipy.io.savemat('K.mat', {'B': B, 'K': K, 'C': C})
    #return [C[:,i].flat for i in xrange(C.shape[1])]
