from sage.all import *
import numpy
from chaincplx import magma_fns
from scipy import sparse

try:
    magma('1')
    use_magma = True
except RuntimeError:
    warnings.warn("Could not load Magma. Falling back to Sage for all functionality.")
    use_magma = False

def kernel_mod_image(d1,d2):
    # Returns the torsion coefficients of (ker d1)/(im d2), where d1 d2 = 0

    #image = column_space_matrix(d2,encoder)
    d1 = d1.to_numpydense()
    d2 = d2.to_numpydense()

    kernel = d1.right_kernel_matrix().to_numpydense()

    #assert numpy.count_nonzero(encoder.numpy_matrix_multiply(d1, d2)) == 0
    print "TODO: restore a check here."

    AB = d1.factory().bmat(([[kernel,-d2]])
    k2 = right_kernel_matrix(AB, encoder) 
    alpha = k2[0:kernel.shape[1],:]
    divisors = encoder.sage_matrix_from_numpy(alpha).elementary_divisors()

    return [ divisor for divisor in divisors if divisor != 1 ]

class NumpyEncoder(object):
    def solve_matrix_equation(self,A,b):
        return self.sage_vector_to_numpy(
                self.sage_matrix_from_numpy(A).solve_right(
                    self.sage_vector_from_numpy(b) ) )

    def matrix_passthrough(self, fn, A, *args, **kwargs):
        A = self.sage_matrix_from_numpy(A)
        ret = getattr(A, fn)(*args, **kwargs)
        return self.sage_matrix_to_numpy(ret)

class GenericMatrix(object):
    def __init__(self):
        raise NotImplementedError

    def right_kernel_matrix(self):
        return self.to_sagedense().right_kernel_matrix().convert_to_like(A)

    def pivots(self):
        return self.to_sagedense().pivots()

    def column_space_matrix(self):
        return self.A[:,self.pivots()]

class ScipyOrNumpyMatrixOverRingTemplate(GenericMatrix):
    def __init__(self):
        raise NotImplementedError

    def dot(self, b):
        return self._canonicalize(self.A.dot(b.A))

    def __getitem__(self, i):
        return self.A[i]

    @property
    def shape(self):
        return self.A.shape

class ScipySparseMatrixOverRingTemplate(ScipyOrNumpyMatrixOverRingTemplate):
    def __init__(self):
        raise NotImplementedError

    def factory(self):
        return sparse

    def to_sagedense(self):
        return self.tonumpydense().tosagedense()

    def to_numpydense(self):
        return NumpyMatrixOverRing(self, self.A, self.ring)

    def to_scipysparse(self):
        return self

    def eye_like(self, n):
        return self._canonicalize(sparse.eye(n, dtype=self.A.dtype))

    def zeros_like(self, shape):
        return self._canonicalize(sparse.dok_matrix(shape, dtype=self.A.dtype))

class NumpyMatrixOverRingTemplate(ScipyOrNumpyMatrixOverRingTemplate):
    def __init__(self, A, ring):
        self.A = A
        self.ring = ring

    def factory(self):
        return numpy

    def to_sagedense(self):
        nrows,ncols = self.A.shape
        B = matrix(self.ring, nrows, ncols)
        B.set_unsafe_from_numpy_int_array(self.A)
        return SageDenseMatrix(B)

    def to_numpydense(self):
        return self

    def to_scipysparse(self):
        raise NotImplementedError

    def eye_like(self, n):
        return self._canonicalize(numpy.eye(n, dtype=self.A.dtype))

    def zeros_like(self, shape):
        return self._canonicalize(numpy.zeros(n, dtype=self.A.dtype))

class NumpyMatrixOverZ(NumpyMatrixOverRingTemplate):
    def __init__(self, A):
        self.ring = ZZ
        self.A = A

    def _canonicalize(self, A):
        return NumpyMatrixOverZ(A)

class NumpyMatrixOverZn(NumpyMatrixOverRingTemplate):
    def __init__(self, A, n):
        self.ring = Integers(n)
        self.A = A % n
        self.n = n

    def _canonicalize(self, A):
        return NumpyMatrixOverZn(A, self.n)

class ScipySparseMatrixOverZ(ScipySparseMatrixOverRingTemplate):
    def __init__(self, A):
        self.ring = ZZ
        self.A = A
        self.bmat = sparse.bmat

    def _canonicalize(self, A):
        return ScipySparseMatrixOverZ(A)

class ScipySparseMatrixOverZn(ScipySparseMatrixOverRingTemplate):
    def __init__(self, A, n):
        self.ring = Integers(n)
        self.A = A % n
        self.n = n

    def _canonicalize(self, A):
        return ScipySparseMatrixOverZn(A, self.n)

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

class SageDenseMatrix(object):
    def __init__(self,A):
        self.A = A

    def to_numpydense(self):
        return NumpyMatrixOverRing(numpy.array(self.A.numpy(dtype=numpy_dtype_for_ring(self.A.base_ring()))),
            ring=self.A.base_ring())

    def to_scipysparse(self):
        raise NotImplementedError

    def to_sagedense(self):
        return self

    @property
    def ring(self):
        return self.A.base_ring()

    def right_kernel_matrix(A):
        # Flint is a lot faster than the default algorithm when working over the
        # integers.
        if self.ring is ZZ:
            kwargs = dict(algorithm='flint')
        else:
            kwargs = dict()

        ret = SageDenseMatrix(self.A.right_kernel_matrix(basis='computed', **kwargs).transpose())
        return ret

    def pivots(self):
        return self.A.pivots()

from sage.rings.finite_rings.integer_mod_ring import IntegerModRing_generic
def NumpyMatrixOverRing(A, ring):
    if ring is ZZ:
        return NumpyMatrixOverZ(A)
    elif isinstance(ring, IntegerModRing_generic):
        return NumpyMatrixOverZn(A, ring.order())
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

def image_of_constrained_subspace(A,B,encoder):
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

    K = B.right_kernel_matrix().to_numpydense()
    AK = A.dot(K.to_numpydense())
    C = AK.column_space_matrix()
    #scipy.io.savemat('K.mat', {'B': B, 'K': K, 'C': C})
    return [C[:,i].flat for i in xrange(C.shape[1])]
