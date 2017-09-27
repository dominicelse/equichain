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

def right_kernel_matrix(A,encoder):
    if isinstance(encoder, NumpyEncoderZ):
        kwargs = dict(algorithm='flint')
    else:
        kwargs = dict()
    ret = numpy.ascontiguousarray(encoder.matrix_passthrough('right_kernel_matrix', A,
        basis='computed', **kwargs).T)
    return ret

def kernel_mod_image(d1,d2,encoder):
    # Returns the torsion coefficients of (ker d1)/(im d2), where d1 d2 = 0

    #image = column_space_matrix(d2,encoder)
    kernel = right_kernel_matrix(d1, encoder)

    assert numpy.count_nonzero(encoder.numpy_matrix_multiply(d1, d2)) == 0

    AB = numpy.bmat([[kernel,-d2]])
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

class NumpyEncoderRingTemplate(NumpyEncoder):
    def __init__(self, ring, numpy_dtype):
        self.ring = ring
        self.numpy_dtype = numpy_dtype

    def sage_matrix_from_numpy(self,A):
        return matrix(self.ring, A)

    def sage_vector_from_numpy(self,A):
        v = vector(A,self.ring)
        assert v.base_ring() == self.ring
        return v

    def sage_vector_to_numpy(self, a):
        return numpy.array(a.numpy(dtype=self.numpy_dtype).flatten())

    def sage_matrix_to_numpy(self, A):
        return numpy.array(A.numpy(dtype=self.numpy_dtype))

    def numpy_eye(self, n):
        return numpy.eye(n, dtype=self.numpy_dtype)

    def numpy_zeros(self, shape):
        return numpy.zeros(shape, dtype=self.numpy_dtype)

class NumpyEncoderZ(NumpyEncoderRingTemplate):
    def __init__(self):
        super(NumpyEncoderZ,self).__init__(ZZ, int)

    def numpy_matrix_multiply(self, A,B):
        return numpy.dot(A, B)

    def sage_matrix_from_numpy(self,A):
        if sparse.issparse(A):
            A = A.toarray()

        nrows,ncols = A.shape
        B = matrix(self.ring, nrows, ncols)
        B.set_unsafe_from_numpy_int_array(A)
        return B

class NumpyEncoderZN(NumpyEncoderRingTemplate):
    def __init__(self, n):
        self.n = n
        super(NumpyEncoderZN,self).__init__(Integers(n), int)

    #def sage_matrix_from_numpy(self,A):
    #    return matrix(self.field, A)

    def sage_matrix_from_numpy(self, A):
        if sparse.issparse(A):
            A = A.toarray()

        nrows,ncols = A.shape
        B = matrix(self.ring, nrows, ncols)
        B.set_unsafe_from_numpy_int_array(A % self.n)
        return B

    def numpy_matrix_multiply(self, A,B):
        return numpy.dot(A, B) % self.n

class NumpyEncoderZ2(NumpyEncoderZN):
    def __init__(self):
        super(NumpyEncoderZ2,self).__init__(2)

def get_numpy_encoder_Zn(n):
    if n == 2:
        return NumpyEncoderZ2()
    elif n > 2:
        return NumpyEncoderZN(n)

def column_space_matrix(B,encoder):
    return B[:,encoder.sage_matrix_from_numpy(B).pivots()]

def column_space_intersection_with_other_space(B, other_space_basis_matrix, encoder):
    W = numpy.bmat([[other_space_basis_matrix, -column_space_matrix(B,encoder)]])
    Wsage = encoder.sage_matrix_from_numpy(W)
    kernel_matrix = Wsage.right_kernel_matrix(basis='computed')

    ret = [ encoder.numpy_matrix_multiply(
                other_space_basis_matrix,
                encoder.sage_matrix_to_numpy(kernel_matrix.transpose()[0:other_space_basis_matrix.shape[1],i]).flatten(),
           ) for i in xrange(kernel_matrix.nrows()) ]

    return ret

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
    if A.dtype != B.dtype:
        raise TypeError

    #AB = numpy.empty( (A.shape[0] + B.shape[0], A.shape[1]), A.dtype)
    #AB[0:A.shape[0],:] = A
    #AB[A.shape[0]:,:] = B

    #a = encoder.numpy_eye(A.shape[0])
    #b = encoder.numpy_zeros( (B.shape[0],A.shape[0]) )
    #target_space_basis = numpy.array(numpy.bmat([[a],
    #                                             [b]]))
    #return [ v[0:A.shape[0]] for v in column_space_intersection_with_other_space(AB, target_space_basis,
    #            encoder) ]

    K = right_kernel_matrix(B, encoder)
    AK = encoder.numpy_matrix_multiply(A,K)
    C = column_space_matrix(AK, encoder)
    #scipy.io.savemat('K.mat', {'B': B, 'K': K, 'C': C})
    return [C[:,i].flat for i in xrange(C.shape[1])]

