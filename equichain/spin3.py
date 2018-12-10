from __future__ import division
import numpy
import numpy.linalg
import scipy.linalg

eps = 1e-7

class AntiUnitaryOperator(object):
    # Represents an anti-unitary operation of the form self.mat * K^self.n, 
    # where self.n = 0 or 1 and K is the complex conjugation operator

    def __init__(self, mat, n):
        self.mat = mat
        self.n = n

    def __mul__(a, b):
        if a.n % 2:
            matprod = numpy.dot(a, numpy.conj(b))
        else:
            matprod = numpy.dot(a, b)

        return AntiUnitaryOperator(matprod, (a.n + b.n) % 2)

    def __repr__(self):
        if self.n % 2:
            return repr(self.mat) + "*K"
        else:
            return repr(self.mat)

def preimage_of_O3_element_in_spinhalfrep(A):
    # Returns (U,m) such that the corresponding action on a spin-1/2 is UK^m
    # where K is complex conjugation

    # Check A is in O(3)
    assert A.shape == (3,3)
    assert numpy.linalg.norm(numpy.imag(A)) < eps
    assert numpy.linalg.norm(numpy.dot(numpy.transpose(A),A) - numpy.eye(3)) < eps

    det = numpy.linalg.det(A)

    if numpy.abs(det-1) < eps:
        return AntiUnitaryOperator(preimage_of_SO3_element_in_SU2(A), 0)
    else:
        Aso = -A
        Aso_spinhalf = preimage_of_SO3_element_in_SU2(-A)

        isigmay = numpy.array([[0,1],[-1,0]])
        return AntiUnitaryOperator(numpy.dot(Aso_spinhalf, isigmay),1)

def preimage_of_SO3_element_in_SU2(A):
    # Check A is in SO(3)
    assert A.shape == (3,3)
    assert numpy.linalg.norm(numpy.imag(A)) < eps
    assert numpy.abs(numpy.linalg.det(A)-1) < eps
    assert numpy.linalg.norm(numpy.dot(numpy.transpose(A),A) - numpy.eye(3)) < eps

    # Compute the matrix logarithm of A. The problem is that matrix logarithm is not unique
    # and we want a particular one [the one that is real and anti-symmetric].
    # The reason why taking the log of the eigenvalues works is that the eigenvalues of an SO(3)
    # matrix are always of the form {e^{it}, 0, e^{-it}} for some real t in [-pi,pi]. Because numpy guarantees that
    # the imag part of the log will be in [-pi,pi], it follows that the log of the eigenvalues gives {it,0.-it}.
    # The case where t is close to +/- pi requires special handling.

    w, vl = scipy.linalg.eig(A)

    wlog = numpy.log(w)

    piindices = numpy.nonzero(numpy.abs(numpy.abs(wlog) - numpy.pi) < eps)[0]
    if len(piindices) == 2:
        wlog[piindices[0]] = 1j*numpy.pi
        wlog[piindices[1]] = -1j*numpy.pi
    elif len(piindices) != 0:
        assert False

    Alog = numpy.dot(vl, numpy.dot(numpy.diag(wlog), numpy.linalg.inv(vl)))

    assert numpy.linalg.norm(numpy.real(Alog)) < eps
    assert numpy.linalg.norm(Alog + numpy.conj(numpy.transpose(Alog))) < eps
    assert numpy.linalg.norm(scipy.linalg.expm(Alog) - A) < eps

    spin1_lie_basis = [
            1/numpy.sqrt(2)*numpy.array([[0,1,0],[1,0,1],[0,1,0]]),
            -1j/numpy.sqrt(2)*numpy.array([[0,1,0],[-1,0,1],[0,-1,0]]),
            numpy.array([[1,0,0],[0,0,0],[0,0,-1]])
            ]

    spinhalf_lie_basis = [
        0.5*numpy.array([[0,1],[1,0]]),
        -0.5j*numpy.array([[0,1],[-1,0]]),
        0.5*numpy.array([[1,0],[0,-1]])
        ]

    coeffs = [ numpy.trace(numpy.dot(numpy.transpose(el), Alog)) for el in spin1_lie_basis ] 
    Alog_su2 = sum(coeffs[i]*spinhalf_lie_basis[i] for i in xrange(len(spin1_lie_basis)))
    return scipy.linalg.expm(Alog_su2)
