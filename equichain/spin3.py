from __future__ import division
import numpy
import numpy.linalg
import scipy.linalg
import equichain.linalg

from sage.all import *
from sage.interfaces.gap import GapElement

eps = 1e-7

class AntiUnitaryOperator(object):
    # Represents an anti-unitary operation of the form self.mat * K^self.n, 
    # where self.n = 0 or 1 and K is the complex conjugation operator

    def __init__(self, mat, n):
        self.mat = mat
        self.n = n

    def __mul__(a, b):
        if a.n % 2:
            matprod = numpy.dot(a.mat, numpy.conj(b.mat))
        else:
            matprod = numpy.dot(a.mat, b.mat)

        return AntiUnitaryOperator(matprod, (a.n + b.n) % 2)

    def __repr__(self):
        if self.n % 2:
            return repr(self.mat) + "*K"
        else:
            return repr(self.mat)

def convert_to_numpy(x):
    if isinstance(x, GapElement):
        return matrix(x.sage()).numpy()
    elif isinstance(x, equichain.linalg.GenericMatrix):
        return x.to_numpydense().A
    else:
        return x

def spin_cocycle(g1,g2, additive=False):
    g1,g2 = (convert_to_numpy(g) for g in (g1,g2)) 
    g1g2 = g1.dot(g2)

    Ug1,Ug2,Ug1g2 = (preimage_of_O3_element_in_spinhalfrep(g) for g in (g1,g2,g1g2))

    if numpy.linalg.norm( (Ug1*Ug2).mat - Ug1g2.mat) < eps:
        return 0 if additive else 1
    elif numpy.linalg.norm( (Ug1*Ug2).mat + Ug1g2.mat ) < eps:
        return Integer(1)/2 if additive else -1
    else:
        assert False

def spin_3cocycle(g1,g2,g3):
    g1,g2,g3 = (convert_to_numpy(g) for g in (g1,g2,g3)) 

    p1 = int(numpy.linalg.det(g1))

    w = lambda a,b: spin_cocycle(a,b,additive=True)

    ret = p1*w(g2,g3) - w(g1.dot(g2), g3) + w(g1, g2.dot(g3)) - w(g1,g2)
    assert ret in ZZ
    return Integer(ret)


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

def levi_civita_tensor(d):
    Sd = SymmetricGroup(d)
    A = numpy.zeros( [d]*d, dtype=int)
    for g in Sd:
        A[tuple(i-1 for i in g.tuple())] = g.sign()
    return A

class PreimageOfSO3ElementInSU2Cacher(object):
    def __init__(self):
        self.cached_values = {}

    def _compute(self, A):
        # Check A is in SO(3)
        assert A.shape == (3,3)
        assert numpy.linalg.norm(numpy.imag(A)) < eps
        assert numpy.abs(numpy.linalg.det(A)-1) < eps
        assert numpy.linalg.norm(numpy.dot(numpy.transpose(A),A) - numpy.eye(3)) < eps

        if numpy.linalg.norm(A - numpy.eye(3)) < eps:
            return numpy.eye(2)

        # Compute the matrix logarithm of A. The problem is that matrix logarithm is not unique
        # and we want a particular one [the one that is real and anti-symmetric].

        w, vl = scipy.linalg.eig(A)

        one_eigindices = numpy.nonzero(numpy.abs(w-1) < eps)[0]
        assert len(one_eigindices) == 1
        one_eigindex = one_eigindices[0]
        other_eigindices = list(set([0,1,2]) - set([one_eigindex]))
        assert len(other_eigindices) == 2

        assert numpy.abs(w[other_eigindices[0]] * w[other_eigindices[1]]-1) < eps

        theta = -1j*numpy.log(w[other_eigindices[0]])
        assert numpy.abs(numpy.imag(theta)) < eps

        axis = vl[:,one_eigindex]
        assert numpy.linalg.norm(numpy.imag(axis)) < eps

        levi_civita = levi_civita_tensor(3)

        spin1_lie_basis = [None]*3
        for i in xrange(3):
            spin1_lie_basis[i] = levi_civita[i,:,:]

        Alog = sum(axis[i]*theta*spin1_lie_basis[i] for i in xrange(len(spin1_lie_basis)))
        if numpy.linalg.norm(scipy.linalg.expm(Alog) - A) > eps:
            theta = -theta
            Alog = -Alog
            assert numpy.linalg.norm(scipy.linalg.expm(Alog)-A) < eps

        spinhalf_lie_basis = [
            0.5*numpy.array([[0,1],[1,0]]),
            -0.5j*numpy.array([[0,1],[-1,0]]),
            0.5*numpy.array([[1,0],[0,-1]])
            ]

        Alog_su2 = theta*sum(axis[i]*spinhalf_lie_basis[i] for i in xrange(len(spinhalf_lie_basis)))
        return scipy.linalg.expm(1j*Alog_su2)

    def __call__(self, A):
        data = str(A.data)
        try:
            return self.cached_values[data]
        except KeyError:
            ret = self._compute(A)
            self.cached_values[data] = ret
            return ret

preimage_of_SO3_element_in_SU2 = PreimageOfSO3ElementInSU2Cacher()
