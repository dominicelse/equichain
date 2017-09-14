import copy
import numpy
from scipy import sparse
import itertools

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s)+1))

def selection_sort_with_parity(l):
    l = list(l)
    parity = 1
    for i in xrange(len(l)):
        index_smallest=i
        for j in xrange(i+1,len(l)):
            if l[j] < l[index_smallest]:
                index_smallest = j
            elif l[j] == l[index_smallest]:
                raise ValueError, "Two elements of list are identical."
        if index_smallest != i:
            l[i], l[index_smallest] = l[index_smallest], l[i]
            parity *= -1
    return l,parity

class ElementWiseArray(tuple):
    def __new__(cls, a):
        return super(ElementWiseArray,cls).__new__(cls,a)

    def __mod__(self, other):
        assert len(self) == len(other)
        return ElementWiseArray([x % y for (x,y) in itertools.izip(self,other)])
    def __add__(self, other):
        assert len(self) == len(other)
        return ElementWiseArray([x + y for (x,y) in itertools.izip(self,other)])

class FormalIntegerSum(object):
    def __init__(self,coeffs={}):
        if not isinstance(coeffs,dict):
            self.coeffs = { coeffs : 1 }
        else:
            self.coeffs = copy.copy(coeffs)

    def __add__(a,b):
        ret = FormalIntegerSum(a.coeffs)
        for o,coeff in b.coeffs.iteritems():
            if o in ret.coeffs:
                ret.coeffs[o] += coeff
            else:
                ret.coeffs[o] = coeff
        return ret

    def __iter__(self):
        return self.coeffs.iteritems()

    def act_with(self,action):
        ret = FormalIntegerSum({})
        for o,coeff in self.coeffs.iteritems():
            ret[o.act_with(action)] = coeff
        return ret

    def __str__(self):
        if len(self.coeffs) == 0:
            return "0"
        else:
            s = ""
            items = self.coeffs.items()
            for i in xrange(len(items)):
                s += str(items[i][1]) + "*" + str(items[i][0])
                if i < len(items)-1:
                    s += " + "
            return s

    def __repr__(self):
        return str(self)

class MultiIndexer(object):
    def __init__(self, *dims):
        self.dims = tuple(dims)

    @staticmethod
    def tensor(dim, ntimes):
        return MultiIndexer(*( (dim,)*ntimes ))

    def to_index(self, *indices):
        index = 0
        stride = 1
        for i in xrange(len(indices)):
            index += stride*indices[i]
            stride *= self.dims[i]

        return index

    def from_index(self,I):
        assert I >= 0 and I < self.total_dim()
        I0 = I
        ret = numpy.zeros(len(self.dims), dtype=int)
        for i in xrange(len(self.dims)):
            ret[i] = I % self.dims[i]
            I = (I - ret[i])//self.dims[i]
        assert self.to_index(*ret) == I0
        return tuple(ret)

    def __call__(self, *indices):
        return self.to_index(*indices)

    def total_dim(self):
        return numpy.prod(self.dims)

class MatrixIndexingWrapper(object):
    def __init__(self, A, out_indexer, in_indexer):
        assert len(A.shape) == 2
        assert A.shape[0] == out_indexer.total_dim()
        assert A.shape[1] == in_indexer.total_dim()

        self.A = A
        self.out_indexer = out_indexer
        self.in_indexer = in_indexer

    @staticmethod
    def from_factory(factory, dtype, out_indexer, in_indexer):
        A = factory( (out_indexer.total_dim(), in_indexer.total_dim()), dtype=dtype)
        return MatrixIndexingWrapper(A, out_indexer, in_indexer)

    def _convert_index(self, i):
        return (self.out_indexer.to_index(*i[0]), self.in_indexer.to_index(*i[1]))

    def __getitem__(self,i):
        #print i, self._convert_index(i), self.A.shape
        return self.A[self._convert_index(i)]

    def __setitem__(self, i, value):
        self.A[self._convert_index(i)] = value

    def raw_access(self):
        return self.A

    def set_raw(self,A):
        self.A = A
