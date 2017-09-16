from scipy import sparse
import chaincplx.utils as utils
import itertools
import functools
from sage.all import *

class FreeModuleSyllable(object):
    def __init__(self, m, g):
        self.m = m
        self.g = g

class FreeModuleWord(object):
    def __init__(self, syllables):
        pass

class FreeZGModuleSyllableIndexer(utils.MultiIndexer):
    def __init__(self, g_size, basis_size):
        super(self,FreeZGModuleSyllableIndexer).__init__(g_size, basis_size)

class ZGResolution(object):
    def __init__(self, G):
        self.cached_d_matrices = {}
        self.G = G

    def dual_d_matrix(self,n, ncells, mapped_cell_indices, mapping_parities, raw=True):
        D = utils.MatrixIndexingWrapper.from_factory(sparse.dok_matrix, int,
                out_indexer=utils.MultiIndexer(self.rank(n+1),ncells),
                in_indexer=utils.MultiIndexer(self.rank(n),ncells))

        d_matrix = self.d_matrix(n+1)

        for mu in xrange(ncells):
            for alpha in xrange(self.rank(n+1)):
                for j in d_matrix.raw_access()[:,alpha].nonzero()[0]:
                    g,beta = d_matrix.out_indexer.from_index(j)
                    D[(alpha,mu), 
                      (beta,mapped_cell_indices[g,mu])] += \
                              mapping_parities[g,mu]*d_matrix.raw_access()[j,alpha]
        
        if raw:
            return D.raw_access()
        else:
            return D

    def _compute_d_matrix(self,n):
        d = self._compute_d_matrix_raw(n)
        return utils.MatrixIndexingWrapper(d, 
                utils.MultiIndexer(self.G.size(), self.rank(n-1)),
                utils.MultiIndexer(self.rank(n))
                )

    def d_matrix(self,n):
        if n not in self.cached_d_matrices:
            self.cached_d_matrices[n] = self._compute_d_matrix(n)
        return self.cached_d_matrices[n]

class BarResolution(ZGResolution):
    def rank(self,n):
        return self.G.size()**n

    def _compute_d_matrix_raw(self,n):
        d = utils.MatrixIndexingWrapper.from_factory(sparse.dok_matrix, int,
                out_indexer = utils.MultiIndexer.tensor(self.G.size(), n),
                in_indexer = utils.MultiIndexer.tensor(self.G.size(), n)
                )

        for gi in itertools.product(*( (xrange(self.G.size()),) * n )):
            g = [ self.G.element_by_index(gii) for gii in gi ]

            d[ (gi[0],) + gi[1:], gi ] += 1
            d[ (0,) + gi[0:-1], gi ] += (-1)**n

            for i in xrange(1,n):
                a = (
                      gi[0:(i-1)] + 
                      ((g[i-1]*g[i]).toindex(),) + 
                      gi[(i+1):]
                    )
                d[ (0,) + a, gi ] += (-1)**i

        return d.raw_access()

    def __init__(self, G):
        super(BarResolution,self).__init__(G)
        self.G = G

class HapResolution(ZGResolution):
    def __init__(self,R,G):
        super(HapResolution,self).__init__(G)

        self.R = HapResolutionThinWrapper(R) 
        self.length = self.R.length()

    def rank(self,k):
        if not (k >= 0 and k <= self.length):
            raise IndexError, k
        return self.R.dimension(k)

    def _compute_d_matrix_raw(self,k):
        if not (k >= 1 and k <= self.length):
            raise ValueError, "Bad k", k

        d = utils.MatrixIndexingWrapper.from_factory(sparse.dok_matrix, int,
                out_indexer = utils.MultiIndexer(self.G.size(), self.rank(k-1)),
                in_indexer = utils.MultiIndexer(self.rank(k))
                )

        # precompute the map from "hap index" to the index corresponding to
        # taking g.toindex()
        elts = self.R.elts()
        index_map = [ self.G.element_from_gap(g).toindex() for g in elts ]

        for m in xrange(d.in_indexer.total_dim()):
            acted_m = self.R.boundary(k, m+1)

            for i, gi_hap in acted_m:
                gi = index_map[ int(gi_hap)-1 ]

                assert i != 0
                if i < 0:
                    i = -i
                    coeff = -1
                else:
                    coeff = 1

                d[(gi,i-1), (m,)] += coeff

        return d.raw_access()

class HapResolutionThinWrapper(object):
    def __init__(self,R):
        self.gap_fns = {}
        self.R = R
        for name in "dimension", "boundary", "homotopy", "elts", "group", "properties":
            self.gap_fns[name] = gap("function(R) return R!." + name + "; end")

        self.properties = dict((str(k),v) for k,v in self.gap_fns['properties'](R))

    def dimension(self,k):
        return int(self.gap_fns['dimension'](self.R)(k))

    def boundary(self,k,j):
        return self.gap_fns['boundary'](self.R)(k,j)

    def homotopy(self, k,ig):
        return self.gap_fns['homotopy'](self.R)(k,ig)

    def elts(self):
        return self.gap_fns['elts'](self.R)

    def group(self):
        return self.gap_fns['group'](self.R)

    def length(self):
        return int(self.properties['length'])

    def characteristic(self):
        return self.properties['characteristic']
