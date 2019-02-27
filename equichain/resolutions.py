from scipy import sparse
import equichain.utils as utils
import itertools
import functools
import equichain.linalg as linalg
import numpy
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
        D = utils.MatrixIndexingWrapper.from_factory(utils.COOMatrixHelper, int,
                out_indexer=utils.MultiIndexer(self.rank(n+1),ncells),
                in_indexer=utils.MultiIndexer(self.rank(n),ncells))

        d_matrix = self.d_matrix(n+1)

        for mu in xrange(ncells):
            for j,alpha,value in itertools.izip(*sparse.find(d_matrix.raw_access())):
                g,beta = d_matrix.out_indexer.from_index(j)
                D[(alpha,mu), 
                  (beta,mapped_cell_indices[g,mu])] += \
                          mapping_parities[g,mu]*value

        D.set_raw(D.raw_access().coomatrix().tocsr())
        
        if raw:
            return linalg.ScipySparseMatrixOverRing(D.raw_access(), ring=ZZ)
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
        d = utils.MatrixIndexingWrapper.from_factory(utils.COOMatrixHelper, int,
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

        return d.raw_access().coomatrix().tocsc()

    def __init__(self, G):
        super(BarResolution,self).__init__(G)
        self.G = G

class HapResolution(ZGResolution):
    def __init__(self,R,G):
        super(HapResolution,self).__init__(G)

        self.R = HapResolutionThinWrapper(R) 
        self.length = self.R.length()
        self.cached_dimensions = {}

    def rawgap(self):
        return self.R.R

    def cocycle_map(self, n, R, gaphomo, twist):
        A = numpy.zeros( (self.rank(n),R.rank(n)),  dtype=int)

        gapchainmap = GapObjWithProperties(gap.EquivariantChainMap(self.R.R, R.R.R, gaphomo))

        for basis_el in xrange(1,self.rank(n)+1):
            word = [[basis_el,1]]
            mappedword = gapchainmap.mapping(word, n)

            for syllable in mappedword:
                i = basis_el-1
                mapped_basis_el = int(syllable[1])
                if mapped_basis_el < 0:
                    mapped_basis_el = -mapped_basis_el
                    coeff = -1
                else:
                    coeff = 1

                j = mapped_basis_el-1
                h = R.G.element_from_gap(R.R.elts()[syllable[2]])
                coeff *= twist.action_on_Z(h)

                A[i,j] += coeff

        return A

    def rank(self,k):
        if not (k >= 0 and k <= self.length):
            raise IndexError, k
        if k in self.cached_dimensions:
            return self.cached_dimensions[k]
        else:
            dim = self.R.dimension(k)
            self.cached_dimensions[k] = dim
            return dim

    def convert_cocycle_from_bar_resolution(self, n, cocycle_fn, twist):
        Ggap = self.G.gap_quotient_grp

        equiv = GapObjWithProperties(gap.BarResolutionEquivalence(self.R.R))

        dim = self.rank(n)

        assert self.R.elts()[1] == gap.One(Ggap)

        cocycle_out = numpy.empty( dim, dtype=int)

        for i in xrange(1, dim+1):
            word = [[1,i,1]]
            translation = equiv.psi(n, word)
            value = 0
            for word_in_translation in translation:
                word_in_translation = list(word_in_translation)
                coeff = int(word_in_translation[0])

                gap_h = word_in_translation[1]
                h = self.G.element_from_gap(gap_h)

                coeff *= twist.action_on_Z(h)

                gs = word_in_translation[2:]
                gs = [ self.G.element_from_gap(g) for g in gs ]
                value += coeff*cocycle_fn(*gs)
            cocycle_out[i-1] = value

        return cocycle_out

    def _compute_d_matrix_raw(self,k):
        if not (k >= 1 and k <= self.length):
            raise ValueError, "Bad k", k

        d = utils.MatrixIndexingWrapper.from_factory(utils.COOMatrixHelper, int,
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

        return d.raw_access().coomatrix().tocsr()

class GapObjWithProperties(object):
    def __init__(self, gapobj):
        self.gapobj = gapobj

    def __getattr__(self, name):
        gap_property_accessor = gap("function(o) return o!." + name + "; end")
        return gap_property_accessor(self.gapobj)

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
