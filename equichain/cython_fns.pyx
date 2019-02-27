#cython: boundscheck=False,profile=True
import equichain
import numpy
import sys
from scipy import sparse
cimport numpy as np
cimport cython
import numpy as np
from cpython.object cimport Py_EQ,Py_NE
from libc.stdlib cimport calloc,free
from equichain import linalg

from sage.all import ZZ

cdef enum:
    strideprime = 997

cdef class Vector:
    cdef np.int_t* coords
    cdef int n

    def __cinit__(self, int n):
        self.n = n
        self.coords = <np.int_t*>calloc(n,sizeof(np.int_t))
        if self.coords == NULL:
            raise MemoryError

    def __dealloc__(self):
        free(self.coords)

    cdef void copy_from_numpy(self, v):
        cdef np.int_t [:] view = np.ascontiguousarray(v)
        cdef int i
        for i in xrange(self.n):
            self.coords[i] = view[i]

    cdef Vector add(Vector x, Vector y):
        cdef Vector ret = Vector(x.n)
        cdef int i

        for i in xrange(x.n):
            ret.coords[i] = x.coords[i] + y.coords[i]

        return ret

    cdef Vector mul(Vector x, np.int_t n):
        cdef Vector ret = Vector(x.n)

        for i in xrange(x.n):
            ret.coords[i] = x.coords[i]*n

        return ret

    @staticmethod
    cdef Vector from_numpy(np.ndarray v):
        vv = Vector(len(v))
        vv.copy_from_numpy(v)
        return vv

    cdef to_numpy(self):
        arr = np.empty(self.n, dtype=int)
        cdef np.int_t [:] view = arr
        cdef int i
        for i in xrange(self.n):
            view[i] = self.coords[i]
        return arr

    cdef np.int_t hashfn(self):
        cdef np.int_t stride = 1
        cdef np.int_t tot = 0
        cdef int i

        for i in xrange(self.n):
            tot += stride*self.coords[i]
            stride *= strideprime

        return tot

    cdef bint equals(self, Vector y):
        cdef int i

        if self.n != y.n:
            return False

        for i in xrange(self.n):
            if self.coords[i] != y.coords[i]:
                return False
        return True

    cdef bint richcmp(self,
            Vector y, int cmptype):
        if cmptype == Py_EQ:
            return self.equals(y)
        elif cmptype == Py_NE:
            return not self.equals(y)
        else:
            raise NotImplementedError

cdef class IntegerPointInUniverse:
    cdef Vector _coords
    cdef public object universe
    #cdef public object coords

    @property
    def coords(self):
        return self._coords.to_numpy()

    def __init__(self, universe, coords):
        if isinstance(universe, equichain.CubicUniverseWithBoundary):
            universe = None

        self.universe = universe
        if universe is not None:
            coords = universe.canonicalize_coords_int(coords)
        ret = Vector.from_numpy(coords)
        self._coords = ret

    @staticmethod
    cdef IntegerPointInUniverse from_vector(object universe, Vector vector):
        cdef IntegerPointInUniverse ret

        if universe is not None:
            return IntegerPointInUniverse(universe, vector.to_numpy())
        else:
            ret = IntegerPointInUniverse.__new__(IntegerPointInUniverse)
            ret.universe = None
            ret._coords = vector
            return ret

    def __richcmp__(IntegerPointInUniverse x not None,
            IntegerPointInUniverse y not None, int cmptype):
        return x._coords.richcmp(y._coords,cmptype)

    def __hash__(self):
        return self._coords.hashfn()

    def __str__(self):
        return str(self.coords)

    def __repr__(self):
        return str(self)

    def __len__(self):
        return self._coords.n

    def act_with(self, action):
        cdef IntegerPointInUniverseTranslationAction cast_action

        if isinstance(action, IntegerPointInUniverseTranslationAction):
            cast_action = action
            return IntegerPointInUniverse.from_vector(self.universe,
                    self._coords.add(cast_action.trans))

        if isinstance(action, equichain.MatrixQuotientGroupElement):
            action = action.as_matrix_representative_numpy_int()

        if isinstance(action, numpy.matrix):
            action = numpy.array(action)

        v = numpy.empty(len(self.coords)+1, dtype=int)
        v[0:-1] = self.coords
        v[-1] = 1
        vout = numpy.dot(action,v)
        ret = IntegerPointInUniverse(self.universe, vout[:-1])

        return ret

cdef class IntegerPointInUniverseTranslationAction:
    cdef Vector trans
    
    def __init__(self, tr):
        self.trans = Vector.from_numpy(tr)

    def __mul__(IntegerPointInUniverseTranslationAction x not None,
            IntegerPointInUniverseTranslationAction y not None):
        cdef IntegerPointInUniverseTranslationAction ret
        ret = IntegerPointInUniverseTranslationAction.__new__(IntegerPointInUniverseTranslationAction)
        ret.trans = x.trans.add(y.trans)
        return ret

    def __pow__(IntegerPointInUniverseTranslationAction self not None, int n, object z):
        cdef IntegerPointInUniverseTranslationAction ret
        ret = IntegerPointInUniverseTranslationAction.__new__(IntegerPointInUniverseTranslationAction)
        ret.trans = self.trans.mul(n)
        return ret

    @staticmethod
    def get_translation_basis(d, scale=1):
        ret = []
        for i in xrange(d):
            trans = numpy.zeros(d, dtype=int)
            trans[i] = scale
            ret.append(IntegerPointInUniverseTranslationAction(trans))
        return ret

cdef int build_index_out(int n, int ncells, int size_of_group, int ci_out, np.int_t* gi_out):
    cdef int index
    cdef int stride
    cdef int i

    index = 0
    stride = 1
    for i in xrange(n+1):
        index += stride*gi_out[i]
        stride *= size_of_group

    index += stride*ci_out
    stride *= ncells

    return index

cdef int build_index_in(int n, int ncells, int size_of_group, int ci_in, np.int_t* gi_in):
    cdef int index
    cdef int stride
    cdef int i

    index=0
    stride=1

    for i in xrange(n):
        index += stride*gi_in[i]
        stride *= size_of_group

    index += stride*ci_in
    stride *= ncells

    return index


def get_group_coboundary_matrix(cells, int n, G, twist):
    #A = sparse.dok_matrix((indexer_out.total_dim(), indexer_in.total_dim()), dtype=int)
    #A = matrix(base_ring, indexer_out.total_dim(), indexer_in.total_dim(), sparse=True)

    mapped_cell_indices, mapping_parities = equichain.get_group_action_on_cells(cells,G,twist=twist,inverse=True)

    cdef np.int_t [:,:] mapped_cell_indices_view = mapped_cell_indices
    cdef np.int_t [:,:] mapping_parities_view = mapping_parities

    cdef int ncells = len(cells)
    cdef int size_of_group = G.size()

    gi_base = numpy.zeros((n+1,), dtype=int)
    cdef np.int_t [:] gi = gi_base

    temp_gi_base = numpy.zeros(n,dtype=int)
    cdef np.int_t [:] temp_gi = temp_gi_base

    cdef int coo_nentries = ncells*size_of_group**(n+1)*(n+2)
    coo_entries = numpy.zeros(coo_nentries,dtype=int)
    coo_i = numpy.zeros(coo_nentries, dtype=int)
    coo_j = numpy.zeros(coo_nentries, dtype=int)

    times_tables = numpy.empty( (size_of_group,size_of_group), dtype=int)

    cdef np.int_t [:] coo_i_view = coo_i
    cdef np.int_t [:] coo_j_view = coo_j
    cdef np.int_t [:] coo_entries_view = coo_entries
    cdef np.int_t [:,:] times_tables_view = times_tables

    cdef int coo_entry_index = 0

    cdef int ci
    cdef int ii
    cdef int i
    cdef int acted_ci,parity

    cdef bint incremented

    for gi1 in xrange(size_of_group):
        for gi2 in xrange(size_of_group):
            times_tables[gi1, gi2] = (G.element_by_index(gi1)*G.element_by_index(gi2)).toindex()

    #print >>sys.stderr, "Finished precomputing times tables."

    ## BEGIN NATIVE BLOCK
    for ci in xrange(ncells):
        for ii in xrange(n+1):
            gi[ii]=0

        while True:

            ## BEGIN NON-NONNATIVE SUBBLOCK
            g = [ G.element_by_index(gii) for gii in gi ]
            ## END NON-NONNATIVE SUBBLOCK

            acted_ci = mapped_cell_indices_view[gi[0],ci]
            parity = mapping_parities_view[gi[0],ci]

            coo_entries_view[coo_entry_index] = parity
            coo_i_view[coo_entry_index] = build_index_out(n,ncells,size_of_group,ci,&gi[0])
            coo_j_view[coo_entry_index] = build_index_in(n,ncells,size_of_group,acted_ci, 
                    &gi[1] if n > 0 else NULL # Stop cython from complaining about an out of bounds error if n=0
                    )
            coo_entry_index += 1

            coo_entries_view[coo_entry_index] = (-1)**(n+1)
            coo_i_view[coo_entry_index] = build_index_out(n,ncells,size_of_group,ci,&gi[0])
            coo_j_view[coo_entry_index] = build_index_in(n,ncells,size_of_group,ci,&gi[0])
            coo_entry_index += 1

            for i in xrange(1,n+1):
                for ii in xrange(i-1):
                    temp_gi[ii] = gi[ii]
                for ii in xrange(i+1,n+1):
                    temp_gi[ii-1] = gi[ii]

                temp_gi[i-1] = times_tables_view[gi[i-1],gi[i]]
                othervalue = (g[i-1]*g[i]).toindex()
                assert othervalue == temp_gi[i-1]

                coo_entries_view[coo_entry_index] = (-1)**i
                coo_i_view[coo_entry_index] = build_index_out(n,ncells,size_of_group,ci,&gi[0])
                coo_j_view[coo_entry_index] = build_index_in(n,ncells,size_of_group,ci,&temp_gi[0])
                coo_entry_index += 1

            # Code to iterate over product of [0..size_of_group] (n+1) time
            incremented=False
            for ii in xrange(n+1):
                gi[ii] += 1
                if gi[ii] < size_of_group:
                    incremented=True
                    break
                else:
                    gi[ii] = 0
            if not incremented:
                break

    ## END NATIVE BLOCK

    indexer_out = equichain.MultiIndexer(*( (G.size(),) * (n+1) + (len(cells),) ))
    indexer_in = equichain.MultiIndexer(*( (G.size(),) * n + (len(cells),) ))

    A = sparse.coo_matrix((coo_entries,(coo_i,coo_j)), (indexer_out.total_dim(),
        indexer_in.total_dim()), dtype=int)
    return linalg.ScipySparseMatrixOverRing(A.tocsc(), ring=ZZ)
