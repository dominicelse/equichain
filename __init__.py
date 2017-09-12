from __future__ import division
import itertools
import copy
import numpy
import sys
import cProfile
import copy
from scipy import sparse

use_sage=True
if use_sage:
    from sage.all import *

    gap.load_package("Cryst")
    from sage.matrix.matrix_mod2_dense import Matrix_mod2_dense
else:
    ZZ = None

import cython_fns

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s)+1))

class PointInUniverse(object):
    def __init__(self, universe, coords):
        self.universe = universe
        self.coords = copy(vector(QQ, universe.canonicalize_coords(coords)))
        self.coords.set_immutable()

    def __hash__(self):
        return hash(self.coords)

    def __eq__(x,y):
        return x.coords == y.coords

    def __gt__(x,y):
        return x.coords > y.coords

    def __ge__(x,y):
        return x.coords >= y.coords

    def __lt__(x,y):
        return x.coords < y.coords

    def __le__(x,y):
        return x.coords <= y.coords

    def __ne__(x,y):
        return x.coords != y.coords

    def __str__(self):
        return str(self.coords)

    def __repr__(self):
        return str(self)

    def __len__(self):
        return len(self.coords)

    def act_with(self, action):
        if isinstance(action, MatrixQuotientGroupElement):
            action = action.as_matrix_representative()

        v = vector(tuple(self.coords) + (1,))
        vout = action*v
        ret = PointInUniverse(self.universe, vout[:-1])

        return ret

class IntegerPointInUniverse(object):
    def __init__(self, universe, coords):
        self.universe = universe
        self.coords = universe.canonicalize_coords_int(coords)
        self.coords.setflags(write=False)

    def __hash__(self):
        return hash(self.coords.data)

    def __eq__(x,y):
        return numpy.array_equal(x.coords,y.coords)

    def __ne__(x,y):
        return not x == y

    def __str__(self):
        return str(self.coords)

    def __repr__(self):
        return str(self)

    def __len__(self):
        return len(self.coords)

    def act_with(self, action):
        if isinstance(action, MatrixQuotientGroupElement):
            action = action.as_matrix_representative_numpy_int()

        v = numpy.empty(len(self.coords)+1, dtype=int)
        v[0:-1] = self.coords
        v[-1] = 1
        vout = numpy.dot(action,v)
        ret = IntegerPointInUniverse(self.universe, vout[:-1])

        return ret

class Universe(object):
    def cell_on_boundary(self, cell):
        return all(not self.point_in_interior(pt) for pt in cell)

def fracpart(x):
    return x-floor(x)

class OpenToroidalUniverse(Universe):
    def __init__(self, extents, open_dirs):
        self.extents = extents
        self.open_dirs = open_dirs

    def canonicalize_coords(self, coords):
        coords = list(QQ(c) for c in coords)
        for i in xrange(len(self.extents)):
            if i not in self.open_dirs:
                coords[i] = (fracpart((coords[i]-self.extents[i][0]) / (self.extents[i][1] - self.extents[i][0]))
                        * (self.extents[i][1] - self.extents[i][0])
                        + self.extents[i][0])
        return vector(QQ,coords)

    def canonicalize_coords_int(self, coords):
        assert len(coords) == len(self.extents)
        coords = numpy.array(coords, dtype=int)
        for i in xrange(len(self.extents)):
            if i not in self.open_dirs:
                coords[i] = (coords[i] - self.extents[i][0]) % (self.extents[i][1] - self.extents[i][0]) + self.extents[i][0]
        return coords

    def point_in_interior(self, point):
        for d in xrange(len(self.extents)):
            if d in self.open_dirs:
                if point.coords[d] <= self.extents[d][0] or point.coords[d] >= self.extents[d][1]:
                    return False
        return True

class FlatUniverse(Universe):
    def canonicalize_coords(self, coords):
        return coords

class FormalIntegerSum(object):
    def __init__(self,coeffs={}):
        if not isinstance(coeffs,dict):
            self.coeffs = { coeffs : 1 }
        else:
            self.coeffs = copy(coeffs)

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

class ConvexHullCell(object):
    def __init__(self,points,orientation):
        # The orientation is an anti-symmetric matrix implementing the
        # projected determinant.
        self.points = frozenset(points)
        self._orientation = orientation
        self.original_my_hash = None
        self.original_my_hash = ConvexHullCell.__hash__(self)

    def act_with(self,action):
        ret = ConvexHullCell([p.act_with(action) for p in list(self)],
                orientation=transform_levi_civita(self._orientation,action))

        return ret

    def __eq__(a,b):
        return a.points == b.points

    def __hash__(self):
        h = hash(self.points)
        if self.original_my_hash is not None and h != self.original_my_hash:
            raise RuntimeError, "ConvexHullCell: hash changed"
        return h

    def __iter__(self):
        return iter(self.points)

    def __str__(self):
        return "CELL" + str(list(self.points))

    def __repr__(self):
        return str(self)

    def orientation(self):
        return self._orientation

class ConvexHullCellWithMidpoint(ConvexHullCell):
    def __init__(self, points, orientation, midpoint):
        super(ConvexHullCellWithMidpoint,self).__init__(points,orientation)
        self.midpoint = midpoint

    def act_with(self,action):
        pt = super(ConvexHullCellWithMidpoint,self).act_with(action)
        return ConvexHullCellWithMidpoint(pt.points, pt.orientation(),
                self.midpoint.act_with(action))

    def forget_midpoint(self):
        return ConvexHullCell(self.points,self.orientation)

    def __hash__(self):
        return super(ConvexHullCellWithMidpoint,self).__hash__() ^ hash(self.midpoint)

    def __eq__(self,b):
        return super(ConvexHullCellWithMidpoint,self).__eq__(b) and self.midpoint == b.midpoint

    def __ne__(a,b):
        return not a == b

    def __str__(self):
        return "MIDP" + super(ConvexHullCellWithMidpoint,self).__str__() + ";" + str(self.midpoint)

    def __repr__(self):
        return str(self)


def cubical_cell(ndims, basepoint_coords, direction, universe, with_midpoint,
        scale, pointclass):
    assert len(direction) == ndims

    basepoint_coords = numpy.array(basepoint_coords)
    points = []
    for increment_dirs in powerset(direction):
        coord = numpy.copy(basepoint_coords)
        for d in increment_dirs:
            coord[d] += scale
        points += [ coord ]

    orientation = projected_levi_civita(len(basepoint_coords),direction)

    if with_midpoint:
        if pointclass is IntegerPointInUniverse:
            s = sum(points)
            if not all(s % len(points) == 0):
                raise ValueError, "Midpoint is not an integer point."
            else:
                midpoint = s//len(points)
        else:
            midpoint = vector(QQ, sum(points))/len(points)

        midpoint = pointclass(universe, midpoint)

    points = [ pointclass(universe, coord) for coord in points ]

    if with_midpoint:
        return ConvexHullCellWithMidpoint(points, orientation, midpoint)
    else:
        return ConvexHullCell(points, orientation)

def projected_levi_civita(ndims_universe, directions):
    E = numpy.zeros(shape=(ndims_universe,)*len(directions), dtype=int)

    directions = sorted(directions)
    for permuted_directions in itertools.permutations(directions):
        s,parity = selection_sort_with_parity(permuted_directions)
        assert tuple(s) == tuple(directions)

        E[permuted_directions] = parity

    return E

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

def reduce_projected_levi_civita(E, normal_vector):
    return numpy.tensordot(E, normal_vector, (0,0))

def transform_levi_civita(E, R):
    if isinstance(R,MatrixQuotientGroupElement):
        R = R.as_matrix_representative()
    Rt = numpy.transpose(R.numpy())[0:-1,0:-1]
    for i in xrange(len(E.shape)):
        E = numpy.tensordot(E, Rt, (0,0))
    return E

def get_relative_orientation(orientation1, orientation2):
    if numpy.array_equal(orientation1,orientation2):
        return 1
    elif numpy.array_equal(orientation1,-orientation2):
        return -1
    else:
        raise RuntimeError, "Orientations are not relative."

def cubical_cell_boundary(ndims, basepoint_coords, direction, orientation,
        universe, with_midpoints, scale, pointclass):
    coeffs = {}
    for face_direction in direction:
        normal_vector = numpy.zeros(shape=(len(basepoint_coords),),dtype=int)
        normal_vector[face_direction] = 1
        remaining_directions = [ d for d in direction if d != face_direction ]

        cell = cubical_cell(ndims-1, basepoint_coords, remaining_directions,
                universe, with_midpoints, scale, pointclass)
        if not universe.cell_on_boundary(cell):
            reduced_orientation = reduce_projected_levi_civita(orientation, normal_vector)
            coeffs[cell] = get_relative_orientation(reduced_orientation,
                    cell.orientation())

        coord = numpy.array(basepoint_coords)
        coord[face_direction] += scale
        cell = cubical_cell(ndims-1, coord, remaining_directions, universe,
                with_midpoints, scale, pointclass)
        if not universe.cell_on_boundary(cell):
            reduced_orientation = reduce_projected_levi_civita(orientation, -normal_vector)
            coeffs[cell] = get_relative_orientation(reduced_orientation,
                    cell.orientation())

    return FormalIntegerSum(coeffs)

def _cubical_complex_base(ndims, extents, universe, with_midpoints, scale, pointclass):
    cplx = ConvexComplex(ndims)
    for celldim in xrange(ndims+1):
        for direction in itertools.combinations(xrange(ndims),celldim):
            coord_ranges = [ xrange(extents[i][0], extents[i][1], scale) for i in xrange(ndims) ]
            for coord in itertools.product(*coord_ranges):
                cell = cubical_cell(celldim,coord,direction,
                        universe, with_midpoints, scale, pointclass)
                if not universe.cell_on_boundary(cell):
                    cplx.add_cell(celldim,
                            cell,
                            cubical_cell_boundary(celldim,coord,direction,
                                cell.orientation(), universe,
                                with_midpoints,scale,
                                pointclass)
                            )
    return cplx

def cubical_complex(ndims, sizes, open_dirs, with_midpoints=False, scale=1,
        pointclass=PointInUniverse):
    assert len(sizes) == ndims
    assert all(d >= 0 and d < ndims for d in open_dirs)

    for d in xrange(ndims):
        if sizes[d] <= 1 and not d in open_dirs and not with_midpoints:
            raise ValueError, "Don't use this function to construct a compactified direction of length <= 2 without setting with_midpoints=True, this causes problems."

    extents = [ [-sizes[i]*scale,sizes[i]*scale] for i in xrange(ndims) ]
    universe = OpenToroidalUniverse(extents, open_dirs)
    return _cubical_complex_base(ndims, extents, universe,
            with_midpoints,scale,pointclass)

def get_stabilizer_group(cell,G):
    gs = [g for g in G if cell.act_with(g) == cell]
    try:
        return G.subgroup(gs)
    except AttributeError:
        return gs

class TorusTranslationGroup(object):
    def __init__(self, *dims):
        self.dims = numpy.array(dims, dtype=int)

        self.els = [g for g in self]
        self.els_reverse_lookup = dict()
        for (i,g) in enumerate(self):
            self.els_reverse_lookup[self.els[i]] = i

    def __iter__(self):
        return (TorusTranslationGroupElement(self, i) for i in itertools.product(*( xrange(d) for d in self.dims )))

    def size(self):
        return numpy.prod(self.dims)

    def element_by_index(self,i):
        return self.els[i]

    def element_to_index(self,g):
        return self.els_reverse_lookup[g]

class MatrixQuotientGroupElement(object):
    pass

class TorusTranslationGroupElement(MatrixQuotientGroupElement):
    def __init__(self, G, i):
        assert len(i) == len(G.dims)
        self.G = G
        self.i = numpy.array(i, dtype=int) % G.dims
        self.i.setflags(write=False)

    def __mul__(a,b):
        assert a.G is b.G
        return TorusTranslationGroupElement(a.G, a.i + b.i)

    def __inv__(self):
        return TorusTranslationGroupElement(self.G, -self.i)

    def __eq__(a,b):
        assert a.G is b.G
        return all(a.i == b.i)

    def __ne__(a,b):
        return not a == b

    def __hash__(self):
        return hash(self.i.data)

    def toindex(self):
        return self.G.element_to_index(self)

    def __pow__(self,n):
        return TorusTranslationGroupElement(self.G, -self.i)

    def as_matrix_representative(self):
        ndims = len(self.G.dims)
        A = numpy.eye(ndims + 1, dtype=int)
        A[0:(ndims+1),-1] = self.i
        return matrix(A)

class NotIntegerMatrixError(Exception):
    pass

def sage_matrix_to_numpy_int(A):
    if not A in MatrixSpace(ZZ,A.nrows()):
        print A
        raise NotIntegerMatrixError

    return matrix(ZZ,A).numpy(dtype=int)

def sage_vector_to_numpy_int(v):
    if not v in FreeModule(ZZ,len(v)):
        raise ValueError, "Not an integer vector." + str(v)

    return vector(ZZ,v).numpy(dtype=int)

class GapAffineQuotientGroup(object):
    def _base_init(self):
        self.stored_coset_representatives = dict()
        for g in self.sage_quotient_grp:
            self.stored_coset_representatives[g] = self._coset_representative(g)

        self.els = [g for g in self.sage_quotient_grp]
        self.els_reverse_lookup = dict()
        for i in xrange(len(self.els)):
            self.els_reverse_lookup[self.els[i]] = i

        self.stored_coset_representatives_numpy_int = None

    def identity(self):
        return GapAffineQuotientGroupElement(self, self.sage_quotient_grp.identity())

    def __init__(self, G,N, scale=1):
        self.homo_to_factor = gap.NaturalHomomorphismByNormalSubgroup(G,N)
        quotient_group = gap.ImagesSource(self.homo_to_factor)
        iso_to_perm = gap.IsomorphismPermGroup(quotient_group)
        self.iso_to_perm_inverse = gap.InverseGeneralMapping(iso_to_perm)
        self.gap_quotient_grp = gap.Image(iso_to_perm)
        self.sage_quotient_grp = PermutationGroup(gap_group = self.gap_quotient_grp)
        self.basegrp = self
        self.scale = scale

        self._base_init()

        #self.multiplication_table = numpy.array(shape=(len(self.els),len(self.els)), dtype=int)
        #for i in len(self.els):
        #    for j in len(self.els):

    def subgroup(self, gens):
        G = GapAffineQuotientGroup.__new__(GapAffineQuotientGroup)
        G.sage_quotient_grp = self.sage_quotient_grp.subgroup([g.sageperm for g in gens])
        G.iso_to_perm_inverse = self.iso_to_perm_inverse
        G.homo_to_factor = self.homo_to_factor
        G.basegrp = self.basegrp
        G.scale = self.scale
        #G.basegrp = G
        G._base_init()
        return G

    def __iter__(self):
        return iter(self.elements())

    def gens(self):
        return [GapAffineQuotientGroupElement(self.basegrp,g) for g in self.sage_quotient_grp.gens()]

    def elements(self):
        return [GapAffineQuotientGroupElement(self.basegrp, g) for g in
                self.sage_quotient_grp]

    def element_to_index(self, g):
        return self.els_reverse_lookup[g.sageperm]

    def element_by_index(self,i):
        return GapAffineQuotientGroupElement(self.basegrp,self.els[i])

    def _coset_representative(self,g):
        g = gap(g)
        A = matrix(gap.PreImagesRepresentative(
                self.homo_to_factor,
                gap.Image(self.iso_to_perm_inverse, g)).sage())
        B = affine_transformation_rescale(A,self.scale)
        return B

    def coset_representative(self,g):
        return self.stored_coset_representatives[g.sageperm]

    def _compute_coset_representatives_numpy_int(self):
        self.stored_coset_representatives_numpy_int = dict(
                (g, sage_matrix_to_numpy_int(A))
                for g,A in self.stored_coset_representatives.iteritems()
                )

    def coset_representative_numpy_int(self,g):
        if self.stored_coset_representatives_numpy_int is None:
            self._compute_coset_representatives_numpy_int()
        return self.stored_coset_representatives_numpy_int[g.sageperm]

    def __len__(self):
        return self.size()

    def size(self):
        return len(self.els)

class TrivialAffineMatrixGroupElement(MatrixQuotientGroupElement):
    def __init__(self,d):
        self.d = d

    def __mul__(self, b):
        return self

    def __eq__(a,b):
        return True

    def __ne__(a,b):
        return False

    def __pow__(self, n):
        return self

    def as_matrix_representative(self):
        return matrix.identity(ZZ,self.d+1)

    def as_matrix_representative_numpy_int(self):
        return numpy.eye(self.d+1, dtype=int)

    def toindex(self):
        return 0

class TrivialAffineMatrixGroup(object):
    def __init__(self, d):
        self.d = d

    def identity(self):
        return TrivialAffineMatrixGroupElement(self.d)

    def subgroup(self, gens):
        return self

    def __iter__(self):
        return iter([self.identity()])

    def gens(self):
        return [self.identity()]

    def elements(self):
        return [self.identity()]

    def element_by_index(self,i):
        return self.identity()

    def __len__(self):
        return 1

    def size(self):
        return 1

class ElementWiseArray(tuple):
    def __new__(cls, a):
        return super(ElementWiseArray,cls).__new__(cls,a)

    def __mod__(self, other):
        assert len(self) == len(other)
        return ElementWiseArray([x % y for (x,y) in itertools.izip(self,other)])
    def __add__(self, other):
        assert len(self) == len(other)
        return ElementWiseArray([x + y for (x,y) in itertools.izip(self,other)])


class GapAffineQuotientGroupElement(MatrixQuotientGroupElement):
    def __init__(self, G, sageperm):
        self.G = G
        self.sageperm = sageperm

    def __mul__(a, b):
        assert a.G is b.G
        return GapAffineQuotientGroupElement(a.G, a.sageperm*b.sageperm)

    def __eq__(a,b):
        return a.sageperm == b.sageperm

    def __ne__(a,b):
        return not a == b

    def __pow__(self, n):
        return GapAffineQuotientGroupElement(self.G, self.sageperm**(-1))

    def as_matrix_representative(self):
        return self.G.coset_representative(self)

    def as_matrix_representative_numpy_int(self):
        return self.G.coset_representative_numpy_int(self)

    def toindex(self):
        return self.G.element_to_index(self)

class FiniteAbelianGroup(object):    
    def __init__(self, n):
        self.n = ElementWiseArray(n)

        self.els = [g for g in self]
        self.els_reverse_lookup = dict()
        for (i,g) in enumerate(self):
            self.els_reverse_lookup[self.els[i]] = i

    def element_to_index(self,g):
        return self.els_reverse_lookup[g]

    def element_by_index(self,i):
        return self.els[i]

    def identity(self):
        return FiniteAbelianGroupElement(self)

    def __iter__(self):
        for x in itertools.product(*([range(n) for n in self.n])):
            yield FiniteAbelianGroupElement(self, x)

    def generators(self):
        for i in xrange(len(self.n)):
            k = [0]*len(self.n)
            k[i] = 1
            yield FiniteAbelianGroupElement(self, k)
            
    def el(self, k):
        return FiniteAbelianGroupElement(self, k)

    def size(self):
        return numpy.prod(self.n)

class FiniteAbelianGroupElement(object):
    def __eq__(self, b):
        return self.k == b.k
    def __ne__(self,b):
        return not self == b
    def __hash__(self):
        return hash(self.k)

    def __init__(self, group, k=None):
        if k is not None:
            self.k = ElementWiseArray(k)
        else:
            self.k = ElementWiseArray([0]*len(group.n))
        self.group = group

    def parent(self):
        return self.group

    def toindex(self):
        return self.group.element_to_index(self)

    def __nonzero__(self):
        return any(self.k)

    def __mul__(a, b):
        if a.group is not b.group and a.group != b.group:
            raise ValueError, "Can only multiply elements of the same group."
        return FiniteAbelianGroupElement(a.group, (a.k + b.k) % a.group.n)

    def __repr__(self):
        return str(tuple(self.k))

    def __pow__(self,p):
        return FiniteAbelianGroupElement(self.group, [(self.k[i]*p) % self.group.n[i] for i in xrange(len(self.k))])

def toroidal_space_group(d,n,L):
    G = gap.SpaceGroupIT(d,n)
    trans = gap.translation_subgroup(G,L)
    return GapQuotientGroup(G,trans)

#def solve_matrix_equation(A, b, over_ring=ZZ):
#    b = vector(over_ring,b)
#    A = scipy_sparse_matrix_to_sage(over_ring, A)
#    return A.solve_right(b).numpy(dtype=int)

def solve_matrix_equation_with_constraint(A, subs_indices, xconstr, over_ring=ZZ):
    """ Solves the equation Ax = 0, given the constraint that x[i] = xconstr[i]
    for i in subs_indices. """ 

    subs_indices_list = list(subs_indices)
    subs_indices_set = frozenset(subs_indices)

    notsubs_indices_set = frozenset(xrange(A.shape[1])) - subs_indices_set
    notsubs_indices_list = list(notsubs_indices_set)

    xconstr_reduced = xconstr[subs_indices_list]

    print "slicing"
    A_reduced_1 = A[:,notsubs_indices_list]
    A_reduced_2 = A[:,subs_indices_list]

    b = -A_reduced_2.dot(xconstr_reduced)

    print "starting conversion"
    A_reduced_1 = scipy_sparse_matrix_to_sage(over_ring, A_reduced_1)
    b = vector(over_ring, b)

    print "starting solve"
    sol = A_reduced_1.solve_right(b).numpy()

    sol_full = numpy.empty(A.shape[1], dtype=int)
    sol_full[notsubs_indices_list] = sol
    sol_full[subs_indices_list] = xconstr[subs_indices_list]

    assert(numpy.count_nonzero(A.dot(sol_full) % 2) == 0)
    return sol_full

def scipy_sparse_matrix_to_sage(R, M):
    if len(M.shape) != 2:
        raise ValueError, "Need to start with rank-2 array"

    M = sparse.coo_matrix(M)
    Mdict = dict(((int(M.row[i]), int(M.col[i])), int(M.data[i])) for i in xrange(len(M.data)))

    return matrix(R, M.shape[0], M.shape[1], Mdict)

def get_notequal(G, val):
    assert len(G) == 2
    return (g for g in G if g != val).next()

def get_nonidentity(G):
    return get_notequal(G, G.identity())

class SimplePermutee(object):
    def __init__(self,k):
        self.k = k % 2

    def act_with(self,g):
        assert isinstance(g, FiniteAbelianGroupElement)
        if g.k[0] % 2 != 0:
            return SimplePermutee((self.k + 1) % 2)
        else:
            return SimplePermutee(self.k)

    def __eq__(a,b):
        return a.k % 2 == b.k % 2

    def __ne__(a,b):
        return not a == b

    def __hash__(self):
        return self.k % 2

    def orientation(self):
        return 1

    @staticmethod
    def gen():
        yield SimplePermutee(0)
        yield SimplePermutee(1)

class MultiIndexer(object):
    def __init__(self, *dims):
        self.dims = tuple(dims)

    def to_index(self, *indices):
        index = 0
        stride = 1
        for i in xrange(len(indices)):
            index += stride*indices[i]
            stride *= self.dims[i]

        return index

    def __call__(self, *indices):
        return self.to_index(*indices)

    def total_dim(self):
        return numpy.prod(self.dims)

class ComplexChainIndexer(object):
    def __init__(self, n, cells, G):
        self.internal_indexer = MultiIndexer(*( (G.size(),) * n + (len(cells),) ))

    def to_index(self, gi, ci):
        return self.internal_indexer.to_index(*(tuple(gi) + (ci,)))

    def __call__(self, gi, ci):
        return self.to_index(gi,ci)

    def total_dim(self):
        return self.internal_indexer.total_dim()

def get_group_coboundary_matrix(cells, n,G, use_cython=True):
    if use_cython:
        return cython_fns.get_group_coboundary_matrix(cells,n,G)

    indexer_out = MultiIndexer(*( (G.size(),) * (n+1) + (len(cells),) ))
    indexer_in = MultiIndexer(*( (G.size(),) * n + (len(cells),) ))

    A = sparse.dok_matrix((indexer_out.total_dim(), indexer_in.total_dim()), dtype=int)
    #A = matrix(base_ring, indexer_out.total_dim(), indexer_in.total_dim(), sparse=True)

    mapped_cell_indices, mapping_parities = get_group_action_on_cells(cells,G,inverse=True)

    def build_index(ci_out, gi_out, ci_in, gi_in):
        return indexer_out(*( gi_out + (ci_out,))) , indexer_in(*( gi_in + (ci_in,)))

    for ci in xrange(len(cells)):
        for gi in itertools.product(*( (xrange(G.size()),) * (n+1) )):
            g = [ G.element_by_index(gii) for gii in gi ]
            acted_ci = mapped_cell_indices[gi[0],ci]
            parity = mapping_parities[gi[0],ci]

            A[build_index(ci, gi, acted_ci, gi[1:])] += parity

            A[build_index(ci, gi, ci, gi[0:-1])] += (-1)**(n+1)

            for i in xrange(1,n+1):
                a = (
                      gi[0:(i-1)] + 
                      ((g[i-1]*g[i]).toindex(),) + 
                      gi[(i+1):]
                    )
                A[build_index(ci, gi, ci, a)] += (-1)**i

    return A.tocsc()

class ComplexNotInvariantError(Exception):
    pass

def get_action_on_cells(cells,action):
    mapped_cell_indices = numpy.empty( len(cells), dtype=int)
    mapping_parities = numpy.empty( len(cells), dtype=int)

    for i in xrange(len(cells)):
        acted_cell = cells[i].act_with(action)
        try:
            acted_ci = cells.index(acted_cell)
        except ValueError:
            raise ComplexNotInvariantError
        parity = get_relative_orientation(cells[acted_ci].orientation(),
                cells[acted_ci].orientation())

        mapped_cell_indices[i] = acted_ci
        mapping_parities[i] = parity
        
    return mapped_cell_indices, mapping_parities

def get_group_action_on_cells(cells, G, inverse=False):
    mapped_cell_indices = numpy.empty( (G.size(), len(cells)), dtype=int)
    mapping_parities = numpy.empty( (G.size(), len(cells)), dtype=int)
    
    for (i,g) in enumerate(G):
        mapped_cell_indices_g, mapping_parities_g = get_action_on_cells(cells,
                g**(-1) if inverse else g)
        mapped_cell_indices[i,:] = mapped_cell_indices_g
        mapping_parities[i,:] = mapping_parities_g

    return mapped_cell_indices, mapping_parities

class ConvexComplex(object):
    @staticmethod
    def _get_action_matrix(cells,action):
        cells = list(cells)
        A = numpy.zeros( (len(cells), len(cells)), dtype=int)
        for i in xrange(len(cells)):
            j = ConvexComplex._get_action_on_cell_index(cells,i,action)
            A[j,i] = get_relative_orientation(acted_cell.orientation(), cells[j].orientation())
        return A

    #def get_first_group_coboundary_matrix(self, G, k):
    #    cells = self.cells[k]

    #    A = numpy.zeros( (G.size(), len(cells), len(cells)), dtype=int)

    #    for ci in xrange(len(cells)):
    #        for g in G:
    #            gi = g.toindex()

    #            acted_cell = cells[ci].act_with( (g**(-1)).as_matrix_representative() )
    #            acted_ci = cells.index(acted_cell)
    #            parity = get_relative_orientation(acted_cell.orientation, cells[acted_ci].orientation) 

    #            A[gi,ci,acted_ci] += parity
    #            A[gi,ci,ci] += -1

    #    return numpy.reshape(A, (G.size()*len(cells), len(cells)))
    
    def _get_action_on_cell_index(self,cells,action,i):
        acted_cell = cells[i].act_with(action)
        return cells.index(acted_cell)

    def get_group_coboundary_matrix(self, n,G,k, use_cython=True):
        return get_group_coboundary_matrix(self.cells[k],n,G, use_cython)

    def get_action_matrix(self, k, action):
        return ConvexComplex._get_action_matrix(self.cells[k], action)

    def get_group_orbits(self, k, G):
        cells = self.cells[k]
        cell_indices = set(xrange(len(cells)))

        orbits = []

        while len(cell_indices) > 0:
            orbit = set()

            i = iter(cell_indices).next()

            for g in G:
                j = self._get_action_on_cell_index(cells,g,i)
                orbit.add(j)
                cell_indices.discard(j)

            orbits.append(list(orbit))

        return orbits

    def get_boundary_matrix(self, k):
        cells_km1 = list(self.cells[k-1])
        cells_k = self.cells[k]

        A = sparse.dok_matrix( (len(cells_km1), len(cells_k)), dtype=int )
        for i in xrange(len(cells_k)):
            for boundary_cell,coeff in self.boundary_data[cells_k[i]]:
                j = cells_km1.index(boundary_cell)
                A[j,i] += coeff
        return A

    # THIS METHOD DEPRECATED
    def get_cochain_indexer_manual(self,k,n,G):
        return MultiIndexer(*( (G.size(),) * n + (len(self.cells[k]),) ))

    def get_chain_indexer(self,k,n,G):
        return ComplexChainIndexer(n=n,G=G,cells=self.cells[k])

    def get_boundary_matrix_group_cochain(self, k,n,G):
        A = self.get_boundary_matrix(k)
        return sparse.kron(A, sparse.eye(G.size()**n,dtype=int))

    #def get_boundary_matrix_group_cochain_2(self, k,n,G):
    #    A = self.get_boundary_matrix(k)
    #    indexer_in = self.get_cochain_indexer_manual(k=k,n=n,G=G)
    #    indexer_out = self.get_cochain_indexer_manual(k=(k-1), n=n, G=G)
    #    AG = numpy.zeros((indexer_out.total_dim(),
    #        indexer_in.total_dim()), dtype=int)

    #    for gi in xrange(G.size()):
    #        for ci_in in xrange(len(self.cells[k])):
    #            for ci_out in xrange(len(self.cells[k-1])):
    #                AG[indexer_out(gi, ci_out), indexer_in(gi,ci_in)] = A[ci_out,ci_in]

    #    return AG

    def add_cell(self, ndim, cell, boundary):
        self.cells[ndim] += [cell]
        self.boundary_data[cell] = boundary

    def __init__(self,ndims):
        self.ndims = ndims
        self.cells = [None]*(ndims+1)
        self.boundary_data = {}
        for i in xrange(ndims+1):
            self.cells[i] = []

    def get_group_action_on_cells(self, G, k, inverse=False):
        return get_group_action_on_cells(self.cells[k], G, inverse)

def test_has_solution(fn):
    try:
        fn()
    except ValueError as e:
        if e.args[0] == "matrix equation has no solutions":
            return False
        else:
            raise e
    return True

class NumpyEncoder(object):
    def solve_matrix_equation(self,A,b):
        return self.sage_vector_to_numpy(
                self.sage_matrix_from_numpy(A).solve_right(
                    self.sage_vector_from_numpy(b) ) )

class NumpyEncoderZN(NumpyEncoder):
    def __init__(self, n):
        self.field = Integers(n)
        if not is_field(self.field):
            raise ValueError, "Not sure if the code logic is correct if we're working in a field."
        self.n = n

    def sage_vector_from_numpy(self,a):
        v = vector(a,self.field)
        assert v.base_ring() == self.field
        return v

    #def sage_matrix_from_numpy(self,A):
    #    return matrix(self.field, A)

    def sage_matrix_from_numpy(self, A):
        if sparse.issparse(A):
            A = A.toarray()

        nrows,ncols = A.shape
        B = matrix(self.field, nrows, ncols)
        B.set_unsafe_from_numpy_int_array(A % self.n)
        return B

    def sage_vector_to_numpy(self, a):
        return numpy.array(a.numpy(dtype=int).flatten())

    def numpy_matrix_multiply(self, A,B):
        return numpy.dot(A, B) % self.n

    def numpy_eye(self, n):
        return numpy.eye(n, dtype=int)

    def numpy_zeros(self, shape):
        return numpy.zeros(shape, dtype=int)

    def sage_matrix_to_numpy(self, A):
        return numpy.array(A.numpy(dtype=int))

class NumpyEncoderZ2(NumpyEncoderZN):
    def __init__(self):
        super(NumpyEncoderZ2,self).__init__(2)


def get_numpy_encoder_Zn(n):
    if n == 2:
        return NumpyEncoderZ2()
    elif n > 2:
        return NumpyEncoderZN(n)

def column_space_intersection_with_other_space(B, other_space_basis_matrix, encoder):
    Bsage = encoder.sage_matrix_from_numpy(B)

    W = numpy.bmat([[other_space_basis_matrix, -B[:,Bsage.pivots()]]])
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

def trivialized_by_E3_space(cplx,n,k,G,encoder,method='column_space_dense'):
    d1 = cplx.get_boundary_matrix_group_cochain(n=n,k=(k+1),G=G)
    d2 = cplx.get_boundary_matrix_group_cochain(n=(n+1), k=(k+2), G=G)
    delta1 = cplx.get_group_coboundary_matrix(n=n, k=(k+1), G=G)
    delta2 = cplx.get_group_coboundary_matrix(n=(n+1), k=(k+2), G=G)

    indexer = cplx.get_chain_indexer(n=n,k=k,G=G)

    B = sparse.bmat([[d1,   None],
                     [None, delta2],
                     [delta1, -d2]])
    B = B.toarray()

    a = encoder.numpy_eye(indexer.total_dim())
    b = encoder.numpy_zeros((B.shape[0]-indexer.total_dim(), indexer.total_dim()))
    target_space_basis = numpy.array(numpy.bmat([[a],[b]]))
    return [ v[0:indexer.total_dim()] for v in column_space_intersection_with_other_space(B, target_space_basis, encoder) ]

def trivialized_by_E3_but_not_E2(cplx,n,k,G,encoder):
    triv_by_E3 = trivialized_by_E3_space(cplx,n,k,G,encoder)

    ret = []
    
    for v in triv_by_E3:
        if not test_has_solution(lambda: find_E2_trivializer(cplx,v,n,k,G,encoder)):
            ret.append(v)
    return ret

#def trivialized_by_E3_space(cplx,n,k,G,field, method='column_space_dense', use_z2_optimization=True):
#    if use_z2_optimization and field in (GF(2), Integers(2)) and method == 'column_space_dense':
#        return trivialized_by_E3_space_Z2(cplx,n,k,G,method)
#    d1 = cplx.get_boundary_matrix_group_cochain(n=n,k=(k+1),G=G)
#    d2 = cplx.get_boundary_matrix_group_cochain(n=(n+1), k=(k+2), G=G)
#    delta1 = cplx.get_group_coboundary_matrix(n=n, k=(k+1), G=G)
#    delta2 = cplx.get_group_coboundary_matrix(n=(n+1), k=(k+2), G=G)
#
#    B = sparse.bmat([[d1,   None],
#                     [None, delta2],
#                     [delta1, -d2]])
#    if method == 'column_space_dense':
#        B = B.toarray()
#        #B = B.tolist()
#
#        # For some reason sage doesn't create a matrix with the right base field
#        # if we just call it on a numpy array directly
#        B = matrix(ZZ, B)
#        B = matrix(field,B)
#    else:
#        B = scipy_sparse_matrix_to_sage(field,B)
#
#    indexer = cplx.get_chain_indexer(n=n,k=k,G=G)
#
#    if method in ('column_space','column_space_dense'):
#        Vext = VectorSpace(field, B.nrows())
#        V = VectorSpace(field, indexer.total_dim())
#
#        column_space = B.column_space()
#        column_space_intersect = column_space.intersection(Vext.subspace(Vext.basis()[0:indexer.total_dim()]))
#
#        return V.subspace([v[0:indexer.total_dim()] for v in column_space_intersect.basis()])
#    elif method=='null_space':
#        P = sparse.bmat([[None,delta2],[delta1,-d2]])
#        P = scipy_sparse_matrix_to_sage(field,P)
#        nullspace = P.kernel()
#        return B.image(nullspace)
#    else:
#        raise ValueError, "Undefined method."

def trivialized_by_E2_space(cplx,n,k,G,field):
    d1 = cplx.get_boundary_matrix_group_cochain(n=n,k=(k+1),G=G)
    delta1 = cplx.get_group_coboundary_matrix(n=n, k=(k+1), G=G)

    delta2 = cplx.get_group_coboundary_matrix(n=0,k=0,G=G)

    B = sparse.bmat([[d1],   [delta1]])
    B = scipy_sparse_matrix_to_sage(field,B)

    indexer = cplx.get_chain_indexer(n=n,k=k,G=G)

    Vext = VectorSpace(field, B.nrows())
    V = VectorSpace(field, indexer.total_dim())

    column_space = B.column_space()
    column_space_intersect = column_space.intersection(Vext.subspace(Vext.basis()[0:indexer.total_dim()]))

    #ret = V.subspace([v[0:indexer.total_dim()] for v in column_space_intersect.basis()])
    #delta2 = scipy_sparse_matrix_to_sage(field,delta2)
    #for v in ret.basis():
    #    print delta2*v

    return V.subspace([v[0:indexer.total_dim()] for v in column_space_intersect.basis()])


def find_E2_trivializer(cplx, a, n, k, G, encoder):
    d = cplx.get_boundary_matrix_group_cochain(n=n,k=(k+1),G=G)
    delta = cplx.get_group_coboundary_matrix(n=n, k=(k+1), G=G)

    A = sparse.bmat([[d],[delta]])
    b = numpy.bmat([a,numpy.zeros(A.shape[0]-len(a))]).flat
    return encoder.solve_matrix_equation(A,b)

def find_E3_trivializer(cplx, a, n, k, G, encoder):
    # a is a k-chain, n-group cochain

    d1 = cplx.get_boundary_matrix_group_cochain(n=n,k=(k+1),G=G)
    d2 = cplx.get_boundary_matrix_group_cochain(n=(n+1), k=(k+2), G=G)
    delta1 = cplx.get_group_coboundary_matrix(n=n, k=(k+1), G=G)
    delta2 = cplx.get_group_coboundary_matrix(n=(n+1), k=(k+2), G=G)

    A = sparse.bmat([[d1,   None],
                     [None, delta2],
                     [delta1, -d2]])
    b = numpy.bmat([a,numpy.zeros(A.shape[0]-len(a))]).flat

    return encoder.solve_matrix_equation(A,b)

def affine_transformation_rescale(A,scale):
    A = copy(A)
    d = A.nrows()-1
    A[0:d,d] *= scale
    return A

def affine_transformation_preserves_integer_lattice(A,scale):
    A = affine_transformation_rescale(A,scale)
    return A in MatrixSpace(ZZ, A.nrows())

#def space_group_preserves_integer_lattice(G,scale):
#    return all(affine_transformation_preserves_integer_lattice(matrix(A),scale) 
#            for A in gap.GeneratorsOfGroup(G).sage())

