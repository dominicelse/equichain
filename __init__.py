from __future__ import division
import itertools
import functools
import numpy
import sys
#import cProfile
#import time
#import scipy.io
import warnings
from scipy import sparse

from chaincplx.utils import *
from chaincplx.grps import *
from chaincplx.sageutils import *
import chaincplx.resolutions as resolutions
from chaincplx.linalg import *
from sage.all import *

from sage.matrix.matrix_mod2_dense import Matrix_mod2_dense

import cython_fns

from sage.modules.vector_rational_dense import Vector_rational_dense

class PointInUniverse(object):
    def __init__(self, universe, coords):
        self.universe = universe
        coords = universe.canonicalize_coords(coords)
        if isinstance(coords, Vector_rational_dense):
            self.coords = copy(coords)
        else:
            self.coords = copy(vector(QQ, coords))
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
        if isinstance(action, PointInUniverseTranslationAction):
            return action(self)

        if isinstance(action, MatrixQuotientGroupElement):
            action = action.as_matrix_representative()

        v = vector(tuple(self.coords) + (1,))
        vout = action*v
        ret = PointInUniverse(self.universe, vout[:-1])

        return ret

#class IntegerPointInUniverse(object):
#    def __init__(self, universe, coords):
#        self.universe = universe
#        self.coords = numpy.ascontiguousarray(universe.canonicalize_coords_int(coords))
#        self.coords.setflags(write=False)
#        self._hash = hash(self.coords.data)
#
#    def __hash__(self):
#        return self._hash
#
#    def __eq__(x,y):
#        return numpy.array_equal(x.coords,y.coords)
#
#    def __ne__(x,y):
#        return not x == y
#
#    def __str__(self):
#        return str(self.coords)
#
#    def __repr__(self):
#        return str(self)
#
#    def __len__(self):
#        return len(self.coords)
#
#    def act_with(self, action):
#        if isinstance(action, IntegerPointInUniverseTranslationAction):
#            return action(self)
#
#        if isinstance(action, MatrixQuotientGroupElement):
#            action = action.as_matrix_representative_numpy_int()
#
#        if isinstance(action, numpy.matrix):
#            action = numpy.array(action)
#
#        v = numpy.empty(len(self.coords)+1, dtype=int)
#        v[0:-1] = self.coords
#        v[-1] = 1
#        vout = numpy.dot(action,v)
#        ret = IntegerPointInUniverse(self.universe, vout[:-1])
#
#        return ret

from cython_fns import IntegerPointInUniverse,IntegerPointInUniverseTranslationAction

class PointInUniverseTranslationAction(object):
    def __init__(self, trans):
        if isinstance(trans, Vector_rational_dense):
            self.trans = copy(trans)
        else:
            self.trans = vector(QQ, trans)

    def __call__(self, pt):
        return PointInUniverse(pt.universe, pt.coords + self.trans)

    def __mul__(x, y):
        return PointInUniverseTranslationAction(x.trans + y.trans)

    def __pow__(self,n):
        return PointInUniverseTranslationAction(n*self.trans)

    @staticmethod
    def get_translation_basis(d):
        ret = []
        for i in xrange(d):
            trans = [0]*d
            trans[i] = 1
            ret.append(PointInUniverseTranslationAction(trans))
        return ret

class Universe(object):
    def cell_on_boundary(self, cell):
        return all(not self.point_in_interior(pt) for pt in cell)

    def cell_outside(self, cell):
        return any(self.point_outside(pt) for pt in cell)

    def contains_cell(self, cell, include_boundary):
        if self.cell_outside(cell):
            ret = False
        else:
            if include_boundary:
                ret = True
            else:
                ret = not self.cell_on_boundary(cell)
        return ret

class FiniteCubicUniverse(Universe):
    def __init__(self, extents, uncompactified_dirs):
        self.extents = extents
        self.uncompactified_dirs = uncompactified_dirs

    def canonicalize_coords(self, coords):
        coords = list(QQ(c) for c in coords)
        for i in xrange(len(self.extents)):
            if i not in self.uncompactified_dirs:
                coords[i] = (fracpart((coords[i]-self.extents[i][0]) / (self.extents[i][1] - self.extents[i][0]))
                        * (self.extents[i][1] - self.extents[i][0])
                        + self.extents[i][0])
        return vector(QQ,coords)

    def canonicalize_coords_int(self, coords):
        assert len(coords) == len(self.extents)
        coords = numpy.array(coords, dtype=int)
        for i in xrange(len(self.extents)):
            if i not in self.uncompactified_dirs:
                coords[i] = (coords[i] - self.extents[i][0]) % (self.extents[i][1] - self.extents[i][0]) + self.extents[i][0]
        return coords

    def point_in_interior(self, point):
        for d in xrange(len(self.extents)):
            if d in self.uncompactified_dirs:
                if point.coords[d] <= self.extents[d][0] or point.coords[d] >= self.extents[d][1]:
                    return False
        return True

    def point_outside(self, point):
        for d in xrange(len(self.extents)):
            if d in self.uncompactified_dirs:
                if point.coords[d] < self.extents[d][0] or point.coords[d] > self.extents[d][1]:
                    return True
        return False

class CubicUniverseWithBoundary(FiniteCubicUniverse):
    # This is a special case of FiniteCubicUniverse in which open_dirs =
    # range(len(extents)), with an optimized version of canonicalize_coords_int
    # for this case.
    def __init__(self,extents):
        super(CubicUniverseWithBoundary,self).__init__(extents, range(len(extents)))

    def canonicalize_coords_int(self, coords):
        return coords

class FlatUniverse(Universe):
    def canonicalize_coords(self, coords):
        return coords

    def canonicalize_coords_int(self, coords):
        return coords

class ConvexHullCell(object):
    def __init__(self,points,orientation):
        # The orientation is an anti-symmetric matrix implementing the
        # projected determinant. Pass None for an unoriented cell.
        self.points = frozenset(points)
        self._orientation = orientation

    def forget_orientation(self):
        return ConvexHullCell(self.points, None)

    def act_with(self,action):
        if self._orientation is None:
            new_orientation = None
        else:
            new_orientation = transform_levi_civita(self._orientation,action)

        ret = ConvexHullCell([p.act_with(action) for p in self.points],
                orientation=new_orientation)

        return ret

    def __eq__(a,b):
        return a.points == b.points

    def __ne__(a,b):
        return not a == b

    def __hash__(self):
        return hash(self.points)

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
        self.midpoint = midpoint
        super(ConvexHullCellWithMidpoint,self).__init__(points,orientation)

    def act_with(self,action):
        pt = super(ConvexHullCellWithMidpoint,self).act_with(action)
        return ConvexHullCellWithMidpoint(pt.points, pt.orientation(),
                self.midpoint.act_with(action))

    def forget_midpoint(self):
        return ConvexHullCell(self.points,self.orientation)

    def forget_orientation(self):
        return ConvexHullCellWithMidpoint(self.points, None, self.midpoint)

    def __hash__(self):
        h = hash((super(ConvexHullCellWithMidpoint,self).__hash__(), self.midpoint))
        return h

    def __eq__(self,b):
        return super(ConvexHullCellWithMidpoint,self).__eq__(b) and self.midpoint == b.midpoint

    def __ne__(a,b):
        return not a == b

    def __str__(self):
        return "MIDP" + super(ConvexHullCellWithMidpoint,self).__str__() + ";" + str(self.midpoint)

    def __repr__(self):
        return str(self)

class QuotientCell(object):
    def __init__(self, representative_cell, equivalence_relation):
        self.representative_cell = representative_cell
        self.equivalence_relation = equivalence_relation

    def __eq__(a,b):
        return a.equivalence_relation.are_equivalent(a.representative_cell, b.representative_cell)

    def __ne__(a,b):
        return not a == b

    def __hash__(self):
        return self.equivalence_relation.hash_equivalence_class(self.representative_cell)

    def act_with(self, action):
        return QuotientCell(self.representative_cell.act_with(action),
                self.equivalence_relation)

    def boundary(self):
        return FormalIntegerSum( dict( (QuotientCell(k,self.equivalence_relation),v) for k,v in
                self.representative_cell.boundary().coeffs.iteritems() ) )

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return "QUOTIENT:" + str(self.representative_cell)

class EquivalenceRelationFromCommutingActionGenerators(object):
    def __init__(self, gens, representatives, default_max_order=5,
            reduce_order=None, precompute_representatives_for=None,
            representatives_helper=None):
        self.gens = gens
        self.default_max_order = default_max_order
        self.representatives_helper = representatives_helper

        # If reduce_order is not None, then the set of representatives might be
        # overcomplete, and we try to reduce it (but we only look for
        # relations generated by powers <= reduce_order.)
        if reduce_order is not None:
            self.map_to_representatives = {}
            self.representatives = set()


            for cell in representatives:
                if not self.canonical_representative(cell,max_order=reduce_order,return_bool=True):
                    self.representatives.add(cell)
                    self.map_to_representatives[cell] = cell
        else:
            self.map_to_representatives = dict( (cell,cell) for cell in
                    representatives )
            self.representatives = frozenset(representatives)

        if precompute_representatives_for is not None:
            for cell in precompute_representatives_for:
                self.canonical_representative(cell)

    def canonical_representative(self,cell, max_order=None, return_bool=False):
        if cell in self.map_to_representatives:
            return self.map_to_representatives[cell]

        # If the cell is not already cached, use the representatives helper to
        # try to get a better representative
        if self.representatives_helper is not None:
            cell = self.representatives_helper(cell)

            # Now try again
            if cell in self.map_to_representatives:
                return self.map_to_representatives[cell]
        
        #max(max(numpy.max(numpy.abs(pt.coords)) for pt in basecell.points)
                #for basecell in cell.vertices)

        if max_order is None:
            max_order = self.default_max_order

        for morder in xrange(max_order+1):
            for k in itertools.product(range(-morder,morder+1), repeat=len(self.gens)):
                g = product( (self.gens[i]**k[i] for i in xrange(1,len(self.gens))),
                        self.gens[0]**k[0] )
                acted_cell = cell.act_with(g)
                if acted_cell in self.representatives:
                    self.map_to_representatives[cell] = acted_cell

                    if return_bool:
                        return True
                    else:
                        return acted_cell

        if return_bool:
            return False
        else:
            raise ValueError, "Cell doesn't appear to have a representative."

    def are_equivalent(self, cell1, cell2):
        return self.canonical_representative(cell1) == self.canonical_representative(cell2)

    def hash_equivalence_class(self, cell):
        return hash(self.canonical_representative(cell))

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

def reduce_projected_levi_civita(E, normal_vector):
    return numpy.tensordot(E, normal_vector, (0,0))

def transform_levi_civita(E, R):
    if isinstance(R,MatrixQuotientGroupElement):
        R = R.as_matrix_representative().numpy()
    Rt = numpy.transpose(R)[0:-1,0:-1]
    for i in xrange(len(E.shape)):
        E = numpy.tensordot(E, Rt, (0,0))
    return E

def get_relative_orientation_cells(cell1, cell2):
    if isinstance(cell1, ConvexHullCell):
        return get_relative_orientation(cell1.orientation(),cell2.orientation())
    else:
        return 1

def get_relative_orientation(orientation1, orientation2):
    if numpy.array_equal(orientation1,orientation2):
        return 1
    elif numpy.array_equal(orientation1,-orientation2):
        return -1
    else:
        raise RuntimeError, "Orientations are not relative."

def cubical_cell_boundary(ndims, basepoint_coords, direction, orientation,
        universe, with_midpoints, scale, include_boundary_cells, pointclass):
    coeffs = {}
    for face_direction in direction:
        normal_vector = numpy.zeros(shape=(len(basepoint_coords),),dtype=int)
        normal_vector[face_direction] = 1
        remaining_directions = [ d for d in direction if d != face_direction ]

        cell = cubical_cell(ndims-1, basepoint_coords, remaining_directions,
                universe, with_midpoints, scale, pointclass)
        if universe.contains_cell(cell, include_boundary_cells):
            reduced_orientation = reduce_projected_levi_civita(orientation, normal_vector)
            coeffs[cell] = get_relative_orientation(reduced_orientation,
                    cell.orientation())

        coord = numpy.array(basepoint_coords)
        coord[face_direction] += scale
        cell = cubical_cell(ndims-1, coord, remaining_directions, universe,
                with_midpoints, scale, pointclass)
        if universe.contains_cell(cell, include_boundary_cells):
            reduced_orientation = reduce_projected_levi_civita(orientation, -normal_vector)
            coeffs[cell] = get_relative_orientation(reduced_orientation,
                    cell.orientation())

    return FormalIntegerSum(coeffs)

def _cubical_complex_base(ndims, extents, universe, with_midpoints, scale,
        include_boundary_cells=False, pointclass=PointInUniverse):
    cplx = CellComplex(ndims)
    for celldim in xrange(ndims+1):
        for direction in itertools.combinations(xrange(ndims),celldim):
            coord_ranges = [ xrange(extents[i][0], extents[i][1] + 
                (1 if i in universe.uncompactified_dirs else 0), scale) for i in xrange(ndims) ]
            for coord in itertools.product(*coord_ranges):
                cell = cubical_cell(celldim,coord,direction,
                        universe, with_midpoints, scale, pointclass)
                if universe.contains_cell(cell, include_boundary_cells):
                    cplx.add_cell(celldim,
                            cell,
                            cubical_cell_boundary(celldim,coord,direction,
                                cell.orientation(), universe,
                                with_midpoints,scale,include_boundary_cells,
                                pointclass)
                            )
    return cplx

#def minimal_complex_torus(ndims,scale=1,pointclass=PointInUniverse):
#    extents = [ [0,scale] ]*ndims
#    return _cubical_complex_base(ndims, 
#            [ [0,scale] ]*ndims, FiniteCubicUniverse(extents, []),
#            with_midpoints=True, scale=scale, pointclass=pointclass)

def sage_polymake_object_from_gap(p):
    filename = gap.FullFilenameOfPolymakeObject(p)
    return polymake('load("' + str(filename) + '");')
    
def cell_complex_from_polytope(p, coord_subset, remember_orientation=True):
    if remember_orientation:
        raise NotImplementedError

    universe = FlatUniverse()

    def conv_vertex(v):
        return PointInUniverse(universe, [ sage_eval(str(v[i])) for i in coord_subset ])

    vertices = [ conv_vertex(v) for v in p.VERTICES ]

    def conv_cell(c):
        vertices_in_cell = [ vertices[int(i)] for i in c ]
        cell = ConvexHullCell(vertices_in_cell, orientation=None)
        return cell

    cells = [ conv_cell(c) for c in p.HASSE_DIAGRAM.FACES ]

    d = int(p.CONE_DIM)-1
    cplx = CellComplex(d)
    for k in xrange(d+1):
        for face in p.HASSE_DIAGRAM.nodes_of_dim(k):
            i = int(face)
            if k > 0:
                boundary = FormalIntegerSum(dict( (cells[int(bi)],1) for 
                    bi in p.HASSE_DIAGRAM.ADJACENCY.in_adjacent_nodes(i) ))
            else:
                boundary = FormalIntegerSum()
            cplx.add_cell(k, cells[int(face)], boundary)

    return cplx

def simplicial_cell_complex_from_polymake(p, coord_subset, remember_orientation=True):
    if remember_orientation:
        raise NotImplementedError

    universe = FlatUniverse()

    def conv_vertex(v):
        return PointInUniverse(universe, [ sage_eval(str(v[i])) for i in coord_subset ])

    vertices = [ conv_vertex(v) for v in p.COORDINATES ]

    def conv_cell(c):
        vertices_in_cell = [ vertices[int(i)] for i in c ]
        return SimplexCell(vertices_in_cell, orientation=None)

    d = int(p.DIM)
    cplx = CellComplex(d)
    for facet in p.FACETS:
        conv_facet = conv_cell(facet)
        assert conv_facet.dimension() == d
        cplx.add_cell(d, conv_facet)
    cplx.complete()

    return cplx

def cubical_complex(ndims, sizes, open_dirs=[], with_midpoints=False, scale=1,
        include_boundary_cells=False,
        pointclass=PointInUniverse):
    assert len(sizes) == ndims
    assert all(d >= 0 and d < ndims for d in open_dirs)

    for d in xrange(ndims):
        if sizes[d] <= 1 and not d in open_dirs and not with_midpoints:
            raise ValueError, "Don't use this function to construct a compactified direction of length <= 2 without setting with_midpoints=True, this causes problems."

    extents = [ [-sizes[i]*scale,sizes[i]*scale] for i in xrange(ndims) ]
    universe = FiniteCubicUniverse(extents, open_dirs)
    return _cubical_complex_base(ndims, extents, universe,
            with_midpoints,scale,include_boundary_cells,pointclass)

def _find_outside_universe(universe, cell):
    for v in cell.vertices:
        for p in v.points:
            if universe.point_outside(p):
                return p

def _torus_minimal_barycentric_subdivision_representatives_helper(toroidal_universe,
        pointclass, cell):
    for v in cell.vertices:
        for p in v.points:
            p_translated = pointclass(toroidal_universe, p.coords)
            if not numpy.array_equal(p_translated.coords,p.coords):
                t = p_translated.coords - p.coords
                return cell.act_with(IntegerPointInUniverseTranslationAction(t))

    return cell

def polymaketest(starting_pt, d,i):
    starting_pt = gap(starting_pt)
    G = gap.StandardAffineCrystGroup(gap.SpaceGroupOnRightIT(d,i))
    P = gap.FundamentalDomainStandardSpaceGroup(starting_pt, G)
    P = sage_polymake_object_from_gap(P)
    return cell_complex_from_polytope(P, remember_orientation=False, coord_subset=range(1,d+1))

def space_group_wigner_seitz_cell(d, i, starting_pt=None):
    if starting_pt is None:
        for denom in xrange(10):
            for coords in itertools.product(xrange(denom), repeat=d):
                coords_divided = [ Integer(coord)/denom for coord in coords ]
                try:
                    return space_group_wigner_seitz_cell(d,i, coords_divided)
                except RuntimeError as e:
                    if e.args[0].args[0].find("Error, center point not in general position") != -1:
                        continue
                    else:
                        raise
    else:
        print starting_pt
        starting_pt = gap(starting_pt)
        G = gap.StandardAffineCrystGroup(gap.SpaceGroupOnRightIT(d,i))
        P = gap.FundamentalDomainStandardSpaceGroup(starting_pt, G)
        P = sage_polymake_object_from_gap(P)
        return cell_complex_from_polytope(P, remember_orientation=False, coord_subset=range(1,d+1))

def space_group_wigner_seitz_barycentric_subdivision(d, i, starting_pt = None):
    c = space_group_wigner_seitz_cell(d, i, starting_pt)

    c2 = c.barycentric_subdivision()

    #gens = list(translation_generators_numpy(ndims,scale=scale,with_inverses=True))
    gens = PointInUniverseTranslationAction.get_translation_basis(d)
    equiv_relation = EquivalenceRelationFromCommutingActionGenerators(gens,
            c2.all_cells_iterator(), reduce_order=1,
            representatives_helper=None)

    return c2.quotient(equiv_relation)

    #starting_pt = gap(starting_pt)
    #G = gap.StandardAffineCrystGroup(gap.SpaceGroupOnRightIT(d,i))
    #P = gap.FundamentalDomainStandardSpaceGroup(starting_pt, G)
    #P = sage_polymake_object_from_gap(P)
    #   
    #B = P.barycentric_subdivision()
    #c = simplicial_cell_complex_from_polymake(B, remember_orientation=False,
    #        coord_subset=range(1,d+1))
    #
    #gens = PointInUniverseTranslationAction.get_translation_basis(d)
    #equiv_relation = EquivalenceRelationFromCommutingActionGenerators(gens,
    #        c.all_cells_iterator(), reduce_order=1,
    #        representatives_helper=None)
    #
    #return c.quotient(equiv_relation)

def torus_minimal_barycentric_subdivision(ndims):
    scale = 2
    extents = [[0,scale]]*ndims
    universe = CubicUniverseWithBoundary(extents)
    toroidal_universe = FiniteCubicUniverse(extents, [])
    pointclass = IntegerPointInUniverse

    c = _cubical_complex_base(ndims, extents, universe, with_midpoints=True,
            scale=scale, include_boundary_cells=True, pointclass=pointclass)
    #return c

    c2 = c.barycentric_subdivision()
    #return c2


    #gens = list(translation_generators_numpy(ndims,scale=scale,with_inverses=True))
    gens = IntegerPointInUniverseTranslationAction.get_translation_basis(ndims,scale)
    equiv_relation = EquivalenceRelationFromCommutingActionGenerators(gens,
            c2.all_cells_iterator(), reduce_order=1,
    # Actually, using the representatives helper doesn't seem to improve
    # performance that much, so I disabled it.
            representatives_helper=functools.partial(_torus_minimal_barycentric_subdivision_representatives_helper,
                toroidal_universe,pointclass))

    return c2.quotient(equiv_relation)

def get_stabilizer_group(cell,G):
    gs = [g for g in G if cell.act_with(g) == cell]
    try:
        return G.subgroup(gs)
    except AttributeError:
        return gs


#def solve_matrix_equation(A, b, over_ring=ZZ):
#    b = vector(over_ring,b)
#    A = scipy_sparse_matrix_to_sage(over_ring, A)
#    return A.solve_right(b).numpy(dtype=int)

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

class TrivialPermutee(object):
    def act_with(self,g):
        return self

    def __eq__(a,b):
        return True

    def __ne__(a,b):
        return False

    def __hash__(self):
        return 0

    def orientation(self):
        return 1


class ComplexChainIndexer(object):
    def __init__(self, n, cells, G):
        self.internal_indexer = MultiIndexer(*( (G.size(),) * n + (len(cells),) ))

    def to_index(self, gi, ci):
        return self.internal_indexer.to_index(*(tuple(gi) + (ci,)))

    def __call__(self, gi, ci):
        return self.to_index(gi,ci)

    def total_dim(self):
        return self.internal_indexer.total_dim()

def get_group_coboundary_matrix(cells, n,G, resolution='cython_bar'):
    if resolution == 'cython_bar':
        return cython_fns.get_group_coboundary_matrix(cells,n,G)

    mapped_cell_indices, mapping_parities = get_group_action_on_cells(cells,G,inverse=True)

    if resolution != 'python_bar':
        return resolution.dual_d_matrix(n, len(cells),
                mapped_cell_indices, mapping_parities, raw=True)

    indexer_out = MultiIndexer(*( (G.size(),) * (n+1) + (len(cells),) ))
    indexer_in = MultiIndexer(*( (G.size(),) * n + (len(cells),) ))

    A = sparse.dok_matrix((indexer_out.total_dim(), indexer_in.total_dim()), dtype=int)
    #A = matrix(base_ring, indexer_out.total_dim(), indexer_in.total_dim(), sparse=True)


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

    return ScipySparseMatrixOverRing(A.tocsc(), ring=ZZ)

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
        parity = get_relative_orientation_cells(acted_cell, cells[acted_ci])

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

class OrderedSimplex(object):
    """ This class represents a (combinatorial) simplex, i.e. a tuple of
    vertices (v_1, ... , v_n). Note that it is assumed that there is a partial
    ordering on vertices and that the vertices are input such that v_1 < v_2 <
    v_3 < ... < v_n. This ordering must also be preserved under action on the
    vertices. """

    def __init__(self, vertices):
        self.vertices = tuple(vertices)
        self._hash = hash(self.vertices)

    def __eq__(a,b):
        return a.vertices == b.vertices

    def __ne__(a,b):
        return a.vertices != b.vertices

    def __hash__(self):
        return self._hash

    def boundary(self):
        n = len(self.vertices)
        if n == 1:
            return FormalIntegerSum({})

        coeffs = {}
        for i in xrange(n):
            reduced_word = self.vertices[0:i] + self.vertices[(i+1):]
            coeffs[OrderedSimplex(reduced_word)] = (-1)**i

        return FormalIntegerSum(coeffs)

    def act_with(self,action):
        return OrderedSimplex(v.act_with(action) for v in self.vertices)

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return "SIMPLEX" + str(self.vertices)

class SimplexCell(ConvexHullCell):
    def __init__(self, points, orientation):
        if orientation is not None:
            raise NotImplementedError, "Oriented simplices are not implemented yet."

        super(SimplexCell,self).__init__(points,orientation)

    def dimension(self):
        return len(self.points)-1

    def boundary(self):
        if self.dimension() == 0:
            return FormalIntegerSum()

        points_ordered = list(self.points)

        boundary_cells = []

        for i in xrange(len(points_ordered)):
            onepoint_removed = points_ordered[0:i] + points_ordered[(i+1):]
            boundary_cells.append(SimplexCell(onepoint_removed, None))

        return FormalIntegerSum(dict(
            (boundary_cell,1) for boundary_cell in boundary_cells
            ))

class CellComplex(object):
    #@staticmethod
    #def _get_action_matrix(cells,action):
    #    cells = list(cells)
    #    A = numpy.zeros( (len(cells), len(cells)), dtype=int)
    #    for i in xrange(len(cells)):
    #        j = CellComplex._get_action_on_cell_index(cells,i,action)
    #        A[j,i] = get_relative_orientation(acted_cell.orientation(), cells[j].orientation())
    #    return A

    def all_cells_iterator(self):
        for cells_k in self.cells:
            for cell in cells_k:
                yield cell

    def all_cells_iterator_unoriented(self):
        for cells_k in self.cells:
            for cell in cells_k:
                yield cell.forget_orientation()

    def quotient(self, equiv_relation):
        cplx = CellComplex(self.ndims)

        for k in xrange(self.ndims+1):
            quotient_cells = set(QuotientCell(cell,equiv_relation) for cell in self.cells[k])
            for q in quotient_cells:
                cplx.add_cell(k, q)

        return cplx

    def _all_contained_cells_iterator_unoriented(self, cell):
        for boundary_cell in self.boundary_data[cell].coeffs.keys():
            yield boundary_cell.forget_orientation()
            for c in self._all_contained_cells_iterator_unoriented(boundary_cell):
                yield c

    def _barycentric_word_iterator(self, base_cell=None):
        if base_cell is None:
            for cell in self.all_cells_iterator_unoriented():
                for word in self._barycentric_word_iterator(cell):
                    yield word
        else:
            yield (base_cell,)
            for cell in self._all_contained_cells_iterator_unoriented(base_cell):
                for word in self._barycentric_word_iterator(cell):
                    yield word + (base_cell,)

    def barycentric_subdivision(self):
        cplx = CellComplex(self.ndims)
        for word in self._barycentric_word_iterator():
            cplx.add_cell(len(word)-1, OrderedSimplex(word))
        return cplx

    def _get_action_on_cell_index(self,cells,action,i):
        acted_cell = cells[i].act_with(action)
        return cells.index(acted_cell)

    def get_group_coboundary_matrix(self, n,G,k, resolution='cython_bar'):
        return get_group_coboundary_matrix(self.cells[k],n,G, resolution=resolution)

    #def get_action_matrix(self, k, action):
    #    return CellComplex._get_action_matrix(self.cells[k], action)

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
        cells_km1 = self.cells[k-1]
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

    def get_boundary_matrix_group_cochain(self, k,n,G, resolution='cython_bar'):
        A = self.get_boundary_matrix(k)
        if resolution in ('cython_bar','python_bar'):
            rank = G.size()**n
        else:
            rank = resolution.rank(n)
        return ScipySparseMatrixOverZ(sparse.kron(A,
            sparse.eye(rank,dtype=int)).tocsc())

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

    def add_cell(self, ndim, cell, boundary=None):
        self.cells[ndim].append(cell)
        if boundary is None:
            self.boundary_data[cell] = cell.boundary()
        else:
            self.boundary_data[cell] = boundary

    # If there are cells that appear in the boundary data (perhaps recursively), 
    # but not have not themselves been added to self.cells, add them now.
    # Note that this requires all cells not already known to have boundary() and
    # dimension() methods.
    def complete(self):
        for cell in self.cells[self.ndims]:
            self._find_all_boundary_cells(cell)

    def _find_all_boundary_cells(self,base):
        for cell in base.boundary().itervectors():
            if cell not in self.cells[cell.dimension()]:
                assert cell.dimension() == base.dimension() - 1 
                self.cells[cell.dimension()].append(cell)
                self._find_all_boundary_cells(cell)

    def __init__(self,ndims):
        self.ndims = ndims
        self.cells = [None]*(ndims+1)
        self.boundary_data = {}
        for i in xrange(ndims+1):
            self.cells[i] = IndexedSet()

    def get_group_action_on_cells(self, G, k, inverse=False):
        return get_group_action_on_cells(self.cells[k], G, inverse)

def test_has_solution(fn):
    try:
        fn()
    except ValueError as e:
        if e.args[0] == "matrix equation has no solutions":
            return False
        else:
            raise
    return True

def trivialized_by_E3_space(cplx,n,k,G,ring, resolution):
    d1 = cplx.get_boundary_matrix_group_cochain(n=n,k=(k+1),G=G, resolution=resolution)
    d2 = cplx.get_boundary_matrix_group_cochain(n=(n+1), k=(k+2), G=G, resolution=resolution)
    delta1 = cplx.get_group_coboundary_matrix(n=n, k=(k+1), G=G, resolution=resolution)
    delta2 = cplx.get_group_coboundary_matrix(n=(n+1), k=(k+2), G=G, resolution=resolution)

    d1,d2,delta1,delta2 = (x.change_ring(ring) for x in (d1,d2,delta1,delta2))
    factory = d1.factory()

    z = factory.zeros((d1.shape[0], delta2.shape[1]))
    A = factory.bmat([[d1,z]])

    B = factory.bmat([[None, delta2],
                     [delta1, -d2]])

    return image_of_constrained_subspace(A,B)

#def trivialized_by_E3_but_not_E2(cplx,n,k,G,encoder):
#    triv_by_E3 = trivialized_by_E3_space(cplx,n,k,G,encoder)
#
#    ret = []
#    
#    for v in triv_by_E3:
#        if not test_has_solution(lambda: find_E2_trivializer(cplx,v,n,k,G,encoder)):
#            ret.append(v)
#    return ret

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

def group_cohomology(G,n, resolution, ring):
    d1 = get_group_coboundary_matrix([TrivialPermutee()], n, G, resolution)
    d2 = get_group_coboundary_matrix([TrivialPermutee()], n-1, G, resolution)

    d1,d2 = (x.change_ring(ring) for x in (d1, d2))

    return kernel_mod_image(d1,d2)

def trivialized_by_E2_space(cplx,n,k,G,ring,resolution):
    d1 = cplx.get_boundary_matrix_group_cochain(n=n,k=(k+1),G=G, resolution=resolution)
    delta1 = cplx.get_group_coboundary_matrix(n=n, k=(k+1), G=G, resolution=resolution)

    d1,delta1 = (x.change_ring(ring) for x in (d1,delta1))

    return image_of_constrained_subspace(d1, delta1)

def find_E2_trivializer(cplx, a, n, k, G, ring):
    d = cplx.get_boundary_matrix_group_cochain(n=n,k=(k+1),G=G)
    delta = cplx.get_group_coboundary_matrix(n=n, k=(k+1), G=G)

    d,delta = (x.change_ring(ring) for x in (d,delta))

    factory = d.factory()
    A = factory.bmat([[d],[delta]])
    b = factory.concatenate_vectors(a,factory.zero_vector(A.shape[0]-len(a)))
    return A.solve_right(b)

def find_E3_trivializer(cplx, a, n, k, G, ring):
    # a is a k-chain, n-group cochain

    d1 = cplx.get_boundary_matrix_group_cochain(n=n,k=(k+1),G=G)
    d2 = cplx.get_boundary_matrix_group_cochain(n=(n+1), k=(k+2), G=G)
    delta1 = cplx.get_group_coboundary_matrix(n=n, k=(k+1), G=G)
    delta2 = cplx.get_group_coboundary_matrix(n=(n+1), k=(k+2), G=G)

    d1,d2,delta1,delta2 = (x.change_ring(ring) for x in (d1,d2,delta1,delta2))

    factory = d1.factory()

    A = factory.bmat([[d1,   None],
                     [None, delta2],
                     [delta1, -d2]])
    b = factory.concatenate_vectors(a,factory.zero_vector(A.shape[0]-len(a)))

    return A.solve_right(b)

def gap_space_group_translation_subgroup(G,n):
    gap_fn = gap("""function(G,n)
    local T,basis,translation_to_affine;

    translation_to_affine := function(T)
        local n,A,i;
        n := Length(T);
        A := NullMat(n+1,n+1);
        for i in [1..n] do
            A[i][n+1] := T[i];
        od;
        for i in [1..(n+1)] do
            A[i][i] := 1;
        od;
        return A;
    end;

    basis := TranslationBasis(G)*n;
    T := List(basis,translation_to_affine);
    return Subgroup(G, T);
    end;
    """)

    return gap_fn(G,n)

def affine_transformation_preserves_integer_lattice(A,scale):
    A = affine_transformation_rescale(A,scale)
    return A in MatrixSpace(ZZ, A.nrows())

#def space_group_preserves_integer_lattice(G,scale):
#    return all(affine_transformation_preserves_integer_lattice(matrix(A),scale) 
#            for A in gap.GeneratorsOfGroup(G).sage())

