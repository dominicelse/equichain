from __future__ import division
import itertools
import numpy
import sys
import cProfile
import time
from scipy import sparse

from chaincplx.utils import *
from chaincplx.grps import *
from chaincplx.sageutils import *
import chaincplx.resolutions as resolutions
from sage.all import *

from sage.matrix.matrix_mod2_dense import Matrix_mod2_dense

import cython_fns

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

        if isinstance(action, numpy.matrix):
            action = numpy.array(action)

        v = numpy.empty(len(self.coords)+1, dtype=int)
        v[0:-1] = self.coords
        v[-1] = 1
        vout = numpy.dot(action,v)
        ret = IntegerPointInUniverse(self.universe, vout[:-1])

        return ret

class Universe(object):
    def cell_on_boundary(self, cell):
        return all(not self.point_in_interior(pt) for pt in cell)

    def cell_outside(self, cell):
        return any(self.point_outside(pt) for pt in cell)

    def contains_cell(self, cell, include_boundary):
        if self.cell_outside(cell):
            return False
        else:
            if include_boundary:
                return True
            else:
                return not self.cell_on_boundary(self,cell)

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

class FlatUniverse(Universe):
    def canonicalize_coords(self, coords):
        return coords

class ConvexHullCell(object):
    def __init__(self,points,orientation):
        # The orientation is an anti-symmetric matrix implementing the
        # projected determinant.
        self.points = frozenset(points)
        self._orientation = orientation
        self.original_my_hash = None
        self.original_my_hash = ConvexHullCell.__hash__(self)

    def act_with(self,action):
        ret = ConvexHullCell([p.act_with(action) for p in self.points],
                orientation=transform_levi_civita(self._orientation,action))

        return ret

    def __eq__(a,b):
        return a.points == b.points

    def __ne__(a,b):
        return not a == b

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

def reduce_projected_levi_civita(E, normal_vector):
    return numpy.tensordot(E, normal_vector, (0,0))

def transform_levi_civita(E, R):
    if isinstance(R,MatrixQuotientGroupElement):
        R = R.as_matrix_representative().numpy()
    Rt = numpy.transpose(R)[0:-1,0:-1]
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
    cplx = CellComplex(ndims)
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

class CellComplex(object):
    @staticmethod
    def _get_action_matrix(cells,action):
        cells = list(cells)
        A = numpy.zeros( (len(cells), len(cells)), dtype=int)
        for i in xrange(len(cells)):
            j = CellComplex._get_action_on_cell_index(cells,i,action)
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

    def get_group_coboundary_matrix(self, n,G,k, resolution='cython_bar'):
        return get_group_coboundary_matrix(self.cells[k],n,G, resolution=resolution)

    def get_action_matrix(self, k, action):
        return CellComplex._get_action_matrix(self.cells[k], action)

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

    def get_boundary_matrix_group_cochain(self, k,n,G, resolution='cython_bar'):
        A = self.get_boundary_matrix(k)
        if resolution in ('cython_bar','python_bar'):
            rank = G.size()**n
        else:
            rank = resolution.rank(n)
        return sparse.kron(A, sparse.eye(rank,dtype=int))

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
    return [C[:,i].flat for i in xrange(C.shape[1])]

def trivialized_by_E3_space(cplx,n,k,G,encoder, resolution):
    d1 = cplx.get_boundary_matrix_group_cochain(n=n,k=(k+1),G=G, resolution=resolution)
    d2 = cplx.get_boundary_matrix_group_cochain(n=(n+1), k=(k+2), G=G, resolution=resolution)
    delta1 = cplx.get_group_coboundary_matrix(n=n, k=(k+1), G=G, resolution=resolution)
    delta2 = cplx.get_group_coboundary_matrix(n=(n+1), k=(k+2), G=G, resolution=resolution)

    z = encoder.numpy_zeros((d1.shape[0], delta2.shape[1]))
    A = numpy.bmat([[d1.toarray(),z]])

    B = sparse.bmat([[None, delta2],
                     [delta1, -d2]])
    B = B.toarray()

    return image_of_constrained_subspace(A,B,encoder)


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

def group_cohomology(G,n, resolution, encoder):
    d1 = get_group_coboundary_matrix([TrivialPermutee()], n, G, resolution)
    d2 = get_group_coboundary_matrix([TrivialPermutee()], n-1, G, resolution)

    return kernel_mod_image(d1.toarray(), d2.toarray(), encoder)

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

def trivialized_by_E2_space(cplx,n,k,G,encoder,resolution):
    d1 = cplx.get_boundary_matrix_group_cochain(n=n,k=(k+1),G=G, resolution=resolution)
    delta1 = cplx.get_group_coboundary_matrix(n=n, k=(k+1), G=G, resolution=resolution)

    return image_of_constrained_subspace(d1.toarray(), delta1.toarray(), encoder)

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

