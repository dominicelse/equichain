import numpy
import chaincplx
import grps
import itertools
import utils
from sage.all import *

def inequalities_array_for_closer_to_one_point(x1, x2, Q):
    assert len(x1) == len(x2)

    x1 = vector(x1)
    x2 = vector(x2)

    a = x2*Q*x2 - x1*Q*x1
    v = 2*(x1-x2)*Q

    return [a] + [ vv for vv in v ]

def voronoi_cell_wrt_neighboring_points(x0, other_pts, Q):
    inequalities = []
    for pt in other_pts:
        inequalities.append(inequalities_array_for_closer_to_one_point(x0, pt, Q))

    return polymake.new_object("Polytope<Rational>", INEQUALITIES=inequalities)

def space_group_orthogonality_matrix(d,G):
    # Some of the space groups in their default setting are not orthogonal with
    # respect to the usual inner product. This function produces a dxd matrix Q
    # such that A'*Q*A = Q for all A in the point group.

    gens = [ matrix(gen.sage())[0:d,0:d] for gen in gap.GeneratorsOfGroup(G) ]

    P = identity_matrix(d)
    if all(gen.transpose() * P * gen == P for gen in gens):
        return P
    
    twothird = Integer(2)/3
    if d == 2:
        P = matrix( [ [ 2*twothird, -twothird ], [ -twothird, 2*twothird ] ])
    elif d == 3:
        P = matrix( [ [ 2*twothird, -twothird, 0 ], [ -twothird, 2*twothird , 0], [0,0,1] ])
    if all(gen.transpose() * P * gen == P for gen in gens):
        return P


    raise RuntimeError, "Could not find orthogonality matrix!"

def wigner_seitz_cell(d, spacegrp, Q=None, x0=None):
    if x0 is None:
        x0 = chaincplx.PointInUniverse( None, (0,)*d )

    if Q is None:
        Q = space_group_orthogonality_matrix(d, spacegrp)

    gens = [ 
            chaincplx.PointInUniverseTranslationAction(gen.sage()) 
            for gen in gap.TranslationBasis(spacegrp)
            ]

    max_order = 2 # Is this sufficient?

    def iterate_neighbors():
        for k in itertools.product(range(-max_order,max_order+1), repeat=len(gens)):
            g = utils.product( (gens[i]**k[i] for i in xrange(1,len(gens))), gens[0]**k[0]  )
            pt = x0.act_with(g)
            yield pt.coords

    return voronoi_cell_wrt_neighboring_points(x0.coords, iterate_neighbors(), Q)
