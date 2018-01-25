import numpy
import chaincplx
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

def wigner_seitz_cell(d, spacegrp, Q=None, x0=None):
    if x0 is None:
        x0 = chaincplx.PointInUniverse( None, (0,)*d )

    if Q is None:
        Q = identity_matrix(d)

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
