import numpy
import equichain
import grps
import itertools
import utils
from sage.all import *

def argmax(l, f):
    return max(l, key=f)

def good_atom_locations(d,G):
    # Finds a good set of "atom locations" within a unit cell

    wyckoff_positions = gap.WyckoffPositions(G)

    best = argmax(wyckoff_positions, lambda pos: gap.Size(gap.WyckoffStabilizer(pos)))
    orbit = gap.WyckoffOrbit(best)

    return [ equichain.PointInUniverse(None, gap.WyckoffTranslation(pos).sage())
            for pos in orbit ]

def inequalities_array_for_closer_to_one_point(x1, x2, Q):
    assert len(x1) == len(x2)

    x1 = vector(x1.coords)
    x2 = vector(x2.coords)

    a = x2*Q*x2 - x1*Q*x1
    v = 2*(x1-x2)*Q

    ret = [ a ] + [ vv for vv in v ]
    assert ret != [0,0,0,0]
    return ret

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

def wigner_seitz_cplx(d, spacegrp):
    basepoints = set(good_atom_locations(d, spacegrp))
    Q = space_group_orthogonality_matrix(d, spacegrp)

    gens = [ 
            equichain.PointInUniverseTranslationAction(gen.sage()) 
            for gen in gap.TranslationBasis(spacegrp)
            ]

    max_order = 2 # Is this sufficient?

    def iterate_neighbors():
        for k in itertools.product(range(-max_order,max_order+1), repeat=len(gens)):
            if all(kk == 0 for kk in k):
                continue
            g = utils.product( (gens[i]**k[i] for i in xrange(1,len(gens))), gens[0]**k[0]  )
            for pt in basepoints:
                acted = pt.act_with(g)
                yield pt.act_with(g)

    cplx = None

    for basept in basepoints:
        neighbors = list(iterate_neighbors())
        otherbasepts = [ pt for pt in basepoints if pt != basept ]
        cell = voronoi_cell_wrt_neighboring_points(basept, neighbors + otherbasepts, Q)
        cplx_for_cell = equichain.cell_complex_from_polytope(cell,
                remember_orientation=False, coord_subset=range(1,d+1))

        if cplx is None:
            cplx = cplx_for_cell
        else:
            cplx = cplx.merge(cplx_for_cell)

    return cplx

def space_group_wigner_seitz_barycentric_subdivision(d, G):
    c = wigner_seitz_cplx(d, G)

    c2 = c.barycentric_subdivision()

    #gens = list(translation_generators_numpy(ndims,scale=scale,with_inverses=True))
    gens = [ 
            equichain.PointInUniverseTranslationAction(gen.sage()) 
            for gen in gap.TranslationBasis(G)
            ]
    equiv_relation = equichain.EquivalenceRelationFromCommutingActionGenerators(gens,
            c2.all_cells_iterator(), reduce_order=2,
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

