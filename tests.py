import unittest
from sage.all import *
import chaincplx
import chaincplx.resolutions
import numpy
import itertools

def make_twistgrp():
    eye = matrix.identity(4)
    rot = matrix([[0,1,0,0],[-1,0,0,0],[0,0,1,0],[0,0,0,1]])
    trans = matrix([[1,0,0,0],[0,1,0,0],[0,0,1,1], [0,0,0,1]])
    refl1 = matrix([[-1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
    refl2 = matrix([[1,0,0,0],[0,-1,0,0],[0,0,1,0],[0,0,0,1]])
    G = gap.AffineCrystGroupOnLeft(rot,trans,refl1,refl2)
    N = gap.AffineCrystGroupOnLeft(trans**2)
    return chaincplx.GapAffineQuotientGroup(G,N)

class ComplexWithGroupActionGenericTests(object):
    def setUp(self):
        if self.descr == 'twistgrp':
            self.G = make_twistgrp()
            self.cplx = chaincplx.cubical_complex(3, [1,1,1], [0,1],
                    with_midpoints=True)
            self.R = Integers(2)
            self.encoder = chaincplx.get_numpy_encoder_Zn(2)
        else:
            raise ValueError, "not valid chain complex description"

    def test_cython(self):
        for k in xrange(len(self.cplx.cells)):
            for n in xrange(2):
                D1 = self.cplx.get_group_coboundary_matrix(n=n,k=k,G=self.G,
                        resolution='cython_bar')
                D2 = self.cplx.get_group_coboundary_matrix(n=n,k=k,G=self.G,
                        resolution=chaincplx.resolutions.BarResolution(self.G))
                self.assertEqual(D1.shape, D2.shape)
                self.assertEqual( (D1 != D2).nnz, 0 )

    def get_cocycle_soln(self,k,n,cell_index):
        cell = self.cplx.cells[k][cell_index]
        s = chaincplx.get_nonidentity(chaincplx.get_stabilizer_group(cell,self.G))

        indexer_in = self.cplx.get_chain_indexer(G=self.G,k=k,n=n)
        
        A = self.cplx.get_group_coboundary_matrix(G=self.G,k=k,n=n)
            
        subs_indices = []
        xconstr = numpy.zeros(indexer_in.total_dim(), dtype=int)
        xconstr[indexer_in( (s.toindex(),), cell_index)] = 1
        subs_indices += [indexer_in( (s.toindex(),), cell_index)]

        for ci in xrange(len(self.cplx.cells[k])):
            xconstr[indexer_in( (0,), ci)] = 0

        soln = chaincplx.solve_matrix_equation_with_constraint(A, subs_indices, xconstr, over_ring=self.R) 

        return soln

    def test_cocycle_soln(self):
        for k in xrange(len(self.cplx.cells)):
            n = 1

            for orbit in self.cplx.get_group_orbits(k,self.G):
                S = chaincplx.get_stabilizer_group(self.cplx.cells[k][orbit[0]],self.G)
                if S.size() != 2:
                    continue

                indexer_in = self.cplx.get_chain_indexer(G=self.G,k=k,n=n)
                soln = self.get_cocycle_soln(k=k,n=n,cell_index=orbit[0])

                for ci in orbit:
                    cell = self.cplx.cells[k][ci]
                    S = chaincplx.get_stabilizer_group(cell,self.G)
                    s = chaincplx.get_nonidentity(S)
                    self.assertEqual(soln[indexer_in( (0,), ci)], 0)
                    self.assertEqual(soln[indexer_in( (s.toindex(),), ci)], 1)

class ComplexWithGroupActionTestIntegerEquivalence(object):
    def test_integer_equivalence(self):
        cplx1 = self.make_complex(chaincplx.PointInUniverse)
        cplx2 = self.make_complex(chaincplx.IntegerPointInUniverse)

        for k in xrange(len(cplx1.cells)):
            act1 = cplx1.get_group_action_on_cells(self.G,k,inverse=True)
            act2 = cplx2.get_group_action_on_cells(self.G,k,inverse=True)
            self.assertTrue(numpy.array_equal(act1[0],act2[0]))
            self.assertTrue(numpy.array_equal(act1[1],act2[1]))

        for k in xrange(1,len(cplx1.cells)):
            D1 = cplx1.get_boundary_matrix(k)
            D2 = cplx2.get_boundary_matrix(k)
            self.assertTrue( (D1!=D2).nnz == 0)

class TwistGrpIntegerTests(ComplexWithGroupActionTestIntegerEquivalence,
        unittest.TestCase):

    def setUp(self):
        self.G = make_twistgrp()

    def make_complex(self,pointclass):
        return chaincplx.cubical_complex(3, [1,1,2], [0,1],
                    with_midpoints=False, pointclass=pointclass)

class TwistGrpTests(ComplexWithGroupActionGenericTests, unittest.TestCase):
    descr = "twistgrp"

    def test_descent(self):
        k=2
        n=1

        orbit = self.cplx.get_group_orbits(k,self.G)[0]
        cell_index = orbit[0]

        soln = self.get_cocycle_soln(k,n,cell_index)

        # soln_boundary is 1-group-cocycle, 1-chain
        D = self.cplx.get_boundary_matrix_group_cochain(k=2,n=1,G=self.G)
        soln_boundary = D.dot(soln) % 2

        # soln2 is 0-group-cocycle, 1-chain
        A2 = self.cplx.get_group_coboundary_matrix(0,self.G,1)
        soln2 = self.encoder.solve_matrix_equation(A2, soln_boundary)

        D2 = self.cplx.get_boundary_matrix(1)
        u = D2.dot(soln2) % 2

        self.assertTrue(numpy.array_equal(u,[0,0]))

    def test_E2(self):
        a = numpy.array([1,1])
        self.assertFalse(
                chaincplx.test_has_solution(lambda:chaincplx.find_E2_trivializer(self.cplx,a,n=0,k=0,G=self.G,encoder=self.encoder))
                )

    def test_E3(self):
        a = numpy.array([1,1])
        self.assertFalse(
                chaincplx.test_has_solution(lambda:chaincplx.find_E3_trivializer(self.cplx,a,n=0,k=0,G=self.G,encoder=self.encoder))
                )

def check_space_group(i, compute=True):
    d=3
    try:
        if d in (2,3):
            G0 = gap.SpaceGroupIT(d,i)
        else:
            G0 = gap.SpaceGroup(d,i)
        G = gap.StandardAffineCrystGroup(G0)
        #if G != G0:
        #    print "not isometric", i
        #    return
        N = gap.translation_subgroup(G,1)
        Gq = GapAffineQuotientGroup(G,N, scale=4)

        #assert space_group_preserves_integer_lattice(G,scale=2)
        n=2

        if compute:
            print i, trivialized_by_E3_space(cplx,0,0,Gq,
                    get_numpy_encoder_Zn(n=n))
            #print i, trivialized_by_E3_but_not_E2(cplx,0,0,Gq,
            #        get_numpy_encoder_Zn(n=n))
        else:
            for k in xrange(len(cplx.cells)):
                cplx.get_group_action_on_cells(Gq,k)

            print "yes:", i
    except ComplexNotInvariantError:
        print "complex not invariant:", i
    except NotIntegerMatrixError:
        print "not integer:", i

class GroupCohomologyTests(unittest.TestCase):
    def setUp(self):
        gap.load_package("hap")

    def test_group_cohomology(self):
        for i in range(2,18):
            G = gap.SpaceGroupIT(2,i)

            expected_answer=list(gap.GroupCohomology(gap.PointGroup(G),4))

            Gq = chaincplx.GapAffineQuotientGroup(G,chaincplx.gap_space_group_translation_subgroup(G,1))
            resolution = chaincplx.resolutions.HapResolution(
                    gap.ResolutionFiniteGroup(Gq.gap_quotient_grp, 5),
                    Gq)
            answer=chaincplx.group_cohomology(Gq, 4,
                    resolution=resolution,
                    encoder=chaincplx.NumpyEncoderZ())

            self.assertEqual(answer,expected_answer)

class SpaceGroupTests(unittest.TestCase):
    def setUp(self):
        self.d = 3
        self.cplx = chaincplx.cubical_complex(self.d, [1]*self.d, [], with_midpoints=True, scale=2,
                pointclass=chaincplx.IntegerPointInUniverse)

    def test_somethings(self):
        expected_answers = {
                2: [],
                10: [],
                11: [],
                33: [[1,1,1,1,1,1,1,1]],
                60: [],
                77: [[0,0,1,1,1,1,0,0]],
                81: [[0,0,1,1,1,1,0,0]]
                }

        for i,expected in expected_answers.items():
            G0 = gap.SpaceGroupIT(self.d,i)
            G = gap.StandardAffineCrystGroup(G0)
            N = chaincplx.gap_space_group_translation_subgroup(G,1)
            Gq = chaincplx.GapAffineQuotientGroup(G,N, scale=4)

            n=2

            space = chaincplx.trivialized_by_E3_space(self.cplx,0,0,Gq,
                        chaincplx.get_numpy_encoder_Zn(n=n))

            self.assertEqual(len(space), len(expected))
            self.assertTrue( all(numpy.array_equal(v,w) for v,w in itertools.izip(space, expected)))

            space = chaincplx.trivialized_by_E2_space(self.cplx,0,0,Gq,
                        chaincplx.get_numpy_encoder_Zn(n=n))

            self.assertEqual(len(space), len(expected))
            self.assertTrue( all(numpy.array_equal(v,w) for v,w in itertools.izip(space, expected)))

if __name__ == '__main__':
    unittest.main()
    #a = TwistGrpTests()
    #a.setUp()
    #a.test_cocycle_soln()
