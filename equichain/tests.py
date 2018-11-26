import unittest
from sage.all import *
import equichain
import equichain.linalg
import equichain.resolutions
import numpy
import itertools
import cProfile

class ComplexWithGroupActionTestIntegerEquivalence(object):
    def test_integer_equivalence(self):
        cplx1 = self.make_complex(equichain.PointInUniverse)
        cplx2 = self.make_complex(equichain.IntegerPointInUniverse)

        for k in xrange(len(cplx1.cells)):
            act1 = cplx1.get_group_action_on_cells(self.G,k,inverse=True)
            act2 = cplx2.get_group_action_on_cells(self.G,k,inverse=True)
            self.assertTrue(numpy.array_equal(act1[0],act2[0]))
            self.assertTrue(numpy.array_equal(act1[1],act2[1]))

        for k in xrange(1,len(cplx1.cells)):
            D1 = cplx1.get_boundary_matrix(k)
            D2 = cplx2.get_boundary_matrix(k)
            self.assertTrue( (D1!=D2).nnz == 0)

class GroupCohomologyTests(unittest.TestCase):
    def setUp(self):
        gap.load_package("hap")

    def test_group_cohomology(self):
        for i in range(2,18):
            G = gap.SpaceGroupIT(2,i)

            expected_answer=list(gap.GroupCohomology(gap.PointGroup(G),4))

            Gq = equichain.GapAffineQuotientGroup(G,equichain.gap_space_group_translation_subgroup(G,1))
            resolution = equichain.resolutions.HapResolution(
                    gap.ResolutionFiniteGroup(Gq.gap_quotient_grp, 5),
                    Gq)
            answer=equichain.group_cohomology(Gq, 4,
                    resolution=resolution,
                    ring=ZZ)

            self.assertEqual(answer,expected_answer)

class SpaceGroupTests(unittest.TestCase):
    def setUp(self):
        self.d = 3
        self.cplx = equichain.cubical_complex(self.d, [1]*self.d, [], with_midpoints=True, scale=2,
                pointclass=equichain.IntegerPointInUniverse)
        gap.load_package("hap")

    def test_somethings(self):
        expected_answers = {
                2: None,
                10: None,
                11: None,
                33: [[1,1,1,1,1,1,1,1]],
                60: None,
                77: [[0,0,1,1,1,1,0,0]],
                81: [[0,0,1,1,1,1,0,0]]
                }

        for i,expected in expected_answers.items():
            G0 = gap.SpaceGroupIT(self.d,i)
            G = gap.StandardAffineCrystGroup(G0)
            #N = equichain.gap_space_group_translation_subgroup(G,1)
            N = None
            Gq = equichain.GapAffineQuotientGroup(G,N, scale=4)

            n=2

            resolution = equichain.resolutions.HapResolution(
                    gap.ResolutionFiniteGroup(Gq.gap_quotient_grp, 3),
                    Gq)

            space = equichain.trivialized_by_E3_space(self.cplx,0,0,Gq,
                        ring=Integers(n),
                        twist=None,
                        resolution=resolution)

            #self.assertEqual(len(space), len(expected))
            #self.assertTrue( all(numpy.array_equal(v,w) for v,w in itertools.izip(space, expected)))

            space = equichain.trivialized_by_E2_space(self.cplx,0,0,Gq,
                    ring=Integers(n),
                    twist=None,
                        resolution=resolution)

            if expected is None:
                self.assertTrue(space.ncols() == 0)
            else:
                self.assertTrue( numpy.array_equal(space.to_numpydense().A.T, expected) )


if __name__ == '__main__':
    #unittest.main()
    cProfile.run('unittest.main()', filename='profile.txt')
    #a = TwistGrpTests()
    #a.setUp()
    #a.test_cocycle_soln()
