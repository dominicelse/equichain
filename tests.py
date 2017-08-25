import unittest
from sage.all import *
import chaincplx
import numpy

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
                    with_midpoints=True, pointclass=self.pointclass)
            self.R = Integers(2)
        else:
            raise ValueError, "not valid chain complex description"

    def test_cython(self):
        for k in xrange(len(self.cplx.cells)):
            for n in xrange(2):
                D1 = self.cplx.get_group_coboundary_matrix(n=n,k=k,G=self.G,
                        use_cython=True)
                D2 = self.cplx.get_group_coboundary_matrix(n=n,k=k,G=self.G,use_cython=False)
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
        soln2 = chaincplx.solve_matrix_equation(A2, soln_boundary, over_ring=self.R)

        D2 = self.cplx.get_boundary_matrix(1)
        u = D2.dot(soln2) % 2

        self.assertTrue(numpy.array_equal(u,[0,0]))

    def test_E2(self):
        a = numpy.array([1,1])
        self.assertFalse(
                chaincplx.test_has_solution(lambda:chaincplx.find_E2_trivializer(self.cplx,a,n=0,k=0,G=self.G,over_ring=self.R))
                )

    def test_E3(self):
        a = numpy.array([1,1])
        self.assertFalse(
                chaincplx.test_has_solution(lambda:chaincplx.find_E3_trivializer(self.cplx,a,n=0,k=0,G=self.G,over_ring=self.R))
                )

if __name__ == '__main__':
    unittest.main()
    #a = TwistGrpTests()
    #a.setUp()
    #a.test_cocycle_soln()
