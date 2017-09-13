from chaincplx.utils import *
from chaincplx.sageutils import *
from sage.all import *
gap.load_package("Cryst")
gap.SetCrystGroupDefaultAction(gap.LeftAction)

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

def affine_transformation_rescale(A,scale):
    A = copy(A)
    d = A.nrows()-1
    A[0:d,d] *= scale
    return A


#def toroidal_space_group(d,n,L):
#    G = gap.SpaceGroupIT(d,n)
#    trans = gap.translation_subgroup(G,L)
#    return GapQuotientGroup(G,trans)
