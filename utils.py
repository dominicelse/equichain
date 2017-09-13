import copy

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s)+1))

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

class ElementWiseArray(tuple):
    def __new__(cls, a):
        return super(ElementWiseArray,cls).__new__(cls,a)

    def __mod__(self, other):
        assert len(self) == len(other)
        return ElementWiseArray([x % y for (x,y) in itertools.izip(self,other)])
    def __add__(self, other):
        assert len(self) == len(other)
        return ElementWiseArray([x + y for (x,y) in itertools.izip(self,other)])

class FormalIntegerSum(object):
    def __init__(self,coeffs={}):
        if not isinstance(coeffs,dict):
            self.coeffs = { coeffs : 1 }
        else:
            self.coeffs = copy.copy(coeffs)

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

