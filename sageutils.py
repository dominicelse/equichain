from sage.all import *

class NotIntegerMatrixError(Exception):
    pass


def sage_matrix_to_numpy_int(A):
    if not A in MatrixSpace(ZZ,A.nrows()):
        print A
        raise NotIntegerMatrixError

    return matrix(ZZ,A).numpy(dtype=int)

def sage_vector_to_numpy_int(v):
    if not v in FreeModule(ZZ,len(v)):
        raise ValueError, "Not an integer vector." + str(v)

    return vector(ZZ,v).numpy(dtype=int)

def fracpart(x):
    return x-sage.all.floor(x)
