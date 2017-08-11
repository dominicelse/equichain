##cython: boundscheck=False, wraparound=False, profile=True 
import chaincplx
import numpy
from scipy import sparse
cimport numpy as np

cdef int build_index_out(int n, int ncells, int size_of_group, int ci_out, np.int_t* gi_out):
    cdef int index
    cdef int stride
    cdef int i

    index = 0
    stride = 1
    for i in xrange(n+1):
        index += stride*gi_out[i]
        stride *= size_of_group

    index += stride*ci_out
    stride *= ncells

    return index

cdef int build_index_in(int n, int ncells, int size_of_group, int ci_in, np.int_t* gi_in):
    cdef int index
    cdef int stride
    cdef int i

    index=0
    stride=1

    for i in xrange(n):
        index += stride*gi_in[i]
        stride *= size_of_group

    index += stride*ci_in
    stride *= ncells

    return index


def get_group_coboundary_matrix(cells, int n, G):
    #A = sparse.dok_matrix((indexer_out.total_dim(), indexer_in.total_dim()), dtype=int)
    #A = matrix(base_ring, indexer_out.total_dim(), indexer_in.total_dim(), sparse=True)

    mapped_cell_indices, mapping_parities = chaincplx.get_group_action_on_cells(cells,G,inverse=True)

    cdef np.int_t [:,:] mapped_cell_indices_view = mapped_cell_indices
    cdef np.int_t [:,:] mapping_parities_view = mapping_parities

    cdef int ncells = len(cells)
    cdef int size_of_group = G.size()

    gi_base = numpy.zeros((n+1,), dtype=int)
    cdef np.int_t [:] gi = gi_base

    temp_gi_base = numpy.zeros(n,dtype=int)
    cdef np.int_t [:] temp_gi = temp_gi_base

    cdef int coo_nentries = ncells*size_of_group**(n+1)*(n+2)
    coo_entries = numpy.zeros(coo_nentries,dtype=int)
    coo_i = numpy.zeros(coo_nentries, dtype=int)
    coo_j = numpy.zeros(coo_nentries, dtype=int)

    cdef np.int_t [:] coo_i_view = coo_i
    cdef np.int_t [:] coo_j_view = coo_j
    cdef np.int_t [:] coo_entries_view = coo_entries

    cdef int coo_entry_index = 0

    cdef int ci
    cdef int ii
    cdef int i
    cdef int acted_ci,parity

    cdef bint incremented

    ## BEGIN NATIVE BLOCK
    for ci in xrange(ncells):
        for ii in xrange(n+1):
            gi[ii]=0

        while True:

            ## BEGIN NON-NONNATIVE SUBBLOCK
            g = [ G.element_by_index(gii) for gii in gi ]
            ## END NON-NONNATIVE SUBBLOCK

            acted_ci = mapped_cell_indices_view[gi[0],ci]
            parity = mapping_parities_view[gi[0],ci]

            coo_entries_view[coo_entry_index] = parity
            coo_i_view[coo_entry_index] = build_index_out(n,ncells,size_of_group,ci,&gi[0])
            coo_j_view[coo_entry_index] = build_index_in(n,ncells,size_of_group,acted_ci, 
                    &gi[1] if n > 0 else NULL # Stop cython from complaining about an out of bounds error if n=0
                    )
            coo_entry_index += 1

            coo_entries_view[coo_entry_index] = (-1)**(n+1)
            coo_i_view[coo_entry_index] = build_index_out(n,ncells,size_of_group,ci,&gi[0])
            coo_j_view[coo_entry_index] = build_index_in(n,ncells,size_of_group,ci,&gi[0])
            coo_entry_index += 1

            for i in xrange(1,n+1):
                for ii in xrange(i-1):
                    temp_gi[ii] = gi[ii]
                for ii in xrange(i+1,n+1):
                    temp_gi[ii-1] = gi[ii]

                ## BEGIN NON-NATIVE SUBBLOCK
                temp_gi[i-1] = (g[i-1]*g[i]).toindex()
                ## END NON-NATIVE SUBBLOCK

                coo_entries_view[coo_entry_index] = (-1)**i
                coo_i_view[coo_entry_index] = build_index_out(n,ncells,size_of_group,ci,&gi[0])
                coo_j_view[coo_entry_index] = build_index_in(n,ncells,size_of_group,ci,&temp_gi[0])
                coo_entry_index += 1

            # Code to iterate over product of [0..size_of_group] (n+1) time
            incremented=False
            for ii in xrange(n+1):
                gi[ii] += 1
                if gi[ii] < size_of_group:
                    incremented=True
                    break
                else:
                    gi[ii] = 0
            if not incremented:
                break

    ## END NATIVE BLOCK

    indexer_out = chaincplx.MultiIndexer(*( (G.size(),) * (n+1) + (len(cells),) ))
    indexer_in = chaincplx.MultiIndexer(*( (G.size(),) * n + (len(cells),) ))

    A = sparse.coo_matrix((coo_entries,(coo_i,coo_j)), (indexer_out.total_dim(),
        indexer_in.total_dim()), dtype=int)
    return A.tocsc()
