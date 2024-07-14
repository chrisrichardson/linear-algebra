from mpi4py import MPI
from scipy.sparse import csr_matrix
import numpy as np

def local_range(N, size):
    n = N // size
    r = N % size
    ranges = []
    for rank in range(r + 1):
        ranges += [rank * (n + 1)]
    for rank in range(r + 1, size + 1):
        ranges += [rank * n + r]
    return ranges

def compute_rank_index(nglobal, size, idx):
    ranges = local_range(nglobal, size)
    w = np.searchsorted(ranges, idx, 'right') - 1
    return (w, int(idx - ranges[w]))

class DistributedCSR(object):

    def __init__(self, comm, values, indices, indptr, l2gmap, nrows_global):
        self.comm = comm
        self.A = csr_matrix((values, indices, indptr))
        self.l2gmap = l2gmap
        self.nrows_global = nrows_global

        # Pre-compute vector fetch locations from other ranks
        # FIXME: inefficient
        size = self.comm.size
        rank = self.comm.rank
        rq = [[] for i in range(size)]
        for idx in self.l2gmap:
            rmrank, rmindex = compute_rank_index(self.nrows_global, size, idx)
            if rmrank != rank:
                rq[rmrank].append(rmindex)
        self.send_indices = self.comm.alltoall(rq)


    def local_range(self):
        rank = self.comm.rank
        size = self.comm.size
        index_ranges = local_range(self.nrows_global, size)
        return index_ranges[rank], index_ranges[rank + 1]

    # Matrix-vector product (maybe we also need MatMat product?)
    def __matmul__(self, other):
        assert len(other) == self.A.shape[0]

        # Fetch/send remote values from precomputed locations
        v = [[other[i] for i in idxlist] for idxlist in self.send_indices]
        retv = self.comm.alltoall(v)

        # extend the vector with the ghost region
        vec = np.concatenate((other, np.concatenate([np.array(r) for r in retv])))

        assert len(vec) == self.A.shape[1]
        result = self.A @ vec
        return result


def create_distributed(comm, A):
    # Split matrix A across processes (assume input A is the same on processes)
    # FIXME: distribute data from rank zero instead

    # Only keep rows in the local range
    rank = comm.rank
    size = comm.size
    local_indices = local_range(A.shape[0], size)
    lr = (local_indices[rank], local_indices[rank + 1])

    ind = A.indptr[lr[0]:lr[1] + 1]
    val = A.data[ind[0]:ind[-1]]
    col = A.indices[ind[0]:ind[-1]]
    ind -= ind[0]

    # Remap columns outside local range
    local_to_global = []
    for indx in col:
        if (indx < lr[0] or indx >= lr[1]):
            local_to_global.append(indx)
    local_to_global = np.array(local_to_global)
    local_to_global = np.sort(local_to_global)
    local_to_global = np.unique(local_to_global)

    # Remap column indices to local index
    for i in range(len(col)):
        if (col[i] >= lr[0] and col[i] < lr[1]):
            col[i] -= lr[0]
        else:
            pos = np.where(local_to_global == col[i])[0][0]
            col[i] = (lr[1] - lr[0]) + pos

    return DistributedCSR(comm, val, col, ind, local_to_global, A.shape[0])
