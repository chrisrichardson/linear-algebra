from mpi4py import MPI
import numpy as np
from scipy.sparse import csr_matrix
from csr_distrib import local_range, create_distributed
import pyamg

# Generic version of unpreconditioned Conjugate Gradient
# with a plugin for dot_product
def cg(A, b, x, dot_product=np.dot):

    tol2 = 1e-18
    r = b - A @ x
    p = r.copy()
    rnorm2 = dot_product(r, r)
    while(rnorm2 > tol2):
        y = A @ p
        pdoty = dot_product(p, y)
        al = rnorm2/pdoty
        x += al * p
        r -= al * y
        rnorm_new = dot_product(r, r)
        beta = rnorm_new/rnorm2
        rnorm2 = rnorm_new
        p = p * beta + r


n = 60
A = pyamg.gallery.poisson((n, n, n)).tocsr()
b = np.arange(A.shape[1], dtype=np.float64)/A.shape[1]
x = np.zeros(A.shape[0])

# Solve in serial
cg(A, b, x)

# Check residual norm
r = A @ x - b
print('norm of x, r (serial) = ', np.linalg.norm(x), np.linalg.norm(r))

# Create distributed version of A
comm = MPI.COMM_WORLD
AA = create_distributed(comm, A)

# Get the local range and set RHS
bb = np.arange(*AA.local_range(), dtype=np.float64)/AA.nrows_global

# Solve
xx = np.zeros(AA.A.shape[0])
mpi_dot = lambda a, b: comm.allreduce(np.dot(a, b), MPI.SUM)
t0 = tic()
cg(AA, bb, xx, mpi_dot)
t1= toc()

# Check result
rr = AA @ xx - bb
print('norm of x, r (MPI):', np.sqrt(mpi_dot(xx, xx)), np.sqrt(mpi_dot(rr, rr)))
