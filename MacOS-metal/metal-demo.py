
# !pip install metalcompute

import pyamg
import numpy as np
import metalcompute as mc


def cg(A, b, x, dot_product=np.dot):

    tol2 = 1e-12
    r = b - A @ x
    p = r.copy()
    rnorm2 = dot_product(r, r)
    k = 0
    while(rnorm2 > tol2):
        k += 1
        y = A @ p
        pdoty = dot_product(p, y)
        al = rnorm2/pdoty
        x += al * p
        r -= al * y
        rnorm_new = dot_product(r, r)
        beta = rnorm_new/rnorm2
        rnorm2 = rnorm_new
        p = p * beta + r
    return k


class MetalCSR(object):

    def __init__(self, A):
        self.dev = mc.Device()
        self.A = A
        self.indptr = self.dev.buffer(self.A.indptr.astype(np.int32))
        self.indices = self.dev.buffer(self.A.indices.astype(np.int32))
        self.values = self.dev.buffer(self.A.data.astype(np.float32))

        self.kernel = self.dev.kernel("""
        // We always need these two lines in a metal kernel
        #include <metal_stdlib>
        using namespace metal;

        // This is the add function
        kernel void csr_matvec(
        // These are the two input arrays, const as we will not write to them
        const device float* vals [[ buffer(0) ]],
        const device int* cols [[ buffer(1) ]],
        const device int* rowptr [[ buffer(2) ]],
        const device float* x [[ buffer(3)]],
        // This is the output array
        device float *y [[ buffer(4) ]],
        // This is the index of the current kernel instance
        uint id [[ thread_position_in_grid ]])
        {
          for (int i = rowptr[id]; i < rowptr[id + 1]; ++i)
            y[id] += vals[i] * x[cols[i]];
        }
        """)
        self.csr_matvec_fn = self.kernel.function("csr_matvec")


    def __matmul__(self, vec):
        vecarr = self.dev.buffer(vec.astype(np.float32))
        result = self.dev.buffer(self.A.shape[0] * 4)
        h = self.csr_matvec_fn(self.A.shape[0], self.values, self.indices, self.indptr,
                               vecarr, result)
        del h
        return np.frombuffer(result, dtype=np.float32)



n = 50
A = pyamg.gallery.poisson((n, n, n)).tocsr()
A.data = A.data.astype(np.float32)
x = np.ones(A.shape[1], dtype=np.float32)
y = np.zeros(A.shape[0], dtype=np.float32)

y = A @ x
print(np.linalg.norm(y))

y.fill(0.0)
AA = MetalCSR(A)
y = AA @ x
print(np.linalg.norm(y))

x = np.zeros(A.shape[0], dtype=np.float32)
b = np.ones(A.shape[1], dtype=np.float32)
k = cg(A, b, x)
print('Solution norm, its = ', np.linalg.norm(x), k)

x.fill(0.0)
k = cg(AA, b, x)
print('Solution norm, its = ', np.linalg.norm(x), k)
