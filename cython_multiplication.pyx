from cython.parallel import prange
import numpy as np
cimport numpy as np
import cython
cimport openmp

# Musimy zainicjować typy NumPy
np.import_array()

@cython.boundscheck(False)
@cython.wraparound(False)
def cython_multiply(np.ndarray[np.int64_t, ndim=1] v):
    cdef Py_ssize_t i
    cdef Py_ssize_t n = v.shape[0]
    cdef np.ndarray[np.int64_t, ndim=1] result = np.empty(n, dtype=np.int64)
    
    return v*v

@cython.boundscheck(False)
@cython.wraparound(False)
def cython_multiply_parallel(np.ndarray[np.int64_t, ndim=1] v):
    cdef Py_ssize_t i
    cdef Py_ssize_t n = v.shape[0]
    cdef np.ndarray[np.int64_t, ndim=1] result = np.empty(n, dtype=np.int64)
    cdef int current_thread
    cdef int m

    with nogil:
        for i in prange(n, schedule='static'):
            
            result[i] = v[i] * v[i]
            with gil:
                m=openmp.omp_get_num_threads()
                print("m: ")
                print(m)
            # Print thread ID only at the start of each chunk
            
            with gil:
                print("Thread ID: ")
                print(cython.parallel.threadid())
    return result, m