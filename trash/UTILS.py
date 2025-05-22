import numpy as np
from functools import lru_cache


from scipy.sparse.linalg import eigsh
def is_positive_definite_sparse(A):
    # Sprawdź czy A jest kwadratowa
    if A.shape[0] != A.shape[1]:
        return False
    try:
        # oblicz najmniejszą wartość własną
        min_eigval = eigsh(A, k=1, which='SA', return_eigenvectors=False)[0]
        return min_eigval > 0
    except:
        return False