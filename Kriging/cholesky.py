from scipy.sparse import linalg as splinalg
import scipy.sparse as sparse
import numpy as np
import sys

# TODO # Não funciona

def sparse_cholesky(A): # The input matrix A must be a sparse symmetric positive-definite.
  
  n = A.shape[0]
  LU = splinalg.splu(A,diag_pivot_thresh=0) # sparse LU decomposition
  
  if ( LU.perm_r == np.arange(n) ).all() and ( LU.U.diagonal() > 0 ).all(): # check the matrix A is positive definite.
    return LU.L.dot( sparse.diags(np.power(LU.U.diagonal(), 0.5)))
  else:
    sys.exit('The matrix is not positive definite')