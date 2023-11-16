import numpy as np
from numpy import matlib as ml

def regpoly0(S: np.ndarray):
    m, n = S.shape
    f = ml.ones([m, 1])
    df = ml.zeros([n, 1])

    return f, df

def regpoly1(S: np.ndarray):
    m, n = S.shape
    f = np.hstack([ml.ones([m, 1]), S])
    df = np.hstack([ml.zeros([n, 1]), ml.eye(n)])

    return f, df

def regpoly2(S: np.ndarray):
    m, n = S.shape
    nn = int((n + 1)*(n + 2)/2)

    # Compute f
    f = np.hstack([ml.ones([m, 1]), S, ml.zeros([m, nn-n-1])])
    j = n
    q = n
    for k in range(n):
        s = S[:, k:k+1]
        b = ml.repmat(s, 1, q)
        c = S[:, k:n]
        f[:, j+1:j+q+1] = np.multiply(b, c)

        j += q
        q -= 1
    
    df = np.hstack([ml.zeros([n, 1]), ml.eye(n), ml.zeros([n, nn-n-1])])

    j = n + 1
    q = n
    for k in range(n):
        df[k, j:j+q] = np.hstack([np.asmatrix(2*S[0, k]), np.asmatrix(S[0, k+1:n])])
        for i in range(1, n-k):
            df[k+i, j+i] = S[0, k]
        
        j += q
        q -= 1

    return f, df
