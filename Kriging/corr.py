import numpy as np
from numpy import matlib as ml

def spline(
        theta,
        d
):
    ... # TODO #

def lin(
        theta,
        d
):
    ... # TODO #

def gauss(
        theta: np.ndarray,
        d: np.ndarray
) -> (np.ndarray, np.ndarray):
    theta = theta.flatten()
    m, n = d.shape

    if len(theta) == 1:
        theta = ml.repmat(theta, 1, n)
    elif len(theta) != n:
        raise Exception(f'Length of theta must be 1 or {d}')
    
    td = np.multiply(np.square(d), ml.repmat(-theta.reshape(1,-1), m, 1))
    r = np.asmatrix(np.exp(np.sum(td, 1))).reshape(-1,1)

    dr = np.multiply(np.multiply(ml.repmat(-2*theta.reshape(1,-1), m, 1), d), ml.repmat(r, 1, n))

    return r, dr
    
def expg(
        theta,
        d
):
    ... # TODO #

def cubic(
        theta,
        d
):
    ... # TODO #