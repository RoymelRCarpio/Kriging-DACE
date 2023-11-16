from numpy import matlib as ml
from colorama import Fore
import numpy as np

from .dmodel import Dmodel

def predictor(
        x: np.ndarray,
        dmodel: Dmodel
):
    """_summary_

    Args:
        x (np.ndarray): _description_
        dmodel (Dmodel): _description_

    Raises:
        Exception: _description_
        Exception: _description_

    Returns:
        _type_: _description_
    """
    assert isinstance(x, np.ndarray)
    assert isinstance(dmodel, Dmodel)

    or1 = np.NaN
    or2 = np.NaN
    dmse = np.NaN

    if np.any(np.isnan(dmodel.beta)):
        raise Exception('DMODEL has not been found')
    
    m, n = dmodel.S.shape
    sx = x.shape

    if (np.min(sx) == 1) and n > 1:
        nx = int(np.max(sx))

        if nx == n:
            mx = 1
            x = x.reshape(-1,1).transpose()
    else:
        mx = sx[0]
        nx = sx[1]
        
    if nx != n:
        raise Exception(f"Dimension of trial sites should be {n}")
        
    # Normalize trial sites
    x = (x - ml.repmat(dmodel.Ssc[0, :], mx, 1)) / ml.repmat(dmodel.Ssc[1, :], mx, 1)
    _, q = dmodel.Ysc.shape
    y = ml.zeros([mx, q])

    # One site only
    if mx == 1:
        dx = ml.repmat(x, m, 1) - dmodel.S

        # Gradient/Jacobian wanted
        f, df = dmodel.regr(x)
        r, dr = dmodel.corr(dmodel.theta, dx)

        # Scaled Jacobian
        if isinstance(dmodel.beta, float):
            dy = (df * dmodel.beta).transpose() + (dmodel.gamma @ dr)
        else:
            dy = (df @ dmodel.beta).transpose() + (dmodel.gamma @ dr)

        # Unscaled Jacobian
        or1 = np.multiply(dy, ml.repmat(np.asmatrix(dmodel.Ysc[1, :]).conj().transpose(), 1, nx)) / ml.repmat(dmodel.Ssc[1, :], q, 1)

        if q == 1:
            or1 = or1.conj().transpose()
        
        # MSE wanted
        rt, _, _, _ = np.linalg.lstsq(dmodel.C, r, rcond=None)
        u = dmodel.Ft.transpose() @ rt - f.transpose()
        
        if isinstance(dmodel.G, float):
            v = u / dmodel.G
        else:
            v, _, _, _ = np.linalg.lstsq(dmodel.G, u, rcond=None)
        
        or2 = ml.repmat(dmodel.sigma2, mx, 1) * ml.repmat((1 + np.sum(np.square(v)) - np.sum(np.square(rt)).conj().transpose()), 1, q)
    
        # Gradient/Jacobian of MSE wanted
        # Scaled gradient as a row vector
        if isinstance(dmodel.G, float):
            Gv = v / dmodel.G
        else:
            Gv, _, _, _ = np.linalg.lstsq(dmodel.G.conj().transpose(), v, rcond=None)
        g1 = (dmodel.Ft @ Gv - rt).conj().transpose()
        g2, _, _, _ = np.linalg.lstsq(dmodel.C, dr, rcond=None)
        g3 = (df @ Gv).conj().transpose()
        g = g1 @ g2 - g3
        # Unscaled Jacobian
        dmse = np.multiply(ml.repmat(2 * np.asmatrix(dmodel.sigma2).conj().transpose(), 1, nx), ml.repmat(g / dmodel.Ssc[1, :], q, 1))

        if q == 1:
            # Gradient as a column vector
            dmse = dmse.conj().transpose()

        # Scaled predictor
        if isinstance(dmodel.beta, float):
            sy = f * dmodel.beta + (dmodel.gamma @ r).transpose()
        else:
            sy = f @ dmodel.beta + (dmodel.gamma @ r).transpose()
        # Predictor
        y = (np.asmatrix(dmodel.Ysc[0, :]) + np.multiply(np.asmatrix(dmodel.Ysc[1, :]), sy)).conj().transpose()
    
    else: # several trial sites
        # Get distances to design sites
        dx = ml.zeros([mx*m, n])
        kk = 0
        for k in range(mx):
            dx[kk:kk + m, :] = ml.repmat(x[k, :], m, 1) - dmodel.S
            kk += m

        # Get regression function and correlation
        f, _ = dmodel.regr(x)
        r, _ = dmodel.corr(dmodel.theta, dx)
        r = np.hstack(np.split(r, mx, axis=0))

        # Scaled predictor
        sy = f @ dmodel.beta + (dmodel.gamma @ r).transpose()
        # Predictor
        y1 = ml.repmat(dmodel.Ysc[0, :], mx, 1)
        y2 = np.multiply(ml.repmat(dmodel.Ysc[1, :], mx, 1), sy)
        y = y1 + y2

        # MSE wanted
        rt, _, _, _ = np.linalg.lstsq(dmodel.C, r, rcond=None)
        u1 = dmodel.Ft.transpose() * rt - f.transpose()
        u, _, _, _ = np.linalg.lstsq(dmodel.G, u1, rcond=None)
        or1_1 = (1 + colsum(np.square(u)) - colsum(np.square(rt))).conj().transpose()
        part1 = np.array(ml.repmat(dmodel.sigma2, mx, 1))
        part2 = ml.repmat(or1_1, 1, q)
        or1 = part1 * part2
        print(Fore.RED + 'WARNING from PREDICTOR.  Only  y  and  or1=mse  are computed' + Fore.RESET)

    ## TODO ## or1 esta dando valor errado
    y = np.asarray(y)
    return y, or1, or2, dmse

def colsum(x: np.ndarray):

    if x.shape[0] == 1:
        return x
    else:
        return np.sum(x)