from numpy import matlib as ml
import scipy.sparse as sp
import numpy as np
import scipy as sci

from .dmodel import Dmodel
from .itpar import Itpar
from .pref import Pref
from .par import Par
from .fit import Fit


def dacefit(
        S: np.ndarray,
        Y: np.ndarray,
        theta0: np.ndarray,
        regr: callable,
        corr: callable,
        lob: list[float] = [],
        upb: list[float] = [],
) -> (Dmodel, Pref):
    assert isinstance(S, np.ndarray)
    assert len(S.shape) == 2
    assert isinstance(Y, np.ndarray)
    assert len(Y.shape) == 2
    assert isinstance(theta0, np.ndarray)
    assert callable(regr)
    assert callable(corr)
    assert isinstance(lob, list)
    assert isinstance(upb, list)
    
    # Check design points
    m , n = S.shape
    stdY = Y.shape

    if min(stdY) == 1:
        lY = max(stdY)
    else:
        lY = stdY[0]
    
    if m != lY:
        raise Exception('S and Y must have the same number of rows')

    # Check correlation parameters
    lth = len(theta0)
    if (lob != []) and (upb != []):
        if (len(lob) != lth) or (len(upb) != lth):
            raise Exception('theta0, lob and upb must have the same length')
        
        for lo, up in zip(lob, upb):
            if (lo <= 0) or (up < lo):
                raise Exception('The bounds must satisfy  0 < lob <= upb')
    
    else:
        if any(num <= 0 for num in theta0):
            raise Exception('theta0 must be strictly positive')

    # Normalize data
    mS = S.mean(axis=0);  Sstd = S.std(axis=0, ddof=1)
    mY = Y.mean(axis=0);  stdY = Y.std(axis=0, ddof=1)

    j = np.where(Sstd == 0)[0]
    if np.any(j):
        Sstd[j[0], j[1]] = 1

    j = np.where(stdY == 0)[0]
    if not any(j) == 0:
        stdY[j[0], j[1]] = 1

    S = (S - ml.repmat(mS, m, 1)) / ml.repmat(Sstd, m, 1)
    Y = (Y - ml.repmat(mY, m, 1)) / ml.repmat(stdY, m, 1)

    # Calculate distances D between points
    mzmax = int(m*(m - 1)/2)
    ij = ml.zeros([mzmax, 2])
    D = ml.zeros([mzmax, n])
    ll = [-1]
    for k in range(1, m):
        ll = ll[-1] + np.array(range(1, m - k + 1))
        ij[ll[0]:ll[-1] + 1] = np.array([[k - 1, val - 1] for val in range(k+1, m+1)])
        D[ll[0]:ll[-1] + 1] = ml.repmat(S[k-1], m-k, 1) - S[k:m]
    
    if min(np.sum(abs(D), axis=1)) == 0:
        raise Exception('Multiple design sites are not allowed')
    
    # Regression matrix
    F, _ = regr(S)
    mF, p = F.shape
    if mF != m:
        raise Exception('number of rows in  F  and  S  do not match')
    if p > mF:
        raise Exception('least squares problem is underdetermined')

    # parameters for objective function
    par = Par(corr, regr, Y, F, D, ij, Sstd)

    # Determine theta
    perf = None
    if (lob != []) and (upb != []):
        # Bound constrained non-linear optimization
        theta, f, fit, perf = boxmin(theta0, lob, upb, par)
        
        if np.isinf(f):
            raise Exception('Bad parameter region.  Try increasing  upb')
    else:
        # Given theta
        theta = theta0.reshape(-1, 1)
        f, fit = objfunc(theta, par)
        perf = Pref(theta, f)

        if np.isinf(f):
            raise Exception('Bad point.  Try increasing theta0')
    
    dmodel = Dmodel(regr, corr, theta, fit, stdY, S, mS, Sstd, mY)

    return dmodel, perf

def boxmin(
        t0: list[float],
        lo: list[float],
        up: list[float],
        par: Par
):
    # Initialize
    t, f, fit, itpar = start(t0, lo, up, par)

    if not np.isinf(f):
        # Iterate
        p = len(t)
        if p <= 2:
            kmax = 2
        else:
            kmax = np.min((p, 4))
        
        for _ in range(kmax):
            th = t
            t, f, fit, itpar = explore(t, f, fit, itpar, par)
            t, f, fit, itpar = move(th, t, f, fit, itpar, par)
    
    perf = Pref(itpar=itpar)

    return t, f, fit, perf

def start(
        t0: np.ndarray,
        lo: list[float],
        up: list[float],
        par: Par
):
    t = t0.reshape(-1, 1)
    lo = np.array(lo).reshape(-1, 1)
    up = np.array(up).reshape(-1, 1)
    p = len(t)
    D = np.power(2, (np.array([[i] for i in range(1, p+1)])/(p + 2)))
    ee = np.where(up == lo)
    if np.any(ee):
        for ix, iy in zip(ee[0], ee[1]):
            D[ix, iy] = 1
            t[ix, iy] = up[ix, iy]
        
    ng = np.where((t < lo) | (up < t))
    if np.any(ng):
        for ix, iy in zip(ng[0], ng[1]):
            t[ix, iy] = np.power(np.power(lo[ix, iy]*up[ix, iy], 7), 1/8)
    
    ne = np.where(D != 1)
    f, fit = objfunc(t, par)
    nv = 1
    itpar = Itpar(
        D = D,
        ne = ne,
        lo = lo,
        up = up,
        perf = ml.zeros((p+2, 200*p)),
    )
    itpar.perf[:, 0] = np.append(t, [[f], [1]] , axis=0)

    if np.isinf(f):
        return t, f, fit, itpar
    
    if np.any(ng):
        d0 = 16
        d1 = 2
        q = len(ng)
        th = t.copy()
        fh = f
        jdom = ng[0]

        for k in range(q):
            line, colu = ng[k]
            fk = fh
            tk = th.copy()
            DD = ml.ones((p, 1))
            for ix, iy in ng:
                DD[ix, iy] = ml.repmat(1/d1, q, 1)
            DD[line, colu] = 1/d0
            ... # TODO # linha 207

    itpar.nv = nv
    return t, f, fit, itpar

def explore(
        t: np.ndarray,
        f: float,
        fit: Fit,
        itpar: Itpar,
        par: Par
):
    # Explore step
    nv = itpar.nv
    ne = itpar.ne

    for ix, iy in zip(ne[0], ne[1]):
        tt = t.copy().astype(float)
        DD = itpar.D[ix, iy]

        if t[ix, iy] == itpar.up[ix, iy]:
            atbd = 1
            tt[ix, iy] = t[ix, iy]/np.sqrt(DD)
        elif t[ix, iy] == itpar.lo[ix, iy]:
            atbd = 1
            tt[ix, iy] = t[ix, iy]*np.sqrt(DD)
        else:
            atbd = 0
            tt[ix, iy] = np.min((itpar.up[ix, iy], t[ix, iy]*DD))
        
        ff, fitt = objfunc(tt, par)
        nv += 1
        itpar.perf[:, nv - 1] = np.append(tt, [[ff], [2]], axis=0)

        if ff < f: # TODO TODO # Revisar os valores de f e ff
            t = tt
            f = ff
            fit = fitt
        else:
            itpar.perf[-1, nv - 1] = -2

            if not atbd:
                tt[ix, iy] = np.max((itpar.lo[ix, iy], t[ix, iy]/DD))
                ff, fitt = objfunc(tt, par)
                nv += 1
                itpar.perf[:, nv - 1] = np.append(tt, [[ff], [2]], axis=0)

                if ff < f: # TODO TODO # f da objfunc sempre da 0.83333...
                    t = tt
                    f = ff
                    fit = fitt
                else:
                    itpar.perf[-1, nv - 1] = -2
        
        itpar.nv = nv

    return t, f, fit, itpar

def move(
        th: np.ndarray,
        t: np.ndarray,
        f: float,
        fit: Fit,
        itpar: Itpar,
        par: Par
):
    # Pattern move
    nv = itpar.nv
    ne = itpar.ne
    p = len(t)
    v = t/th

    if np.all(v == 1):
        m, n = itpar.D.shape
        ax = ml.zeros((m, n))
        itpar.D = np.power(itpar.D, 0.2)
        itpar.D = np.vstack((itpar.D[1:], itpar.D[0]))

        return t, f, fit, itpar
    
    # Proper move
    rept = 1
    while rept:
        tt = np.min((itpar.up, np.max((itpar.lo, t*v), axis=0)), axis=0)
        ff, fitt = objfunc(tt, par)
        nv += 1
        itpar.perf[:, nv - 1] = np.append(tt, [[ff], [3]], axis=0)
        
        if ff < f:
            t = tt
            f = ff
            fit = fitt
            v = np.square(v)
        else:
            itpar.perf[-1, nv -1] = -3
            rept = 0
        
        if np.any((tt == itpar.lo) | (tt == itpar.up)):
            rept = 0
    
    itpar.nv = nv

    for i, line in enumerate(itpar.D):
        if i == 0:
            fistLine = line.copy()
            itpar.D[0] = itpar.D[1]
        elif i == (len(itpar.D) - 1):
            itpar.D[i] = fistLine
        else:
            itpar.D[i] = itpar.D[i + 1]
        
    itpar.D = np.power(itpar.D, 0.25)
    
    return t, f, fit, itpar

def objfunc(theta: np.ndarray, par: Par) -> (np.array, Fit):
    # Initialize
    obj = np.inf
    fit = Fit()
    m = par.F.shape[0]

    # Set up R
    r, _ = par.corr(theta, par.D)
    ind = np.where(r > 0)
    o = np.array([i for i in range(m)])
    mu = (10 + m)*np.finfo(float).eps
    data = np.array([])
    row_id = np.array([])
    col_id = np.array([])
    for ix, iy in zip(ind[0], ind[1]):
        data = np.append(data, [r[ix, iy]])
        row_id = np.append(row_id, [par.ij[ix, 0]])
        col_id = np.append(col_id, [par.ij[ix, 1]])
    data = np.hstack([data, np.array((ml.ones((1, m)) + mu)).flatten()])
    row_id = np.hstack([row_id, o]).astype(int)
    col_id = np.hstack([col_id, o]).astype(int)
    R = sp.csr_matrix((data, (row_id, col_id)))
    
    try:
        C, _ = sci.linalg.cho_factor(R.toarray())
    except Exception:
        return obj, fit

    C = C.conj().transpose()
    Ft, _, _, _ = np.linalg.lstsq(C, par.F, rcond=None)
    Q, G = np.linalg.qr(Ft)

    if (1/np.linalg.cond(G)) < 1e-10:
        
        # Check F
        if np.linalg.cond(par.F) > 1e15:
            raise Exception('F is too ill conditioned\nPoor combination of regression model and design sites')
    
    Yt, _, _, _ = np.linalg.lstsq(C, par.y, rcond=None)
    Q_tc = Q.conj().transpose()
    mult = Q_tc.astype(np.double) @ np.asmatrix(Yt).astype(np.double)
    beta, _, _, _ = np.linalg.lstsq(G.astype(np.double), mult.astype(np.double), rcond=None) # TODO # "Q.conj().transpose() @ Yt" esta dando resultado errado
    rho = Yt - (Ft @ beta)
    sigma2 = np.sum(np.square(rho))/m
    detR = np.power(np.diag(C), 2/m).prod()
    obj = np.sum(sigma2) * detR

    fit.setVars(
        sigma2=sigma2,
        beta=beta, # Esta errado TODO
        gamma= rho.conj().transpose() @ np.linalg.pinv(C), #np.linalg.lstsq(C.transpose(), rho)[0].transpose()
        C=C,
        Ft=Ft,
        G=G.conj().transpose()
    )

    return obj, fit
