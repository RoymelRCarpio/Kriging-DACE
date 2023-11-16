import numpy as np

class Dmodel():

    def __init__(self, regr, corr, theta, fit, sY, S, mS, sS, mY) -> None:
        self.regr = regr
        self.corr = corr
        self.theta = theta.transpose()
        self.beta: np.ndarray = fit.beta
        self.gamma: np.ndarray = fit.gamma
        self.sigma2: np.ndarray = np.square(sY)*fit.sigma2
        self.S: np.ndarray = S
        self.Ssc = np.asmatrix(np.vstack([mS, sS]))
        self.Ysc = np.asmatrix(np.vstack([mY, sY]))
        self.C = fit.C
        self.Ft: np.ndarray = fit.Ft
        self.G: np.ndarray = fit.G
