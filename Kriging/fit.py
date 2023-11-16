import numpy as np

class Fit():
    
    def __init__(self) -> None:
        self.sigma2 = np.NaN
        self.beta = np.NaN
        self.gamma = np.NaN
        self.C = np.NaN
        self.Ft = np.NaN
        self.G = np.NaN
    
    def setVars(self, sigma2, beta, gamma, C, Ft, G) -> None:
        self.sigma2 = sigma2
        self.beta = beta
        self.gamma = gamma
        self.C = C
        self.Ft = Ft
        self.G = G
