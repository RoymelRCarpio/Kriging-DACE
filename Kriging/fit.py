import numpy as np

class Fit():
    
    def __init__(self) -> None:
        self.sigma2 = np.nan
        self.beta = np.nan
        self.gamma = np.nan
        self.C = np.nan
        self.Ft = np.nan
        self.G = np.nan
    
    def setVars(self, sigma2, beta, gamma, C, Ft, G) -> None:
        self.sigma2 = sigma2
        self.beta = beta
        self.gamma = gamma
        self.C = C
        self.Ft = Ft
        self.G = G
