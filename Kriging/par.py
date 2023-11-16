import numpy as np

class Par():
    
    def __init__(
            self,
            corr: callable,
            regr: callable, 
            Y: np.ndarray,
            F: np.ndarray,
            D: np.ndarray,
            ij: np.ndarray,
            stdS: float
    ) -> None:
        self.corr = corr
        self.regr = regr
        self.y = Y
        self.F = F
        self.D = D
        self.ij = ij
        self.scS = stdS
