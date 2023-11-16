import numpy as np

class Itpar():
    def __init__(self, D, ne, lo, up, perf) -> None:
        self.D = D
        self.ne = ne
        self.lo = lo
        self.up = up
        self.perf = perf
        self.nv = 1
