

class Pref():

    def __init__(self, theta=None, f=None, itpar=None) -> None:
        if itpar is None:
            self.pref = (theta, f, 1)
            self.nv = 1
        else:
            self.pref = itpar.perf[:, 0:itpar.nv]
            self.nv = itpar.nv
