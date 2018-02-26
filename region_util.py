import numpy as np
from regions import read_ds9, PixCoord


class Rutil:
    def __init__(self, path):
        self.rgslist = read_ds9(path)

    def __call__(self, x, y):
            pc = PixCoord(x, y)
            result = np.zeros(len(pc), dtype=bool)
            for rg in self.rgslist:
                result = result | rg.contains(pc)
            return result
