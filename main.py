from ACTDS import ANFIS
from ACTDS import ACTDS
from ACTDS import SAMPLER
import numpy as np
from sympy import *

def Test(XY):

    actds = ACTDS(XY)
    actds.Generator(np.array([0, 0, 0]))

if __name__ == '__main__':
    # XY = [[1, 1, 1, 1, 0, 1, 1, 0, 1, 90],
    #       [1, 1, 1, 0, 0, 0, 0, 0, 0, 5],
    #       [0, 0, 0, 1, 1, 1, 0, 1, 0, 30],
    #       [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]
    XY = [[1, 0, 0, 6],
          [0, 1, 1, 3],
          [0, 0, 1, 1],
          [0, 1, 0, 2],
          [1, 0, 1, 4],]
    Test(XY)

