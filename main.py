from ACTDS import ANFIS
from ACTDS import ACTDS
import numpy as np

def Test(XY):

    actds = ACTDS(XY)
    #
    print(actds.Compare_1(XY[1][0:-1], XY[2][0:-1]))

if __name__ == '__main__':
    # XY = [[1, 1, 1, 1, 0, 1, 1, 0, 1, 90],
    #       [1, 1, 1, 0, 0, 0, 0, 0, 0, 5],
    #       [0, 0, 0, 1, 1, 1, 0, 1, 0, 30],
    #       [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]
    XY = [[1, 1, 1, 1, 0, 1, 0, 0, 1, 90],
          [1, 1, 1, 1, 0, 0, 0, 0, 0, 90],
          [1, 1, 1, 0, 1, 0, 0, 0, 0, 5],
          [0, 0, 0, 0, 0, 0, 1, 1, 1, 30],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]
    Test(XY)

