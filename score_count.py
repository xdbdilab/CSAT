import pandas as pd
import numpy as np

PATH = 'mysql/data/'
Alg = 'random'
sys = 'mysql'

for j in range(1,4):
    log = np.zeros(3)
    for i in range(1,4):
        data = pd.read_csv(PATH + Alg + '_' + sys + '_' + str(j*100) + '_' + str(i) + ".csv")
        if sys == 'mysql':
            log[i - 1] = np.min(data['PERF'])
        else:
            log[i - 1] = np.max(data['PERF'])

    print(Alg + '_' + str(j*100) + ':', np.mean(log))