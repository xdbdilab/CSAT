import pandas as pd
import numpy as np

PATH = 'Hadoop/data/'
Alg = 'ours'
sys = 'Hadoop_Terasort_CNF'

for j in range(1,4):
    log = np.zeros(3)
    for i in range(1,4):
        data = pd.read_csv(PATH + Alg + '_' + sys + '_' + str(j*100) + '_' + str(i) + ".csv")
        log[i - 1] = np.max(data['PERF'])

    print(Alg + '_' + str(j*100) + ':', np.mean(log))