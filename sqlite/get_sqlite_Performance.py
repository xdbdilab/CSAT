import csv
import random
import numpy as np
import pandas as pd
from tqdm import trange
import time

def get_performance(params):

    data = pd.read_csv("H:/FSE_2022_ACTDS/ACTDS2/sqlite/sqlite.csv")
    data_m = np.array(data)[:, 0:-1]
    params = np.array(params)
    try:
        return data['TPS'][np.where(np.dot(np.exp(params), np.exp(-data_m.transpose()))-22 == 0)[0][0]]
    except:
        return -1

if __name__ == "__main__":
    print(get_performance([0,0,1,0,0,0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0,1, 0, 1, 1, 0, 0]))