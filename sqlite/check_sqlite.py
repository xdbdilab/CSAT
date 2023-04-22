import pandas as pd
import numpy as np

def config_fix(configuration):
    new_configuration = np.copy(configuration)
    data = pd.read_csv("sqlite/sqlite.csv")
    data_a = np.array(data)
    data_m = np.array(data)[:, 0:-1]
    params = np.array(new_configuration[0:22])
    new_configuration = data_a[np.argmin(np.dot(np.exp(params), np.exp(-data_m.transpose())) - 22)][0:len(configuration)]

    return new_configuration

if __name__ == "__main__":
    print(config_fix(np.zeros(22)))