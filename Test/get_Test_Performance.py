import numpy as np

def get_performance(configuration):
    x1 = configuration[0] - np.pi
    x2 = configuration[1] - np.pi

    return - x1**2 - x2**2 + 2 * (np.sin(5 * x1) + np.sin(5 * x2)) + 50