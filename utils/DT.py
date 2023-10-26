import numpy as np

def I(v):
    return -np.sum(v*np.log2(v))
