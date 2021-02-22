import os
import matplotlib.pyplot as plt
import numpy as np

PATH = r"E:\full32\1.npy"
SHAPE = (80977,64,64,10)
DTYPE = 'flaot32'


np.memmap(PATH, shape=SHAPE, dtype=DTYPE, mode='r')
# under work...
