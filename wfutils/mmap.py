import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import wfutils.infofile

def get_class_mmaps_read(path, shapes=None, dtype=None):
    if shapes is None or dtype is None:
        path_infofile = wfutils.infofile.get_path_to_infofile(path)
        shapes, dtype = wfutils.infofile.read_info(path_infofile)

    mmaps = []
    for i in range(3):
        path_mmap = os.path.join(path, "%d.npy" % (i+1))
        mmaps.append(np.memmap(path_mmap, shape=shapes[i], dtype=dtype, mode='r'))
    return mmaps


def get_class_mmaps_write(path, shapes, dtype):
    mmaps = []
    for i in range(3):
        path_mmap = os.path.join(path, "%d.npy" % (i+1))
        mmaps.append(np.memmap(path_mmap, shape=shapes[i], dtype=dtype, mode='w+'))
    return mmaps

