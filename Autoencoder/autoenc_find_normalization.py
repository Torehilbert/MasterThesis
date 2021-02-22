import os
import sys
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import wfutils.mmap 
from wfutils.progess_printer import ProgressPrinter

PATH_DATA = r"E:\full32_redist"

if __name__ == "__main__":
    # Parameters
    METHOD = 'quantile'
    QUANTILE_F = 0.01
    DATA_RANGE = [0,15000]
    X_RANGE = [16,48]
    Y_RANGE = [16,48]

    mmaps = wfutils.mmap.get_class_mmaps_read(PATH_DATA)

    vmins = np.zeros(shape=(len(mmaps), 10), dtype="float32")
    vmaxs = np.zeros(shape=(len(mmaps), 10), dtype="float32")

    if METHOD == 'minmax':
        for i,mmap in enumerate(mmaps):
            print("CLASS %d" % i)
            cursor = 0
            mmap.min(axis=(0,1,2), out=vmins[i,:])
            mmap.max(axis=(0,1,2), out=vmaxs[i,:])
    
        vmins = np.min(vmins, axis=0)
        vmaxs = np.max(vmaxs, axis=0)
        print("Min values:", vmins)
        print("Max values:", vmaxs)
    elif METHOD == 'quantile':
        for i, mmap in enumerate(mmaps):
            DATA = mmap[DATA_RANGE[0]:DATA_RANGE[1], X_RANGE[0]:X_RANGE[1], Y_RANGE[0]:Y_RANGE[1], :]
            vmins[i,:] = np.quantile(DATA, QUANTILE_F, axis=(0,1,2))
            vmaxs[i,:] = np.quantile(DATA, 1 - QUANTILE_F, axis=(0,1,2))
        
        vmins = np.min(vmins, axis=0)
        vmaxs = np.max(vmaxs, axis=0)

        print("Results (min, max):")
        for i in range(len(vmins)):
            print("  %f %f" % (vmins[i], vmaxs[i]))




