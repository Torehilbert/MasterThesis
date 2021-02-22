import os
import sys
import random
import argparse
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import wfutils.log


parser = argparse.ArgumentParser()
parser.add_argument('-src', required=False, type=str, default=r"E:\validate32")
parser.add_argument('-samples', required=False, type=int, default=2847)  # max 13932 for training set, max 2847 for validation set 
parser.add_argument('-seed', required=False, type=int, default=-1)



if __name__ == "__main__":
    args = parser.parse_args()
    path_output_folder = wfutils.log.create_output_folder("ExploreSimpleStats")
    wfutils.log.log_arguments(path_output_folder, args)

    # Read info file
    shapes = []
    data_dtype = None
    with open(os.path.join(args.src, 'info.txt'), 'r') as f:
        for i in range(3):
            shapes.append(tuple((int(s) for s in f.readline().split(","))))
        data_dtype = f.readline()

    # Read mode order
    modes = []
    with open(os.path.join(args.src, 'channel_order.txt'), 'r') as f:
        modes = [mode.replace("\n", "") for mode in f.readlines()]

    # Error checks
    ns = np.array([shapes[0][0], shapes[1][0], shapes[2][0]])
    if args.samples > np.min(ns):     
        raise Exception("Specified to pull %d samples but class %d only has %d samples available" % (args.samples, np.argmin(ns), np.min(ns)))
    
    if len(modes) != 10:
        raise Exception("Wrong number of modes, got %d modes, expected %d." % (len(modes), 10))

    # Create mmap
    mmaps = [np.memmap(os.path.join(args.src, "%d.npy" % (i+1)), dtype=data_dtype, shape=shapes[i], mode='r') for i in range(3)]
    idx_lists = [list(range(0, ns[i])) for i in range(3)]
    for i in range(3):
        random.shuffle(idx_lists[i])
        idx_lists[i] = np.array(idx_lists[i])
        idx_lists[i] = idx_lists[i][0:args.samples]

    # Extract stats from data
    n_stats = 3
    data_arr = np.empty(shape=(3 * args.samples, 1 + n_stats*len(modes)), dtype=float)

    data_cursor = 0
    for i in range(3):
        extract = mmaps[i][idx_lists[i]]
        extract_masked = np.ma.masked_array(extract, mask=extract==0)
        means = extract_masked.mean(axis=(1,2))
        stds = extract_masked.std(axis=(1,2))
        nums = np.sum(extract!=0, axis=(1,2))

        istart = data_cursor
        iend = data_cursor+args.samples
        data_arr[istart:iend, 0] = (i+1) * np.ones(shape=(args.samples,)) 

        for j in range(len(modes)):
            data_arr[istart:iend, 1 + j*n_stats] = means[:,j]
            data_arr[istart:iend, 2 + j*n_stats] = stds[:,j]
            data_arr[istart:iend, 3 + j*n_stats] = nums[:,j]
        data_cursor += args.samples
            
    # Save to CSV
    header_names = ["class"]
    for mode in modes:
        header_names.append("mean (%s)" % mode)
        header_names.append("std (%s)" % mode)
        header_names.append("n (%s)" % mode)
    df = pd.DataFrame(data=data_arr, columns=header_names)
    df.to_csv(os.path.join(path_output_folder, 'data.csv'), index=False)
