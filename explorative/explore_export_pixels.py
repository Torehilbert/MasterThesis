import os
import sys
import random
import argparse
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import wfutils.log


parser = argparse.ArgumentParser()
parser.add_argument('-src', required=False, type=str, default=r"E:\full32")
parser.add_argument('-samples', required=False, type=int, default=10)  # 13932 is the num samples of class 2 which has the fewest samples
parser.add_argument('-seed', required=False, type=int, default=-1)



if __name__ == "__main__":
    args = parser.parse_args()
    path_output_folder = wfutils.log.create_output_folder("PixelData")
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

    # Extract pixels from data
    class_pixels = []
    n_pixels = []
    for i in range(3):
        extract = mmaps[i][idx_lists[i]]
        indices = np.nonzero(extract[:,:,:,0])
        pixels = extract[indices[0], indices[1], indices[2], :]
        class_pixels.append(pixels)
        n_pixels.append(pixels.shape[0])

    class_index_list = np.ones(shape=(sum(n_pixels),1))
    class_index_list[n_pixels[0]:(n_pixels[0] + n_pixels[1])] = 2
    class_index_list[(n_pixels[0] + n_pixels[1]):] = 3
    
    pixels_data = np.concatenate((class_index_list, np.vstack(class_pixels)), axis=1)
            
    # Save to CSV
    header_names = ["class"]
    for mode in modes:
        header_names.append("%s" % mode)
    df = pd.DataFrame(data=pixels_data, columns=header_names)
    df.to_csv(os.path.join(path_output_folder, 'data.csv'), index=False)
