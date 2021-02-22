import os
import pandas as pd
import numpy as np
import random


DIRECTORY = r"E:\test32"
REF_DIR = r"E:\original_test_data"
SAMPLE_SIZE = 100

if __name__ == "__main__":
    path_keyfile = os.path.join(DIRECTORY, "keys.csv")
    df = pd.read_csv(path_keyfile)
    ints = np.random.randint(0, df.values.shape[0], size=SAMPLE_SIZE)

    # load info
    shapes = []
    with open(os.path.join(DIRECTORY, 'info.txt'), 'r') as f:
        for i in range(3):
            shapes.append(tuple((int(s) for s in f.readline().split(","))))
        data_dtype = f.readline()
    print("Loaded shapes: ", shapes)

    # load modes
    modes = os.listdir(REF_DIR)
    for i, mode in enumerate(modes):
        if mode == 'labels':
            del modes[i]

    # check
    for i in range(len(ints)):
        index = ints[i]
        name = df.values[index,0]
        clas = df.values[index,1]
        map_entry = df.values[index,2]

        arr = np.memmap(os.path.join(DIRECTORY, '%d.npy' % clas), dtype=data_dtype, shape=shapes[clas-1], mode='r')
        im = arr[map_entry,:,:,:]

        for j in range(len(modes)):
            im_single = np.array(np.load(os.path.join(REF_DIR, modes[j], name)), dtype=data_dtype)
            difsum = np.sum(np.abs(im[:,:,j] - im_single))
            if difsum > 0.01:
                print("ERROR: Encountered a non-matching pair!")
    
    print("Done")