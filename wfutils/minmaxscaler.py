import os
import numpy as np


class MinMaxScaler:
    def __init__(self, path_file):
        if path_file is None:
            self.vmin = None
            self.vmax = None
        else:
            if not os.path.isfile(path_file):
                raise Exception("No file %s exists!" % path_file)
            
            # read file
            mins = []
            maxs = []
            with open(path_file, 'r') as f:
                for l in f.readlines():
                    splits = l.split(" ")
                    mins.append(float(splits[0]))
                    maxs.append(float(splits[1]))
            
            self.vmin = np.array(mins)
            self.vmax = np.array(maxs)


    def scale(self, x):
        return (x-self.vmin) / (self.vmax - self.vmin)


if __name__ == "__main__":
    scaler = MinMaxScaler(r"E:\normalization_values\quantile_p1_crop.txt")