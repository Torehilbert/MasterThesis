import numpy as np
import os
import matplotlib.pyplot as plt

PTEMPLATE = r"D:\Speciale\Code\output\DeepSubspaceClustering\DSCNet Initial Training\Matrices\cm_NUM.npy"

if __name__ == "__main__":
    nums = [1,25,50,75,100]


    all_max = -500000
    all_min = 500000
    datas = []
    for i, num in enumerate(nums):
        path_file = PTEMPLATE.replace("NUM", "%d" % num)
        data = np.load(path_file)
        datas.append(data)
        if data.max() > all_max:
            all_max = data.max()
        if data.min() < all_min:
            all_min = data.min()
    
    abs_max = max(abs(all_max), abs(all_min))
    all_max = abs_max
    all_min = -abs_max

    plt.figure()
    for i, dat in enumerate(datas):
        plt.subplot(1, len(datas), i+1)
        plt.imshow(dat[:,:], vmin=all_min, vmax=all_max, cmap='bwr')
    plt.show()