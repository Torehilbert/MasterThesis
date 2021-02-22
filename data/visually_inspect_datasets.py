import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import wfutils.mmap


#path_dset_primary = r"E:\full32_redist"
#path_dset_secondary = r"E:\full32_redist_crop_rescale" 

path_dset_primary = r"E:\validate32_redist"
path_dset_secondary = r"E:\validate32_redist_crop_rescale" 

if __name__ == "__main__":
    mmaps_primary = wfutils.mmap.get_class_mmaps_read(path_dset_primary)
    mmaps_secondary = wfutils.mmap.get_class_mmaps_read(path_dset_secondary)

    plt.figure()
    for i,(mmap_p, mmap_s) in enumerate(zip(mmaps_primary, mmaps_secondary)):
        for ch in range(10):
            plt.subplot(10,2,2*ch+1)
            plt.imshow(mmap_p[43,:,:,ch], cmap='gray')
            plt.subplot(10,2,2*ch+2)
            plt.imshow(mmap_s[43,:,:,ch], cmap='gray', vmin=0, vmax=1)
        plt.show()
          