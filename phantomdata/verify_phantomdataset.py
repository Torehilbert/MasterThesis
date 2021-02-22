import os
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import wfutils.infofile
import wfutils.mmap
from plotlib.style import COLORS
#from style import COLORS

if __name__ == "__main__":
    path = r"E:\Phantom_v3\train\images_DPI"

    shapes, data_dtype = wfutils.infofile.read_info(os.path.join(path, 'info.txt'))
    mmaps = wfutils.mmap.get_class_mmaps_read(path, shapes, data_dtype)

    n_channels = mmaps[0].shape[3]


    # verify interaction features
    plt.figure()
    for c in range(len(mmaps)):
        X = []
        for i in range(mmaps[c].shape[0]):
            image = mmaps[c][i,:,:,-2]
            inds = np.where(image!=0)
            pixels = image[inds[0], inds[1]]
            
            image2 = mmaps[c][i,:,:,-1]
            inds2 = np.where(image2!=0)
            pixels2 = image2[inds2[0], inds2[1]]
            X.append([pixels[0], pixels2[0]])
        X = np.array(X)
        plt.scatter(X[:,0], X[:,1], c=COLORS[c])
    

    plt.figure()
    for i in range(n_channels):
        plt.subplot(4,4,i + 1)
        plt.imshow(mmaps[0][0,:,:,i], cmap='gray')
    plt.show()
