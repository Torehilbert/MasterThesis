import os
import sys
import random
import matplotlib.pyplot as plt
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plotlib.style import COLORS, COLOR_NEGATIVE, COLOR_NEUTRAL, COLOR_POSITIVE
import math

DATA_PATH = r"E:\full32"
SAMPLES = 14
ROWS = 2

if __name__ == "__main__":
    modes = ['Aperture', 'ApodizedAP', 'BrightField', 'DarkField', 'DFIOpen', 'DFIPhase', 'DPI', 'iSSC', 'Phase', 'UVPhase']

    # Load info file
    shapes = []
    data_dtype = None
    with open(os.path.join(DATA_PATH, 'info.txt'), 'r') as f:
        for i in range(3):
            shapes.append(tuple((int(s) for s in f.readline().split(","))))
        data_dtype = f.readline()
    
    # Open memory-maps
    mmaps = []
    idx_lists = []
    for i in range(3):
        path_mmap = os.path.join(DATA_PATH, '%d.npy' % (i+1))
        mmaps.append(np.memmap(path_mmap, shape=shapes[i], dtype=data_dtype, mode='r'))
        idx = np.linspace(0, shapes[i][0], shapes[i][0], endpoint=False)
        random.shuffle(idx)
        idx_lists.append(idx[:ROWS*SAMPLES])


    # Extra crop limits
    res = 24
    crop_start = math.ceil((64 - res)/2)
    crop_end = res + math.floor((64-res)/2)

    # Examples of cell type 1
    ims = [np.empty(shape=(ROWS*res, SAMPLES*res, 10), dtype=float) for i in range(3)]
    for cla in range(3):
        for sample in range(SAMPLES):
            for row in range(ROWS):
                index = int(idx_lists[cla][sample + row*SAMPLES])
                rstart = row*res
                rend = rstart + res
                cstart = int((sample % SAMPLES)*res)
                cend = cstart + res
                ims[cla][rstart:rend,cstart:cend, :] = mmaps[cla][index, crop_start:crop_end, crop_start:crop_end,:]
    
    # Finding global min and max for each channel
    c1_mins = np.min(ims[0][:,:,:], axis=(0,1))
    c2_mins = np.min(ims[1][:,:,:], axis=(0,1))
    c3_mins = np.min(ims[2][:,:,:], axis=(0,1))
    c1_max = np.max(ims[0][:,:,:], axis=(0,1))
    c2_max = np.max(ims[1][:,:,:], axis=(0,1))
    c3_max = np.max(ims[2][:,:,:], axis=(0,1))
    all_mins = [min(c1_mins[i], c2_mins[i], c3_mins[i]) for i in range(10)]
    all_max = [max(c1_max[i], c2_max[i], c3_max[i]) for i in range(10)]
    
    # Construct class example plots
    if False:
        for i in range(len(modes)):
            plt.figure(figsize=(10.5,5))
            plt.subplot(3,1,1)
            plt.imshow(ims[0][:,:,i], cmap='gray', vmin=all_mins[i], vmax=all_max[i])
            plt.xticks(ticks=[])
            plt.yticks(ticks=[])
            plt.ylabel("Healthy")
            plt.title(modes[i], fontweight="bold")

            plt.subplot(3,1,2)
            plt.imshow(ims[1][:,:,i], cmap='gray', vmin=all_mins[i], vmax=all_max[i])
            plt.xticks(ticks=[])
            plt.yticks(ticks=[])
            plt.ylabel("Apoptosis")

            plt.subplot(3,1,3)
            plt.imshow(ims[2][:,:,i], cmap='gray', vmin=all_mins[i], vmax=all_max[i])
            plt.xticks(ticks=[])
            plt.yticks(ticks=[])
            plt.ylabel("Dead")

            plt.savefig("explore_samples_%s.png" % modes[i], bbox_inches = 'tight', pad_inches=0)

    # Contruct channel example plot
    if True:
        ROWS_2ND = 3
        channel_ex_data = np.empty(shape=(ROWS_2ND*res, res*10))
        for i in range(10):
            for j in range(ROWS_2ND):
                channel_ex_data[j*res:(j+1)*res,i*res:(i+1)*res] = ims[j][0:res, (j)*res:(j +1)*res, i]
        
        plt.figure(figsize=(11,5))
        plt.imshow(channel_ex_data, cmap='gray')
        plt.xticks(ticks=[res/2 + res*i for i in range(10)], labels=modes, rotation=45, fontsize=11)
        plt.yticks(ticks=[res/2 + res*i for i in range(ROWS_2ND)], labels=['Healthy', 'Apoptosis', 'Dead'], fontsize=11)
        
        plt.savefig("explore_cell_channels.png", bbox_inches = 'tight', pad_inches=0)
        plt.show()

    # RGB Experimental visualization
    # BrightField, DPI, Phase
    if False:
        RGB_CHANNELS = [3,7,2] #[2,6,8]
        ims_rgb = [np.empty(shape=(ims[i].shape[0], ims[i].shape[1], len(RGB_CHANNELS))) for i in range(3)]
        for iclass in range(3):
            for i, chan in enumerate(RGB_CHANNELS):
                ims_rgb[iclass][:,:,i] = ims[iclass][:,:,chan]

        
        for j in range(len(RGB_CHANNELS)):
            minval = []
            maxval = []
            for i in range(3):
                minval.append(np.min(ims_rgb[i][:,:,j]))
                maxval.append(np.max(ims_rgb[i][:,:,j]))
            minval = min(minval)
            maxval = max(maxval)
            for i in range(3):
                ims_rgb[i][:,:,j] = (ims_rgb[i][:,:,j] - minval)/(maxval - minval)
        
        plt.figure(figsize=(10.5,5))
        plt.subplot(3,1,1)
        plt.imshow(ims_rgb[0])#, vmin=all_mins[i], vmax=all_max[i])
        plt.xticks(ticks=[])
        plt.yticks(ticks=[])
        plt.ylabel("Healthy")
        plt.title("RGB Visualization Test", fontweight="bold")

        plt.subplot(3,1,2)
        plt.imshow(ims_rgb[1])#, cmap='gray', vmin=all_mins[i], vmax=all_max[i])
        plt.xticks(ticks=[])
        plt.yticks(ticks=[])
        plt.ylabel("Apoptosis")

        plt.subplot(3,1,3)
        plt.imshow(ims_rgb[2])#, cmap='gray', vmin=all_mins[i], vmax=all_max[i])
        plt.xticks(ticks=[])
        plt.yticks(ticks=[])
        plt.ylabel("Dead")

        plt.savefig("explore_experimental_rgb_vis.png", bbox_inches = 'tight', pad_inches=0)
