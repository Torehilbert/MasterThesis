import os
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

PATH_DATA = r"D:\Speciale\Code\output\DSC_New\Runs_3264_relu"
FOLDER_CLUSTERING = "clustering"
FILENAME_PLOT = "dendrogram_division_average.png"
INDICES = [0,5,10,15,20,25]

if __name__ == "__main__":
    subfolders = os.listdir(PATH_DATA)

    plt.figure(figsize=(12,9))
    for i,idx in enumerate(INDICES):
        folder = subfolders[idx]
        
        plt.subplot(2, 3, i+1)
        filename = os.path.join(PATH_DATA, folder, FOLDER_CLUSTERING, FILENAME_PLOT)
        image = np.array(cv.imread(filename))
        plt.imshow(image)
        plt.xticks(ticks=[])
        plt.yticks(ticks=[])
    
    plt.tight_layout()
    plt.savefig("dsc_app_dendrograms.png", dpi=400)
    plt.show()
