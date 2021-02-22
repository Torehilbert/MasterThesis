import cv2 as cv
import matplotlib.pyplot as plt
import os
import numpy as np


PATH_IMAGES_FOLDER=r"D:\Speciale\Data\Dummy_dataset\Xcyto10-4x\original_images\images"
IMAGE_NAME = "20200702-00002_1.tiff"

USE_SUBPLOTS = False

CROP_SIZE = (300,100)  # (rows, columns)
CROP_ORIGO = (100, 500)   # (rows, columns)
CROP_STACK = (1,10)  # (rows, coluns)


if __name__ == "__main__":
    image_folders = os.listdir(PATH_IMAGES_FOLDER)

    arr = np.empty(shape=(CROP_SIZE[0]*CROP_STACK[0], CROP_SIZE[1]*CROP_STACK[1]), dtype="float32")
    plt.figure()

    for i, folder in enumerate(image_folders):
        path_image = os.path.join(PATH_IMAGES_FOLDER, folder, IMAGE_NAME)
        
        crop_pos = (i // CROP_STACK[1], i % CROP_STACK[1])

        rstart = crop_pos[0]*CROP_SIZE[0]
        rend = rstart + CROP_SIZE[0]
        cstart = crop_pos[1]*CROP_SIZE[1]
        cend = cstart + CROP_SIZE[1]

        rstart_load = CROP_ORIGO[0]
        rend_load = rstart_load + CROP_SIZE[0]
        cstart_load = CROP_ORIGO[1]
        cend_load = cstart_load + CROP_SIZE[1]
        image  = cv.imread(path_image, -1)[rstart_load:rend_load, cstart_load:cend_load]
        arr[rstart:rend, cstart:cend] = image

        if USE_SUBPLOTS:
            plt.subplot(CROP_STACK[0], CROP_STACK[1], i+1)
            plt.imshow(arr[rstart:rend, cstart:cend], cmap='gray')
            plt.xticks(ticks=[])
            plt.yticks(ticks=[])
            plt.title(folder.split('_')[1])
            
    
    if not USE_SUBPLOTS:
        plt.imshow(arr, cmap='gray')
        #plt.xticks(ticks=[])
        plt.yticks(ticks=[])
    
    plt.xticks(ticks=np.array(list(range(0,10)))*CROP_SIZE[1] + CROP_SIZE[1]/2,
    labels=[f.split("_")[1] for f in image_folders], rotation=45, fontsize=7)
    plt.tight_layout()
    plt.savefig('explore_channels.png', dpi=500, bbox_inches='tight', pad_inches=0)
