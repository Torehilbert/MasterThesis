import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='whitegrid')
import numpy as np
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import wfutils.infofile
import wfutils.mmap
from wfutils.minmaxscaler import MinMaxScaler


PATH_DATA = r"E:\validate32_redist_crop_rescale"
PATH_MODEL = r"D:\Speciale\Code\output\DSC_New\Runs_3264_relu\2020-12-08--14-43-12_DSC_3264_relu\model_best"
CHANNEL = [0,1,2,3,4,5,6,7,8,9]
#CHANNEL = [2,7,8]
IMAGES_TO_VISUALIZE = [(0,0,CHANNEL), (1,0,CHANNEL), (2,0,CHANNEL)]


if __name__ == "__main__":
    # Load data
    shapes, data_dtype = wfutils.infofile.read_info(wfutils.infofile.get_path_to_infofile(PATH_DATA))
    mmaps = wfutils.mmap.get_class_mmaps_read(PATH_DATA, shapes, data_dtype)
    

    # Load model
    model = tf.keras.models.load_model(PATH_MODEL)

    # Reconstruct images
    images = []
    images_reconstructed = []

    for image_setting in IMAGES_TO_VISUALIZE:
        X = mmaps[image_setting[0]][image_setting[1]]
        #X = scaler.scale(X)

        # crop
        #X = np.array(X[16:48, 16:48], dtype="float32")

        Xre, _, _ = model(np.expand_dims(X, axis=0))
        
        images.append(X[:,:,image_setting[2]])
        images_reconstructed.append(np.array(Xre[0])[:,:,image_setting[2]])

    # Plot reconstructions
    for i, imset in enumerate(IMAGES_TO_VISUALIZE):
        # Find min-max across both real and reconstruction

        vmin = 0
        vmax = 1
        num_channels = len(CHANNEL)
        plt.figure(figsize=(8, 1.75))
        for j in range(num_channels):
            plt.subplot(2, num_channels, j+1)


            plt.imshow(images[i][6:26,6:26,j], cmap='gray', vmin=vmin, vmax=vmax)
            plt.xticks(ticks=[])
            plt.yticks(ticks=[])
            if j==0:
                plt.ylabel("$X_{i}$", rotation=0, labelpad=10)
            plt.subplot(2,num_channels, j+1 + num_channels)
            plt.imshow(images_reconstructed[i][6:26,6:26,j], cmap='gray', vmin=vmin, vmax=vmax)
            plt.xticks(ticks=[])
            plt.yticks(ticks=[])            
            if j==0:
                plt.ylabel("$\hat{X}_{i}$", rotation=0, labelpad=10)
        plt.tight_layout()
        plt.savefig("dsc_reconstruct_visualization_C%d.png" % (imset[0] + 1), dpi=250)
        plt.savefig("dsc_reconstruct_visualization_C%d.pdf" % (imset[0] + 1))
    plt.show()
        

