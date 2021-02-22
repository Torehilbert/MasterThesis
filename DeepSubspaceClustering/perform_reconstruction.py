import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import wfutils.infofile
import wfutils.mmap

parser = argparse.ArgumentParser()
parser.add_argument("-model", required=False, type=str, default=r"D:\Speciale\Code\output\DeepSubspaceClustering\DSCNet Initial Training\model_end")
parser.add_argument("-data", required=False, type=str, default=r"E:\full32_redist")
parser.add_argument("-image_healthy_idx", required=False, type=int, nargs="+", default=[0,1])
parser.add_argument("-image_apotosis_idx", required=False, type=int, nargs="+", default=None)
parser.add_argument("-image_dead_idx", required=False, type=int, nargs="+", default=None)
parser.add_argument("-image_channels", required=False, type=int, nargs="+", default=[0, 6])


if __name__ == "__main__":
    args = parser.parse_args()

    model = tf.keras.models.load_model(args.model)
    
    shapes, data_dtype = wfutils.infofile.read_info(wfutils.infofile.get_path_to_infofile(args.data))
    mmaps = wfutils.mmap.get_class_mmaps_read(args.data, shapes, data_dtype)

    n_images = len(args.image_healthy_idx) if args.image_healthy_idx is not None else 0
    n_images += len(args.image_apotosis_idx) if args.image_apotosis_idx is not None else 0
    n_images += len(args.image_dead_idx) if args.image_dead_idx is not None else 0

    images = np.zeros(shape=(n_images, shapes[0][1], shapes[0][2], shapes[0][3]), dtype="float32")

    if args.image_healthy_idx is not None:
        images[0:len(args.image_healthy_idx)] = mmaps[0][args.image_healthy_idx]

    image_reconstructed = model(images).numpy()


    for i in range(n_images):
        im = images[i]
        imre = image_reconstructed[i]

        for j in range(len(args.image_channels)):
            chidx = args.image_channels[j]
            ch = im[:,:,chidx]
            chre = imre[:,:,chidx]

            minval = min(np.min(ch), np.min(chre))
            maxval = max(np.max(ch), np.max(chre))

            plt.figure()
            plt.title("Image %d Channel %d" % (i, chidx))
            plt.subplot(1,2,1)
            plt.imshow(ch, vmin=minval, vmax=maxval, cmap='gray')
            plt.subplot(1,2,2)
            plt.imshow(chre, vmin=minval, vmax=maxval, cmap='gray')
    
    plt.show()
        