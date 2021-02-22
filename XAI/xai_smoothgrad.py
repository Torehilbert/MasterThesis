import os
import sys
import tensorflow as tf
import numpy as np
import argparse
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import wfutils.infofile
import wfutils.mmap
import smoothgrad


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-path_model", required=False, type=str, default=r"D:\Speciale\Code\output\Performance Trainings\NONOISE\2021-01-17--13-48-18_Training_57580\model_end")
    parser.add_argument("-evaluation_data", required=False, type=str, default=r"E:\validate32_redist")
    parser.add_argument("-n_max", required=False, type=int, default=10)
    args = parser.parse_args()
    
    # A) Load model
    model = tf.keras.models.load_model(args.path_model)

    # B) Load inference data
    path_info_file = wfutils.infofile.get_path_to_infofile(args.evaluation_data)
    shapes, data_dtype = wfutils.infofile.read_info(path_info_file)

    mmaps = wfutils.mmap.get_class_mmaps_read(args.evaluation_data, shapes, data_dtype)
    
    # C) Run inference to gather channel-wise importance
    for CLASS in range(3):
        for IMAGE in range(shapes[CLASS][0]):
            test_image = np.expand_dims(mmaps[CLASS][IMAGE], axis=0)



    CLASS = 0
    IMAGE = 0

    test_image = np.expand_dims(mmaps[CLASS][IMAGE], axis=0)
    grads, reference_class = smoothgrad.smooth_grad(
        model=model, 
        inp=test_image, 
        noise_std=0.01, 
        samples=32, 
        signed=True, 
        squared=False)

    maxabsval = max(np.abs(np.max(grads)), np.abs(np.min(grads)))
    minval = -maxabsval
    maxval = maxabsval

    import matplotlib.pyplot as plt
    plt.figure()
    plt.suptitle("Real: %d  Predicted: %d" % (CLASS, reference_class))
    for i in range(grads.shape[2]):
        plt.subplot(6,4, 2*i+1)
        plt.imshow(grads[:,:,i], cmap='bwr', vmin=minval, vmax=maxval)
        plt.subplot(6,4,2*i+1+1)
        plt.imshow(test_image[0,:,:,i], cmap='gray')

    plt.show()