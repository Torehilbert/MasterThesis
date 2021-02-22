import argparse
import tensorflow as tf
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data.pipe as pipe

#@tf.function
def inference(x, y, model, loss_fn):
    y_pred = model(x).numpy()
    pred_class = np.argmax(y_pred, axis=1)
    inds_miss = np.where(y!=pred_class)[0]
    return inds_miss, y_pred[inds_miss,:]


parser = argparse.ArgumentParser()
parser.add_argument('-path_test_data', required=False, type=str, nargs='+', default=[r"E:\full32_redist\1.npy", r"E:\full32_redist\2.npy", r"E:\full32_redist\3.npy"])
parser.add_argument('-path_model', required=False, type=str, default=r"D:\Speciale\Repos\cell crop phantom\output\Trainings Full New\2020-09-29--01-04-27_Training\model")

if __name__ == "__main__":
    args = parser.parse_args()

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    model = tf.keras.models.load_model(args.path_model)
    
    test_data = pipe.MyDataPipeline(path_classes=args.path_test_data, batch_size=8)


    difficult_images = []
    for step, (x_batch, y_batch) in enumerate(test_data):
        inds_miss, probs = inference(x_batch, y_batch, model, loss_fn)
        for i,ind in enumerate(inds_miss):
            difficult_images.append((x_batch[ind], probs[i], y_batch[ind]))
        if step ==20:
            break

    import matplotlib.pyplot as plt
    plt.figure()
    for ch in range(10):
        plt.subplot(2,5,ch+1)
        plt.imshow(difficult_images[0][0][:,:,ch], cmap='gray')
    plt.suptitle("Preds: %f %f %f,  True: %d" % (difficult_images[0][1][0], difficult_images[0][1][1], difficult_images[0][1][2], difficult_images[0][2]))
    plt.show()
        #accuracies.append(float(accuracy))
    #print("Accuracy=", sum(accuracies)/len(accuracies))