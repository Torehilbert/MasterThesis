import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='whitegrid')
import tensorflow as tf
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nets.student import student_network, student_channel_importance

import wfutils.infofile
#import wfutils.log


PATH_MODEL = r"D:\Speciale\Code\output\TeacherStudent\Real Student 16\Original Regularization\Teacher A\Run 1\student"
PATH_DATA = r"D:\Speciale\Code\output\TeacherStudent\Real Teacher 16\Teacher A\TC Val 16"

if __name__ == "__main__":
     # Load data input info
    shapes_val, data_dtype_val = wfutils.infofile.read_info_codes(wfutils.infofile.get_path_to_infofile(PATH_DATA))

    model = tf.keras.models.load_model(PATH_MODEL)

    Xval = np.memmap(os.path.join(PATH_DATA, 'X.npy'), shape=shapes_val[0], dtype=data_dtype_val, mode='r')[:]
    Yval = np.memmap(os.path.join(PATH_DATA, 'Y.npy'), shape=shapes_val[1], dtype=data_dtype_val, mode='r')[:]
    Ypred = np.zeros_like(Yval)
    
    for i in range(Xval.shape[0]):
        x = np.expand_dims(Xval[i], axis=0)
        pred = model(x)
        Ypred[i] = pred

    vmin = min(np.min(Ypred), np.min(Yval))
    vmax = max(np.max(Ypred), np.max(Yval))

    plt.figure(figsize=(8,8))
    for i in range(16):
        plt.subplot(4,4,1+i)
        plt.scatter(Ypred[:,i], Yval[:,i], marker='.', alpha=0.25) #, edgecolors='white')
        plt.xlim([vmin, vmax])
        plt.ylim([vmin, vmax])

        if i%4==0:
            plt.ylabel('True code')
        else:
            plt.yticks(ticks=[])
        
        if i//4==3:
            plt.xlabel("Pred. code")
        else:
            plt.xticks(ticks=[])

        plt.grid()
    plt.show()

    
    # print(Ypred.shape)

    # print(model(np.expand_dims(Xval[0], axis=0)))
    # exit(0)
    # for i in range(Xval.shape[0]):
    #     x = np.expand_dims(Xval[i], axis=0)
    #     pred = model.predict(x)
    #     Ypred[i] = pred

