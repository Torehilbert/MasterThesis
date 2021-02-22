import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np


if __name__ == "__main__":

    PATHS = [r"D:\Speciale\Code\Autoencoder\plots_for_report\dsc_validation_loss_bs_exp.png", r"D:\Speciale\Code\Autoencoder\plots_for_report\dsc_validation_loss_lr_exp.png"]
    plt.figure(figsize=(5,8))
    for i, imname in enumerate(PATHS):
        plt.subplot(2,1,1+i)
        im = cv.cvtColor(cv.imread(imname), cv.COLOR_BGR2RGB)
        plt.imshow(np.array(im))
        plt.xticks(ticks=[])
        plt.yticks(ticks=[])
    
    plt.tight_layout()
    plt.savefig("dsc_appendix_experiments.png", dpi=250)
    plt.show()