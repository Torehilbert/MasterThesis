import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from wfutils import CHANNELS


PATH_MATRIX = r"D:\Speciale\Code\output\DSC_New\Runs_3264_relu\2020-12-08--14-43-12_DSC_3264_relu\matrices\Z_norm.csv"


if __name__ == "__main__":
    M = np.loadtxt(PATH_MATRIX, delimiter=",")

    plt.figure()
    sns.heatmap(M, cmap='bwr', vmin=-np.max(np.abs(M)), vmax=np.max(np.abs(M)))
    plt.xticks(ticks=np.arange(M.shape[0]), labels=CHANNELS, rotation=-90)
    plt.yticks(ticks=np.arange(M.shape[0]), labels=CHANNELS, rotation=0)
    plt.title("$\mathbf{Z}$")
    plt.tight_layout()
    plt.show()


