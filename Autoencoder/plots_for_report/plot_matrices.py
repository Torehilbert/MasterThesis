import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='whitegrid')

import numpy as np

PATH_Z = r"D:\Speciale\Code\output\DSC_New\Runs_3264_relu\2020-12-08--14-43-12_DSC_3264_relu\matrices\Z_norm.csv"
PATH_D = r"D:\Speciale\Code\output\DSC_New\Runs_3264_relu\2020-12-08--14-43-12_DSC_3264_relu\matrices\D_norm_division.csv"
CHANNELS = ["Aperture", "ApodizedAP", "BrightField", "DarkField", "DFIOpen", "DFIPhase", "DPI", "iSSC", "Phase", "UVPhase"]


if __name__ == "__main__":
    Z = np.loadtxt(PATH_Z, delimiter=',')
    D = np.loadtxt(PATH_D, delimiter=',')

    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1)
    sns.heatmap(Z, vmin=0, vmax=1, cmap='coolwarm', linewidths=1)
    plt.xticks(ticks=np.arange(10)+0.75, labels=CHANNELS, rotation=-65)
    plt.yticks(ticks=np.arange(10)+0.5, labels=CHANNELS, rotation=0)
    plt.xlabel("(a) $\mathbf{Z}$")
    plt.subplot(1,2,2)
    sns.heatmap(D, cmap='coolwarm', linewidths=1)
    plt.yticks(ticks=[])
    plt.xticks(ticks=np.arange(10)+0.75, labels=CHANNELS, rotation=-65)
    plt.xlabel("(b) $\mathbf{D}$")
    plt.tight_layout()

    plt.savefig("dsc_matrices_final.pdf")
    plt.savefig("dsc_matrices_final.png", dpi=250)
    
    plt.show()
    