import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

PATH_MATRIX_C = r"D:\Speciale\Code\output\DSC_New\Runs_3264_relu\2020-12-08--14-43-12_DSC_3264_relu\matrices\C.csv"


if __name__ == "__main__":

    C = np.loadtxt(PATH_MATRIX_C, delimiter=',')
    Cabs = np.abs(C)
    Cnorm = Cabs / np.sum(Cabs, axis=1, keepdims=True)
    Z = Cnorm + np.transpose(Cnorm)
    D = 1/Z
    np.fill_diagonal(D, 0)

    cmap = 'bwr'

    plt.figure(figsize=(8,2))
    for i,(matrix, title) in enumerate(zip([C,Cabs,Cnorm,Z,D], ["$\mathbf{C}$", "$\mathbf{C}_{abs}$", "$\mathbf{C}_{norm}$", "$\mathbf{Z}$", "$\mathbf{D}$"])):
        plt.subplot(1, 5, 1+i)
        vmax = max(np.max(matrix), np.max(np.abs(matrix)))
        plt.imshow(matrix, vmin=-vmax, vmax=vmax, cmap=cmap)
        plt.title(title)
        plt.xticks(ticks=[])
        plt.yticks(ticks=[])

    plt.tight_layout()
    plt.savefig('dsc_similarity_processing.pdf')
    plt.savefig('dsc_similarity_processing.png', dpi=250)
    plt.show()