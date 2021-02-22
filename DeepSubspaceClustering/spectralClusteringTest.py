from sklearn.cluster import SpectralClustering   
import numpy as np

CHANNELS = ["Aperture", "Apodized", "BrightField", "DarkField", "DFIOpen", "DFIPhase", "DPI", "iSSC", "Phase", "UVPhase"]
KS = [2,3,4,5,6,7,8,9]


if __name__ == "__main__":
    np.set_printoptions(linewidth=150)
    path_file = r"D:\Speciale\Code\output\DSC_New\Runs_3264_relu\2020-12-08--14-43-12_DSC_3264_relu\matrices\Z_norm.csv"

    Z = np.loadtxt(path_file, delimiter=',')
    np.fill_diagonal(Z, 1)

    X = Z
    for K in KS:
        print("K=%d" % K)
        clustering = SpectralClustering(n_clusters=K, affinity='precomputed').fit(X)
        labels = clustering.labels_

        for idx in range(np.max(labels) + 1):
            clusters = []
            for i in range(10):
                if labels[i]==idx:
                    clusters.append(CHANNELS[i])
            print(clusters)
        print("")

