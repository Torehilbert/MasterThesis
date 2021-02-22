import argparse
import os
import sys
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.cluster import adjusted_rand_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Autoencoder.dsc_clustering_ng import convert_cluster_set_to_labels, read_cluster_sets, perform_hierarchical_clustering, create_dendrogram_plot
from dsc_extract_c_matrices import calculate_d, OUTPUT_FOLDER_NAME


def compute_rand_scores(labels):
    n = len(labels)
    rand_scores = np.zeros(shape=(n,n))
    for i, labels_primary in enumerate(labels):
        for j, labels_secondary in enumerate(labels):
            rand_scores[i,j] = adjusted_rand_score(labels_primary, labels_secondary)
    return rand_scores


parser = argparse.ArgumentParser()
parser.add_argument("-root_folder", required=False, type=str, default=r"D:\Speciale\Code\output\DSC_New\Runs_3264_relu")
parser.add_argument("-filename_matrix", required=False, type=str, nargs="+", default=["Z.csv", "Z_norm.csv"])


FOLDERNAME_AGGREGATION = 'AggregationAnalysis'
FOLDERNAME_MATRIX_VISUALIZATIONS = OUTPUT_FOLDER_NAME + "_visualizations"

if __name__ == "__main__":
    args = parser.parse_args()

    # create output folder
    path_output_folder= os.path.join(args.root_folder, FOLDERNAME_AGGREGATION)
    os.makedirs(path_output_folder, exist_ok=True)
    os.makedirs(os.path.join(path_output_folder, OUTPUT_FOLDER_NAME), exist_ok=True)
    os.makedirs(os.path.join(path_output_folder, FOLDERNAME_MATRIX_VISUALIZATIONS), exist_ok=True)

    # extract matrices
    for matrix_name in args.filename_matrix:
        basename = matrix_name.split(".")[0]
        Zs = []
        for cont in os.listdir(args.root_folder):
            if cont == FOLDERNAME_AGGREGATION:
                continue

            path_matrix = os.path.join(args.root_folder, cont, 'matrices', matrix_name)
            Z = np.loadtxt(path_matrix, delimiter=',')
            Zs.append(Z)

        # aggregate matrices
        Zs = np.stack(Zs, axis=0)
        Z = np.mean(Zs, axis=0)
        Zstd = np.std(Zs, axis=0)
        Zstdnormalized = Zstd / Z
        np.fill_diagonal(Zstdnormalized, 0)

        for d_method, postfix in zip(['subtract', 'division'], ['_subtract', '_division']):
            D = calculate_d(Z, method=d_method)
            np.savetxt(os.path.join(path_output_folder, OUTPUT_FOLDER_NAME, 'D%s%s.csv' % ('_norm' if basename.endswith('_norm') else '', postfix)), D, delimiter=',')
        #D_sub = calculate_d(Z, method='subtract')
        #D_div = calculate_d(Z, method='division')

        for M, statname in zip([Z, Zstd, Zstdnormalized], ['', '_std', '_stdrel']):
            np.savetxt(os.path.join(path_output_folder, OUTPUT_FOLDER_NAME, '%s%s.csv' % (basename, statname)), M, delimiter=',')

            plt.figure()
            sns.heatmap(M, vmin=0, vmax=np.max(M), cmap='magma', annot=True, fmt=".2f")
            plt.savefig(os.path.join(path_output_folder, FOLDERNAME_MATRIX_VISUALIZATIONS, '%s%s.png' % (basename, statname)))
            plt.clf()
            plt.close()

