import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, set_link_color_palette
from sklearn.metrics import silhouette_score
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import wfutils.log
from wfutils import CHANNELS


def get_cluster_model(D, linkage):
    return AgglomerativeClustering(
        affinity='precomputed',
        distance_threshold=0,
        n_clusters=None,
        compute_full_tree=True,
        linkage=linkage).fit(D)


def calculate_linkage_matrix(model):
    ns = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        n_current = 0
        for child_idx in merge:
            if child_idx < n_samples:
                n_current += 1  # leaf node
            else:
                n_current += ns[child_idx - n_samples]
        ns[i] = n_current

    return np.column_stack([model.children_, model.distances_, ns]).astype(float)


def create_dendrogram_plot(linkage_matrix, threshold, channel_labels, ylabel='Distance', figsize=(6,4)):
    BIAS_ADD = 0.01
    set_link_color_palette(['k']*20)
    fig = plt.figure(figsize=figsize)
    color_threshold = threshold
    linkage_added_bias = linkage_matrix
    linkage_added_bias[:,2] += BIAS_ADD
    dendrogram(
        linkage_added_bias, 
        above_threshold_color='k',
        labels=channel_labels)
    plt.xticks(rotation=-65)
    plt.ylabel(ylabel)
    plt.hlines(color_threshold, 0, 100, linestyles='dashed', colors='gray')

    yticks = np.linspace(0, np.max(linkage_matrix[:,2]), 5, endpoint=True)
    yticklabels = ["%.2f" % val for val in yticks]
    plt.yticks(ticks=yticks + BIAS_ADD, labels=yticklabels)
    plt.tight_layout()
    return fig


def save_dendrogram_plot(path_file, linkage_matrix, threshold, channel_labels, ylabel='Distance', dpi=250, figsize=(6,4)):
    fig = create_dendrogram_plot(linkage_matrix, threshold, channel_labels, ylabel, figsize=figsize)
    plt.savefig(path_file, dpi=dpi)
    plt.close(fig)


LINKAGE = 'single'
PATH = r"D:\Speciale\Code\output\DSC_New\Runs_3264_relu\2020-12-08--14-43-12_DSC_3264_relu\matrices\D_norm_division.csv"


if __name__ == "__main__":
    D = np.loadtxt(PATH, delimiter=",")
    
    model = get_cluster_model(D, LINKAGE)
    linkage_matrix = calculate_linkage_matrix(model)

    # Save plot
    save_dendrogram_plot(
        path_file='DENDROGRAM.png',
        linkage_matrix=linkage_matrix,
        threshold=2.50,
        channel_labels=CHANNELS,
        ylabel='Distance',
        dpi=250,
        figsize=(6,4)
    )

    # Save plot
    save_dendrogram_plot(
        path_file='DENDROGRAM.pdf',
        linkage_matrix=linkage_matrix,
        threshold=2.50,
        channel_labels=CHANNELS,
        ylabel='Distance',
        dpi=250,
        figsize=(6,4)
    )