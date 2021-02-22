import numpy as np
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, set_link_color_palette
from sklearn.cluster import AgglomerativeClustering
import os
import sys
import math
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plotlib.style import COLORS
from wfutils import CHANNELS
import wfutils.log
from plotlib.plot_pair_scatter import _get_subplot_layout


def plot_channel_wise_losses(dfs, loss_stat_id, title, labels=None):
    columns = [str(i) +" %d" % loss_stat_id for i in range(10)]
    vmin = min([np.min(df.loc[:,columns].values) for df in dfs])
    vmax = max([np.max(df.loc[:,columns].values) for df in dfs])

    for c in range(3):
        df = dfs[c]
        for i, cname in enumerate(columns):
            mu = np.mean(df.loc[:, cname].values)
            std = np.std(df.loc[:, cname].values)
            plt.errorbar(i+(c-1)*0.1, mu, yerr=std, fmt='o', color=COLORS[c])

    plt.title(title)  
    plt.ylim([vmin, 0.4*vmax])
    if labels is not None:
        plt.xticks(ticks=np.arange(0,10), labels=CHANNELS, rotation=-65)
    else:
        plt.xticks(ticks=[])
    plt.tight_layout()


def plot_channel_wise_unorms(dfs, norm_id, title, labels=None):
    columns = [str(i) +" %d" % norm_id for i in range(10)]
    #vmin = min([np.min(df.loc[:,columns].values) for df in dfs])
    #vmax = max([np.max(df.loc[:,columns].values) for df in dfs])

    vmin = math.inf
    vmax = -math.inf
    for c in range(3):
        df = dfs[c]
        for i, cname in enumerate(columns):
            mu = np.mean(df.loc[:, cname].values)
            std = np.std(df.loc[:, cname].values)
            plt.errorbar(i+(c-1)*0.1, mu, yerr=std, fmt='o', color=COLORS[c])

            if mu+std > vmax:
                vmax = mu + std
            if mu-std < vmin:
                vmin = mu - std

    plt.title(title)
    #plt.ylim([0, vmax])
    #plt.ylim([0, 1.0*vmax])
    if labels is not None:
        plt.xticks(ticks=np.arange(0,10), labels=CHANNELS, rotation=-65)
    else:
        plt.xticks(ticks=[])
    plt.tight_layout()    


def assemble_cluster_sets(model):
    cluster_sets = []
    
    clusters = {}
    for i in range(10):
        clusters[i] = [i]
    n_clusters = 10
    
    for merge in model.children_:
        # export to output list
        cluster_sets.append([])
        for (key,value) in clusters.items():
            cluster_sets[-1].append(tuple(value))
        
        new_cluster = []
        new_cluster.extend(clusters[merge[0]])
        new_cluster.extend(clusters[merge[1]])

        del clusters[merge[0]]
        del clusters[merge[1]]
        clusters[n_clusters] = new_cluster
        n_clusters += 1
    
    # add final cluster (complete) not included in model.children_
    cluster_sets.append([(0,1,2,3,4,5,6,7,8,9)])

    return cluster_sets


def write_cluster_sets(path_file, cluster_sets):
    with open(path_file, 'w') as f:
        for cluster_set in cluster_sets:
            cluster_strings = []
            for cluster in cluster_set:
                cluster_strings.append("-".join(["%d" % ch for ch in cluster]))
            cluster_set_string = ",".join(cluster_strings)
            f.write(cluster_set_string + "\n")

def read_cluster_sets(path_file):
    with open(path_file, 'r') as f:
        cluster_set_lines = f.readlines()
        cluster_sets = []
        for cluster_set_string in cluster_set_lines:
            cluster_set = []
            cluster_strings = cluster_set_string.split(",")
            for cstring in cluster_strings:
                cluster_set.append(tuple([int(ch_string) for ch_string in cstring.split("-")]))
            cluster_sets.append(cluster_set)
    return cluster_sets


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)
    # Plot the corresponding dendrogram
    set_link_color_palette(['k']*20)
    return dendrogram(linkage_matrix, above_threshold_color='k',**kwargs)


def row_normalize_matrix(matrix):
    N,M = matrix.shape
    row_sums = np.sum(matrix, axis=1)
    row_sum_matrix = np.reshape(np.repeat(row_sums, N), (N,M))
    return matrix / row_sum_matrix


parser = argparse.ArgumentParser()
parser.add_argument("-path_inference", required=False, type=str, default=r"D:\Speciale\Code\output\DSC_New\2020-12-03--11-50-08_DSC_Small\Inference")
parser.add_argument("-use_correction_unorm", required=False, type=int, default=0)
parser.add_argument("-use_correction_rowsum", required=False, type=int, default=0)
parser.add_argument("-use_correction_loss", required=False, type=int, default=1)
parser.add_argument("-clustering_linkage", required=False, type=str, default="single") # single, average, or complete
parser.add_argument("-unorm_stat_idx", required=False, type=int, default=1)

parser.add_argument("-create_loss_plots", required=False, type=int, default=1)
parser.add_argument("-create_norm_plots", required=False, type=int, default=1)
parser.add_argument("-create_matrix_plots", required=False, type=int, default=1)

parser.add_argument("-show_plots", required=False, type=int, default=1)
parser.add_argument("-show_loss_plots", required=False, type=int, default=0)
parser.add_argument("-show_norm_plot", required=False, type=int, default=0)

parser.add_argument("-show_matrix_plots", required=False, type=int, default=0)
parser.add_argument("-matrix_include_c", required=False, type=int, default=1)
parser.add_argument("-matrix_include_cabs", required=False, type=int, default=0)
parser.add_argument("-matrix_include_cunorm", required=False, type=int, default=1)
parser.add_argument("-matrix_include_crow", required=False, type=int, default=1)
parser.add_argument("-matrix_include_closs", required=False, type=int, default=1)
parser.add_argument("-matrix_include_z", required=False, type=int, default=1)
parser.add_argument("-matrix_include_d", required=False, type=int, default=1)

parser.add_argument("-show_dendrogram", required=False, type=int, default=1)

parser.add_argument("-dpi", required=False, type=int, default=250)


if __name__ == "__main__":
    args = parser.parse_args()

    path_output_folder = wfutils.log.create_output_folder("DSCNet_Clustering")
    wfutils.log.log_arguments(path_output_folder, args)

    ### LOAD DATA
    dfs = [pd.read_csv(os.path.join(args.path_inference, 'raw_losses_class_%d.csv' % (i+1))) for i in range(3)]
    dfs_norms = [pd.read_csv(os.path.join(args.path_inference, 'raw_unorms_class_%d.csv' % (i+1))) for i in range(3)]


    ### COEFFICIENT MATRICES FLOW
    matrices = []
    labels = []

    # A) Get raw C matrix
    C = np.loadtxt(os.path.join(args.path_inference, 'C.csv'), delimiter=',')
    matrices.append(C)
    labels.append("%d_C" % len(matrices))

    # B) Absolute of C
    matrices.append(np.abs(matrices[-1]))
    labels.append("%d_C (abs)" % len(matrices))

    # C) U 2-norm correction
    if args.use_correction_unorm==1:
        unorm_column = np.zeros(shape=(10,10), dtype="float64")
        for ch in range(10):
            for df in dfs_norms:
                unorm_column[:,ch] += np.mean(df.loc[:, "%d %d" % (ch, args.unorm_stat_idx)])
            unorm_column[:,ch] = unorm_column[:,ch] / 3

        matrices.append(matrices[-1] * unorm_column)
        labels.append("%d_C (U-norm-corrected)" % len(matrices))

    # D) Normalize C by row sums
    if args.use_correction_rowsum==1:
        matrices.append(row_normalize_matrix(matrices[-1]))
        labels.append("%d_C (row-normalized)" % len(matrices))

    # E) Correct C with loss
    if args.use_correction_loss==1:
        loss_x = np.zeros(shape=(10,10), dtype='float64')
        for ch in range(10):
            for df in dfs:
                loss_x[ch, :] += np.mean(df.loc[:, "%d %d" % (ch, 1)]) / len(dfs)
        
        matrices.append(matrices[-1] * (1/loss_x))
        labels.append("%d_C (loss-corrected)" % len(matrices))

    # F) Make symmetric into affinity matrix Z
    matrices.append(matrices[-1] + np.transpose(matrices[-1]))
    labels.append("%d_Z" % len(matrices))


    # G) Convert to distance matrix
    matrices.append(1/matrices[-1])
    np.fill_diagonal(matrices[-1], 0)
    labels.append("%d_D" % len(matrices))

    
    ### PERFORM AGGLOREMATIVE CLUSTERING
    cluster = AgglomerativeClustering(
        affinity='precomputed', 
        distance_threshold=0, 
        n_clusters=None, 
        compute_full_tree=True, 
        linkage=args.clustering_linkage).fit(matrices[-1])
    

    cluster_sets = assemble_cluster_sets(cluster)   
    write_cluster_sets(os.path.join(path_output_folder, 'cluster_levels.txt'), cluster_sets)

    ### PLOT DENDROGRAM
    fig = plt.figure()
    dendrogram_info = plot_dendrogram(cluster, labels=CHANNELS)
    plt.xticks(rotation=-65)
    plt.ylabel('Distance linkage (average)')
    plt.tight_layout()
    plt.savefig(os.path.join(path_output_folder, 'dendrogram.png'), dpi=args.dpi)
    if args.show_dendrogram==0:
        plt.close(fig)

    ### PLOT CHANNEL USAGE
    plt.figure()
    usage = np.sum(matrices[-2], axis=0)
    plt.bar(np.arange(0, 10), usage)

    # plot channel-wise losses
    if args.create_loss_plots==1:
        fig = plt.figure()
        for stat_idx, title in enumerate(['Loss: Total', 'Loss: X reconstruction', 'Loss: U reconstruction']):
            plt.subplot(3,1,stat_idx+1)
            plot_channel_wise_losses(dfs, stat_idx, title, labels=CHANNELS if stat_idx==2 else None)
        
        plt.savefig(os.path.join(path_output_folder, 'loss.png'), dpi=args.dpi)
        if args.show_loss_plots==0:
            plt.close(fig)
    
    # plot channel-wise 2-norms of U  
    if args.create_norm_plots==1:
        fig = plt.figure()
        unorm_means = np.zeros(shape=(10,), dtype="float64")
        unorm_stds = np.zeros(shape=(10,), dtype="float64")
        for ch in range(10):
            for df in dfs_norms:
                unorm_means[ch] += np.mean(df.loc[:, "%d %d" % (ch, args.unorm_stat_idx)]) / 3
                unorm_stds[ch] += np.std(df.loc[:, "%d %d" % (ch, args.unorm_stat_idx)]) / 3
        plt.errorbar(np.arange(0,10), unorm_means, yerr=unorm_stds, fmt='ok')
        plt.xticks(ticks=np.arange(0,10), labels=CHANNELS, rotation=-65)
        plt.ylabel("$||U||_2$", rotation=0)
        plt.tight_layout()

        plt.savefig(os.path.join(path_output_folder, 'unorm.png'), dpi=args.dpi)
        if args.show_norm_plot==0:
            plt.close(fig)

    # plot matrices
    if args.create_matrix_plots==1:
        # determine which plots to include
        m_includes = [args.matrix_include_c, args.matrix_include_cabs]

        corrections = [args.use_correction_unorm, args.use_correction_rowsum, args.use_correction_loss]      
        for use, value in zip(corrections, [args.matrix_include_cunorm, args.matrix_include_crow, args.matrix_include_closs]):
            if use==1:
                m_includes.append(value)
        m_includes.append(args.matrix_include_z)
        m_includes.append(args.matrix_include_d)

        matrices_for_plotting = []
        labels_for_plotting = []
        for matrix, label, use in zip(matrices, labels, m_includes):
            if use==1:
                matrices_for_plotting.append(matrix)
                labels_for_plotting.append(label)
        
        # performing plot
        layout, _ = _get_subplot_layout(len(matrices_for_plotting))
        layout = (layout[1], layout[0])
        fig = plt.figure(figsize=(10,16))
        for i, (matrix, label) in enumerate(zip(matrices_for_plotting, labels_for_plotting)):
            matrix_absmax = np.nanmax(abs(matrix))
            plt.subplot(layout[0], layout[1], i+1)
            plt.imshow(matrix, cmap='bwr', vmin=-matrix_absmax, vmax=matrix_absmax)
            plt.xticks(ticks=np.arange(0,10), labels=CHANNELS, rotation=-65)
            plt.yticks(ticks=np.arange(0,10), labels=CHANNELS, rotation=0)
            plt.title(label)  
        plt.tight_layout()
        plt.savefig(os.path.join(path_output_folder, 'matrices.png'), dpi=args.dpi)      

        if args.show_matrix_plots==0:
            plt.close(fig)

    # output matrices
    for matrix, label in zip(matrices, labels):
        np.savetxt(os.path.join(path_output_folder, label + ".csv"), matrix, delimiter=",")

    ### SHOW PLOTS
    if args.show_plots==1 and sum([args.show_loss_plots, args.show_norm_plot, args.show_matrix_plots, args.show_dendrogram]) > 0:
        plt.show()
