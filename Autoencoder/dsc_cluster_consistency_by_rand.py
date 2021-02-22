from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import numpy as np
import argparse
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Autoencoder.dsc_clustering_ng import convert_cluster_set_to_labels, read_cluster_sets, perform_hierarchical_clustering, create_dendrogram_plot
from plotlib.style import COLORS, CMAP_NEGPOS
from plotlib.plot_conformed import plot_line


def compute_rand_scores(labels):
    n = len(labels)
    rand_scores = np.zeros(shape=(n,n))
    for i, labels_primary in enumerate(labels):
        for j, labels_secondary in enumerate(labels):
            rand_scores[i,j] = adjusted_rand_score(labels_primary, labels_secondary)
    return rand_scores


def analysis_rand_by_k(labels_by_k, func_stat=np.mean):
    '''
    labels_by_k     numpy array of shape = (n_runs, n_ks, n_channels)
    '''
    rand_scores = np.zeros(shape=(labels_by_k.shape[1]))
    for k_idx in range(labels_by_k.shape[1]):
        raw_rands = []
        for i in range(labels_by_k.shape[0]):
            for j in range(i+1, labels_by_k.shape[0]):
                raw_rands.append(adjusted_rand_score(labels_by_k[i,k_idx,:], labels_by_k[j,k_idx,:]))
        rand_scores[k_idx] = func_stat(raw_rands)
    return rand_scores


def extract_labels_from_runs(path_root, subfolders, filename_cluster_set, foldername_clustering='clustering', exclusion_folders=None):
    all_labels = []
    for _, subfolder in enumerate(subfolders):
        if exclusion_folders is not None and subfolder in exclusion_folders:
            print("INFO: The subfolder %s was excluded from the analysis as instructed." % subfolder)
            continue
        
        path_cluster_set_file = os.path.join(path_root, subfolder, foldername_clustering, filename_cluster_set)
        labels_by_cluster_set = list(reversed([convert_cluster_set_to_labels(cset) for cset in read_cluster_sets(path_cluster_set_file)]))
        all_labels.append(np.stack(labels_by_cluster_set[1:-1], axis=0))
    
    return np.stack(all_labels, axis=0)  # (Run, K, channel)


def distance_matrix_rand_matrix(rmatrix):
    Z =  (rand_scores - np.min(rand_scores)) / (np.max(rand_scores) - np.min(rand_scores)) + 0.01
    return 1/Z


parser = argparse.ArgumentParser()
parser.add_argument("-root_folder", required=False, type=str, default=r"D:\Speciale\Code\output\DSC_New\Runs_3264_relu")
parser.add_argument("-foldername_cluster", required=False, type=str, default="clustering")
parser.add_argument("-format_cluster_filename", required=False, type=str, default="cluster_levels<N>_<D>_<L>.txt")
parser.add_argument("-format_linkage_filename", required=False, type=str, default="linkage<N>_<D>_<L>.csv")
parser.add_argument("-foldernames_exclusion", required=False, type=str, nargs="+", default="AggregationAnalysis")


if __name__ == "__main__":
    args = parser.parse_args()
    subfolders = os.listdir(args.root_folder)
   

    fig = None
    save_configs={'dpi':250, 'format':['png', 'pdf'], 'filename':'dsc_ari_by_method'}
    for n, m_n in enumerate(['','norm']):
        for i, m_d in enumerate(["division"]):
            for j, m_l in enumerate(["single", "average", "complete"]):
                fname = args.format_cluster_filename.replace("<N>", "_"+m_n if m_n == 'norm' else "").replace("<D>", m_d).replace("<L>", m_l)
                all_labels = extract_labels_from_runs(args.root_folder, subfolders, fname, args.foldername_cluster, args.foldernames_exclusion)          
                rands_by_k = analysis_rand_by_k(labels_by_k=all_labels, func_stat=np.mean)

                last_line = (n==1 and j==2)
                fig = plot_line(x=np.arange(2,10), y=rands_by_k,
                        label=("%s%s" % (m_n+"-" if m_n == 'norm' else "", m_l)),
                        xlabel="Number of clusters (K)", ylabel="Mean ARI",
                        call_legend=last_line, tight_layout=last_line,
                        fig=fig, save_configs=save_configs if last_line else None,
                        color=['black', COLORS[2]][n],
                        linestyle=['solid', 'dashed', 'dotted'][j]
                )
    
    # parameters for final clustering method for RUNS
    FNAME_CLUSTER_SET = args.format_cluster_filename.replace("<N>", '_norm').replace("<D>", 'division').replace("<L>", 'single')
    LINKAGE_METHOD = 'average'
    K_RANGE = range(2,10)
    WINNER_IDX = 0
    
    ### Plot 2) ARI matrix between runs by K
    mean_rand_scores = []
    #plt.figure(figsize=(4.5, 3.5))
    plt.figure(figsize=(6,4))
    for k in reversed(K_RANGE):
        all_labels = []
        for i,folder in enumerate(os.listdir(args.root_folder)):
            if folder in args.foldernames_exclusion:
                print("Excluded %s from analysis!" % folder)
                continue
                
            path_cluster_file = os.path.join(args.root_folder, folder, args.foldername_cluster, FNAME_CLUSTER_SET)
            labels = convert_cluster_set_to_labels(list(reversed(read_cluster_sets(path_cluster_file)))[k-1])
            all_labels.append(labels)

        # compute initial rand scores
        rand_scores = compute_rand_scores(all_labels)
        mean_rand_scores.append(np.mean(rand_scores, axis=0))
        
        # reorder data based on initial clustering
        cluster_sets, _ = perform_hierarchical_clustering(distance_matrix_rand_matrix(rand_scores), LINKAGE_METHOD)
        idx_order = []
        all_labels_sorted = []
        for cluster in cluster_sets[-2]:
            idx_order.extend([idx for idx in cluster])
            all_labels_sorted.extend([all_labels[idx] for idx in cluster])

        # compute rand scores (on sorted data)
        rand_scores_sorted = compute_rand_scores(all_labels_sorted)

        # plotting
        plt.subplot(2, 4, k-1)
        plt.imshow(rand_scores_sorted, cmap=CMAP_NEGPOS, vmin=-1, vmax=1)
        plt.xticks(ticks=[])
        plt.yticks(ticks=[])
        plt.title('K=%d' % k)

        # add golden patch indicating most representative run
        local_idx = np.argmax(np.array(idx_order)==WINNER_IDX)
        plt.gca().add_patch(patches.Rectangle((local_idx, local_idx),1,1,linewidth=1, edgecolor='gold',facecolor='none'))
    
    plt.tight_layout()
    plt.savefig("dsc_ari_pairwise_by_k.pdf")
    plt.savefig("dsc_ari_pairwise_by_k.png", dpi=250)

    # Plot C) Run selection criteria
    global_mean_rand_scores = np.mean(np.stack(mean_rand_scores, axis=0), axis=0)
    idx_sorted = list(reversed([idx for idx in np.argsort(global_mean_rand_scores)]))
    
    
    plot_line(
        x=np.arange(len(idx_sorted)),
        y=np.array([global_mean_rand_scores[idx] for idx in idx_sorted]), 
        xlabel="Sorted run ID", 
        ylabel="Mean ARI",
        tight_layout=True,
        color=COLORS[2],
        marker='o')
    
    plt.savefig("dsc_ari_run_id.pdf")
    plt.savefig("dsc_ari_run_id.png", dpi=250)

    # plt.figure(figsize=(5,3))
    # plt.plot(np.array([global_mean_rand_scores[idx] for idx in idx_sorted]), '-ok', color=COLORS[2])
    # plt.ylabel("Mean ARI"); plt.xlabel("Run IDs (sorted by y)"); plt.tight_layout()

    plt.show()
    exit(0)

    winner_idx = np.argmax(global_mean_rand_scores)
    print("Winner idx=%d, currently set to %d" % (winner_idx, WINNER_IDX))

    # Plot D) Mean ARI for best run
    plt.figure(figsize=(4.5, 3.5))
    plt.plot(np.arange(2,10), np.stack(list(reversed(mean_rand_scores)), axis=0)[:,WINNER_IDX], '-or', color=COLORS[2])
    plt.plot(np.arange(2,10), np.mean(np.stack(list(reversed(mean_rand_scores)), axis=0), axis=1), '-ok')
    plt.xlabel("Number of clusters (K)"); plt.ylabel("Mean ARI"); plt.legend(["Best", "Average run"]); plt.ylim([0, 1]); plt.tight_layout() 
    plt.savefig("dsc_ari_best_run.png", dpi=500)
    plt.show()
