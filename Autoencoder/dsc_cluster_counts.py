import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Inference.subset_combinations import get_possible_channel_combinations
from Autoencoder.dsc_cluster_consistency_by_rand import extract_labels_from_runs


def read_cluster_file(path):
    with open(path_cluster_file, 'r') as f:
        content_str = f.read().splitlines()[0]
    clusters_str = content_str.split(",")      
    return clusters_str


def plot(clusters):
    plt.figure()
    plt.bar(np.arange(0, len(clusters)), clusters.values())
    plt.xticks(ticks=np.arange(0,len(clusters)), labels=clusters.keys(), rotation=-65)
    plt.ylabel("Frequency")
    plt.xlabel("Cluster channels")
    plt.tight_layout()


parser = argparse.ArgumentParser()
parser.add_argument("-root_folder", required=False, type=str, default=r"D:\Speciale\Code\output\DSC_New\Runs_3264_relu")
parser.add_argument("-filename_cluster", required=False, type=str, default="clusters_subtract_complete.txt")
parser.add_argument("-foldername_cluster", required=False, type=str, default="clustering")


if __name__ == "__main__":
    args = parser.parse_args()

    folders_run = os.listdir(args.root_folder)

    all_labels = extract_labels_from_runs(args.root_folder, folders_run, "cluster_levels_norm_division_single.txt", exclusion_folders='AggregationAnalysis')
    
    clusters_by_k = []
    for k_idx in range(0, all_labels.shape[1]):
        clusters = {}
        for run_idx in range(0, all_labels.shape[0]):
            run_clusters = []
            unqs = np.unique(all_labels[run_idx, k_idx, :])
            for unq in unqs:
                ch_cluster = np.where(all_labels[run_idx, k_idx, :] == unq)[0]
                cluster_key = "".join([str(ch) for ch in ch_cluster])

                if cluster_key not in clusters:
                    clusters[cluster_key] = 0
                
                clusters[cluster_key] += 1
        
        clusters = {k: v for k,v in reversed(sorted(clusters.items(), key=lambda item: item[1]))}
        clusters_by_k.append(clusters)

    
    # cummulative
    clusters_cummulative_by_k = []
    for clusters in clusters_by_k:
        clusters_cummulative = {}
        for key,value in clusters.items():
            if len(key) == 1:
                clusters_cummulative[key] = value
                continue
            else:
                clusters_cummulative[key] = 0

            for key2,value2 in clusters.items():
                key_supported = True
                for char in key:
                    if char not in key2:
                        key_supported = False
                if key_supported:
                    clusters_cummulative[key] += value2
        clusters_cummulative = {k: v for k,v in reversed(sorted(clusters_cummulative.items(), key=lambda item: item[1]))}
        clusters_cummulative_by_k.append(clusters_cummulative)   


    cluster_index = 0
    plt.figure()
    cluster = clusters_by_k[cluster_index]
    plt.bar(np.arange(0, len(cluster)), np.array(list(cluster.values()))/all_labels.shape[0])
    plt.xticks(ticks=np.arange(0, len(cluster)), labels=list(cluster.keys()), rotation=-65)
    plt.tight_layout()

    plt.figure()
    cluster = clusters_cummulative_by_k[cluster_index]
    plt.bar(np.arange(0, len(cluster)), np.array(list(cluster.values()))/all_labels.shape[0])
    plt.xticks(ticks=np.arange(0, len(cluster)), labels=list(cluster.keys()), rotation=-65)
    plt.tight_layout()
    plt.show()
    exit(0)

    clusters = {}
    n_runs = 0
    for folder in folders_run:
        path_cluster_file = os.path.join(args.root_folder, folder, args.foldername_cluster, args.filename_cluster)

        if not os.path.isfile(path_cluster_file):
            print("WARNING: Could not find %s!" % path_cluster_file)
            continue
        
        for clust_str in read_cluster_file(path_cluster_file):
            sorted_clust_str = "-".join(sorted(clust_str.split("-")))       
            if sorted_clust_str not in clusters:
                clusters[sorted_clust_str] = 1
            else:
                clusters[sorted_clust_str] = clusters[sorted_clust_str] + 1
        
        n_runs += 1

    # Cummulative clustering
    clusters_cummulative = {}
    for key,val in clusters.items():
        members = sorted([int(s) for s in key.split("-")])
        all_combs = []
        for j in range(len(members)):
            combs = get_possible_channel_combinations(j + 1, members)
            all_combs.extend(combs)
        
        for comb in all_combs:
            comb_str = "-".join([str(ch) for ch in comb])
            if comb_str in clusters_cummulative:
                clusters_cummulative[comb_str] = clusters_cummulative[comb_str] + val
            else:
                clusters_cummulative[comb_str] = val

    # Convert to frequency
    for key,val in clusters.items():
        clusters[key] = val/n_runs
    for key,val in clusters_cummulative.items():
        clusters_cummulative[key] = val/n_runs

    # Sort dictionary
    clusters = {k: v for k,v in reversed(sorted(clusters.items(), key=lambda item: item[1]))}
    clusters_cummulative = {k: v for k,v in reversed(sorted(clusters_cummulative.items(), key=lambda item: item[1]))}

    # Plot
    plot(clusters)
    plot(clusters_cummulative)
    plt.show()

