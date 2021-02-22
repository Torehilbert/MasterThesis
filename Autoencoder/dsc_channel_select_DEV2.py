import numpy as np
import matplotlib.pyplot as plt

from dsc_clustering_ng import read_cluster_sets
from dsc_channel_select_DEV import recursive_combs, evaluate_total_intra_similarity


def recursive_cluster_selection(n, chosen_clusters, remaining_clusters):
    n_left_to_choose = n - len(chosen_clusters)
    
    if n_left_to_choose == 1:
        return [chosen_clusters + [cluster] for cluster in remaining_clusters]
 
    combs = []
    for i,cluster in enumerate(remaining_clusters):
        subcombs = recursive_cluster_selection(n, chosen_clusters + [cluster], remaining_clusters[i+1:])
        combs.extend(subcombs)
    return combs


def recursive_channel_selection(chosen_chs, clusters):
    if len(clusters)==1:
        return [chosen_chs + [channel] for channel in clusters[0]]
    
    combs = []
    for i, channel in enumerate(clusters[0]):
        subcombs = recursive_channel_selection(chosen_chs + [channel], clusters[1:])
        combs.extend(subcombs)
    return combs


def calculate_combination_subscores(channels, cluster_set, Z, entropies=None):
    f1 = np.mean
    f3 = np.mean

    clusters_included = []
    clusters_not_included = []
    for cluster in cluster_set:
        included = sum([ch in cluster for ch in channels]) > 0
        if included:
            clusters_included.append(cluster)
        else:
            clusters_not_included.append(cluster)
    
    # similarity between chosen channels
    intra_similarity = evaluate_total_intra_similarity(Z, channels, func=f1)

    # similarity between non-included channels and selected channels
    non_included_channels = []
    for i in range(10):
        if i not in channels:
            non_included_channels.append(i)
    
    pairs = generate_cross_pairs(channels, non_included_channels)
    non_include_similarity = f3([Z[p[0], p[1]] for p in pairs])

    return intra_similarity, non_include_similarity


def generate_cross_pairs(choices_primary, choices_secondary):
    pairs = []
    for ch1 in choices_primary:
        for ch2 in choices_secondary:
            pairs.append(list(sorted([ch1, ch2])))
    return pairs


def aggregate_subscores(subscores):
    vmins = np.min(subscores, axis=0, keepdims=True)
    vmaxs = np.max(subscores, axis=0, keepdims=True)

    span = vmaxs - vmins
    span[np.where(np.isclose(span, 0))[0]] = 1

    subscores = (subscores - vmins) / span

    return subscores[:,1] - subscores[:,0]


MINIMUM_CLUSTERS = 6
PATH_Z = r"D:\Speciale\Code\output\DSC_New\Runs_3264_relu\2020-12-08--14-43-12_DSC_3264_relu\matrices\Z_norm.csv"
PATH_CLUSTER_SET = r"D:\Speciale\Code\output\DSC_New\Runs_3264_relu\2020-12-08--14-43-12_DSC_3264_relu\clustering\cluster_levels_norm_division_single.txt"
PATH_ENTROPY = r"D:\Speciale\Code\output\SimpleML\Ranking Entropy - Real (Correct)\entropy_scores.csv"


if __name__ == "__main__":
    cluster_sets = list(reversed(read_cluster_sets(PATH_CLUSTER_SET)))
    Z = np.loadtxt(PATH_Z, delimiter=",")
    cluster_set = [list(cluster) for cluster in cluster_sets[MINIMUM_CLUSTERS - 1]]
    entropies = np.loadtxt(PATH_ENTROPY, delimiter=",")

    for N in range(1,MINIMUM_CLUSTERS+1):
        cluster_combs = recursive_cluster_selection(N, [], cluster_set)
        channel_combs = []
        for c_comb in cluster_combs:
            combs = recursive_channel_selection([], c_comb)
            channel_combs.extend([list(sorted(comb)) for comb in combs])

        subscores = []
        for ch_comb in channel_combs:
            intra_sim, non_incl_sim = calculate_combination_subscores(ch_comb, cluster_set, Z)
            subscores.append([intra_sim, non_incl_sim])
        subscores = np.stack(subscores)
        scores = aggregate_subscores(subscores)
        print("N=%d channels:" % N)
        for i, idx in enumerate(list(reversed(np.argsort(scores)))[0:3]):
            print("%d. Combination" % (i+1), channel_combs[idx], "with score = %.3f" % scores[idx])
        
        idx_best = list(reversed(np.argsort(scores)))[0]     
        print("Best combo: ", channel_combs[idx_best])


