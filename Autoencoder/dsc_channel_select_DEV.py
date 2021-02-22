import numpy as np
import matplotlib.pyplot as plt

from dsc_clustering_ng import read_cluster_sets


def generate_combinations(clusters):
    pass

def recursive_combs(chosen_chs, clusters_rest):
    n_clusters_left = len(clusters_rest)
    if n_clusters_left == 1:
        return [chosen_chs + [ch] for ch in clusters_rest[0]]
    else:
        combs = []
        clusters_rest_duplicate = clusters_rest.copy()
        del clusters_rest_duplicate[0]

        for ch in clusters_rest[0]:
            combs.extend(recursive_combs(chosen_chs + [ch], clusters_rest_duplicate))
        return combs

def generate_pairs(choices):
    pairs = []
    for i in range(len(choices)):
        for j in range(i+1, len(choices)):
            pairs.append(list(sorted([choices[i], choices[j]])))
    return pairs



def evaluate_total_intra_similarity(Z, comb, func=np.mean):
    pairs = generate_pairs(comb)
    if len(pairs) == 0:
        return 0
    
    sim = np.zeros(shape=(len(pairs)))
    for i,pair in enumerate(pairs):
        sim[i] = Z[pair[0], pair[1]]
    return func(sim)


def evaluate_total_inter_similarity(Z, comb, num_channels=10, func=np.mean):
    comb = list(sorted(comb))
    rev_comb = []
    for i in range(num_channels):
        if i not in comb:
            rev_comb.append(i)
    
    sims = []
    for ch1 in comb:
        for ch2 in rev_comb:
            if ch2 > ch1:
                sims.append(Z[ch1, ch2])
    
    return func(sims)


if __name__ == "__main__":
    path_z = r"D:\Speciale\Code\output\DSC_New\Runs_3264_relu\2020-12-08--14-43-12_DSC_3264_relu\matrices\Z_norm.csv"
    path_d = r"D:\Speciale\Code\output\DSC_New\Runs_3264_relu\2020-12-08--14-43-12_DSC_3264_relu\matrices\D_norm_division.csv"
    path_clustering = r"D:\Speciale\Code\output\DSC_New\Runs_3264_relu\2020-12-08--14-43-12_DSC_3264_relu\clustering\cluster_levels_norm_division_single.txt"

    cluster_sets = list(reversed(read_cluster_sets(path_clustering)))
    Z = np.loadtxt(path_z, delimiter=",")
    D = np.loadtxt(path_d, delimiter=',')

    channel_selections = []
    for k_idx in range(1,len(cluster_sets) - 1):
        combs = recursive_combs([], cluster_sets[k_idx])
        intra_sims = []
        inter_sims = []
        for i,comb in enumerate(combs):        
            intra_sims.append(evaluate_total_intra_similarity(Z, comb, func=np.mean))
            inter_sims.append(evaluate_total_inter_similarity(Z, comb, func=np.mean))

        intra_sims = np.array(intra_sims)
        #intra_sims = (intra_sims - np.min(intra_sims))/(np.max(intra_sims) - np.min(intra_sims))
        inter_sims = np.array(inter_sims)
        #inter_sims = (inter_sims - np.min(inter_sims))/(np.max(inter_sims) - np.min(inter_sims))

        total_sim_score = (inter_sims/intra_sims)  #inter_sims - intra_sims
        channel_selections.append(combs[np.argmax(total_sim_score)])

        if k_idx == 1 or k_idx == 2 or k_idx == 3:
            idx_order = list(reversed([idx for idx in np.argsort(total_sim_score)]))
            combs_ordered = [combs[idx] for idx in idx_order]
            total_sims_ordered = [total_sim_score[idx] for idx in idx_order]
            intra_sims_ordered = [intra_sims[idx] for idx in idx_order]
            inter_sims_ordered = [inter_sims[idx] for idx in idx_order]
            

            plt.figure(figsize=(4.5, 3.5))
            plt.plot(intra_sims_ordered, '-o', color='royalblue', alpha=0.5)
            plt.plot(inter_sims_ordered, '-o', color='indianred', alpha=0.5)
            plt.plot(total_sims_ordered, '-o', color='black')
            plt.xticks(ticks=np.arange(len(intra_sims_ordered)), labels=["-".join([str(ch) for ch in comb]) for comb in combs_ordered], rotation=-65)
            plt.legend(['Intra-similarity mean (MS1)', 'Inter-similarity mean (MS2)', 'Subset score (MS2/MS1)'])
            plt.xlabel("Subset")
            plt.ylabel('Mean similarity')
            plt.tight_layout()
    
    channel_selections.insert(0, [channel_selections[0][np.argmax(np.mean(Z,axis=0)[channel_selections[0]])]])
    
    print(channel_selections)
    
    plt.show()
    exit(0)


    plt.figure()
    plt.imshow(Z)
    plt.show()