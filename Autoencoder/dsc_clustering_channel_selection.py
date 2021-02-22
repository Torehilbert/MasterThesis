import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wfutils import CHANNELS
import wfutils.log
from plotlib.style import COLORS

from Autoencoder.dsc_clustering import read_cluster_sets


def calculate_channel_depths(cluster_sets):
    cluster_sizes = np.zeros(shape=(10,10), dtype=int)

    for i,cluster_set in enumerate(cluster_sets):
        for cluster in cluster_set:
            for ch in cluster:
                cluster_sizes[i,ch] = len(cluster)

    return np.sum((cluster_sizes[:-1, :] - cluster_sizes[1:,:]) > 0, axis=0)


def extract_candidates(cluster_set, channel_depths=None):
    candidates = []
    for cluster in cluster_set:
        if channel_depths is not None:
            depths = np.array([channel_depths[idx] for idx in cluster])
            inds = np.where(depths == np.max(depths))[0]
            candidates.append([cluster[idx] for idx in inds])
        else:
            candidates.append([ch for ch in cluster])
    return candidates


def calculate_mwcd_for_candidates(candidates, cluster_set, D):
    out = []
    for cands, cluster in zip(candidates, cluster_set):
        # fast-forward if cluster with single element (within-cluster distance cannot be calculated!)
        if len(cluster)==1:
            out.append(tuple([1]))
            continue

        msd_all = []
        for cand in cands:
            msd_all.append(np.mean(np.square(np.array([D[cand, ch] for ch in cluster], dtype="float32"))))  # valid because D[i,i]=0
        out.append(tuple(msd_all))
    return out


def calculate_mbcd_for_candidates(candidates, cluster_set, D):
    # quick exit
    if len(cluster_set) == 1:
        return [tuple([1 for _ in candidates])]

    out = []
    for cands in candidates:
        belonging_cluster = None
        for cluster in cluster_set:
            if cands[0] in cluster:
                belonging_cluster = cluster
                break
        
        mean_distance = []
        # create non-candidates
        noncluster = []
        for ch in range(10):
            if ch not in belonging_cluster:
                noncluster.append(ch)

        # calculate distances
        for cand in cands:
            dist = np.mean(np.square([D[cand,ch] for ch in noncluster]))
            mean_distance.append(dist)

        out.append(tuple(mean_distance))
    
    return out


def select_channels(candidates, clusters, D):
    # calculate scores
    mwcds = calculate_mwcd_for_candidates(candidates, clusters, D)
    mbcds = calculate_mbcd_for_candidates(candidates, clusters, D)
    scores = [np.array(mbcd)/np.array(mwcd) for (mbcd, mwcd) in zip(mbcds,mwcds)]

    # select channels
    channel_selection = []
    for (cands, score) in zip(candidates, scores):
        choice_idx = np.argmax(score)
        choice = cands[choice_idx]
        channel_selection.append(choice)
    channel_selection = sorted(channel_selection)

    # return
    return channel_selection, scores, mwcds, mbcds


def select_channels_series(channel_depths=None):
    selections = []
    all_candidates = []
    all_scores = []
    all_mwcds = []
    all_mbcds = []

    for i in range(1,10):
        clusters = cluster_sets[i - 1]
        candidates = extract_candidates(clusters, channel_depths=channel_depths)
        channel_selection, scores, mwcds, mbcds = select_channels(candidates, clusters, D)

        selections.append(channel_selection)
        all_candidates.append(candidates)
        all_scores.append(scores)
        all_mwcds.append(mwcds)
        all_mbcds.append(mbcds)
    
    return selections, all_candidates, all_scores, all_mwcds, all_mbcds


def find_distance_file(path_cluster, fileending):
    conts = os.listdir(path_cluster)
    for cont in conts:
        if cont.endswith(fileending):
            return os.path.join(path_cluster, cont)
    return None


parser = argparse.ArgumentParser()
parser.add_argument("-path_cluster", required=False, type=str, default=r"D:\Speciale\Code\output\DSC_New\2020-12-03--15-15-54_DSC_Small\Clustering_Average_RowSum")
parser.add_argument("-filename_cluster_levels", required=False, type=str, default="cluster_levels.txt")
parser.add_argument("-fileending_distance_matrix", required=False, type=str, default="D.csv")
parser.add_argument("-plot_channel_depths", required=False, type=int, default=1)
parser.add_argument("-plot_score_plots", required=False, type=int, default=1)
parser.add_argument("-plot_score_channels_min", required=False, type=int, default=1)
parser.add_argument("-plot_score_channels_max", required=False, type=int, default=4)
parser.add_argument("-dpi", required=False, type=int, default=200)


if __name__ == "__main__":
    args = parser.parse_args()

    # Output folder
    path_output_folder = wfutils.log.create_output_folder("DSCNet_ChannelSelection")
    wfutils.log.log_arguments(path_output_folder, args)

    ### Load data
    #   load clusters
    path_cluster_sets = os.path.join(args.path_cluster, args.filename_cluster_levels)
    cluster_sets = list(reversed(read_cluster_sets(path_cluster_sets)))

    #   load distance matrix
    path_d_matrix = find_distance_file(args.path_cluster, args.fileending_distance_matrix)
    #path_d_matrix = os.path.join(args.path_cluster, args.filename_distance_matrix)
    D = np.loadtxt(path_d_matrix, delimiter=',')


    ### Channel Selection
    #   select candidates and select
    channel_depths = calculate_channel_depths(cluster_sets)

    for i, cdepths in enumerate([channel_depths, None]):
        selections, all_candidates, all_scores, all_mwcds, all_mbcds = select_channels_series(channel_depths=cdepths)
        
        # output files
        filenames_postfix = "_no_depths" if cdepths is None else ""
        with open(os.path.join(path_output_folder, 'selection%s.txt' % filenames_postfix), 'w') as f:
            for selection in selections:
                f.write(",".join([str(ch) for ch in selection]) + "\n")
    
        with open(os.path.join(path_output_folder, 'candidates%s.txt' % filenames_postfix), 'w') as f:
            for candidates in all_candidates:
                cluster_strs = []
                for cluster in candidates:
                    cluster_strs.append("-".join([str(ch) for ch in cluster]))
                f.write(",".join(cluster_strs) + "\n")

        if args.plot_score_plots == 1:
            for j in range(args.plot_score_channels_min-1, args.plot_score_channels_max):
                plt.figure()
                mbcds_selection = all_mbcds[j]
                mwcds_selection = all_mwcds[j]
                scores_selection = all_scores[j]
                candidates_selection = all_candidates[j]

                for k0,(VALS, ylabel) in enumerate(zip([mbcds_selection, mwcds_selection, scores_selection], ['Between-CMSD', 'Within-CMSD', 'Score (%)'])):
                    plt.subplot(3, 1, k0+1)
                    xcounter = 0
                    xs = []
                    labels = []
                    for k1,VALS_CLUSTER in enumerate(VALS):
                        x = []
                        y = []
                        for k2,VAL in enumerate(VALS_CLUSTER):
                            y.append(VAL/sum(VALS_CLUSTER) * (100 if k0==2 else 1))
                            xcounter += 1
                            x.append(xcounter)
                            xs.append(xcounter)
                            labels.append(CHANNELS[candidates_selection[k1][k2]])

                            if k2==len(VALS_CLUSTER)-1:
                                xcounter += 1

                        plt.bar(x, y, color=COLORS[k1] if len(COLORS) > k1 else COLORS[-1])
                    
                    if k0==0:
                        plt.legend(['Cluster %d' % C_IDX for C_IDX in range(len(VALS))])
                    if k0==2:
                        plt.xticks(xs, labels=labels, rotation=-65)
                    else:
                        plt.xticks(ticks=[])
                    plt.ylabel(ylabel)
                plt.tight_layout()
                plt.savefig(os.path.join(path_output_folder, "score_plot_%d_%s.png" % (j+1, 'depth' if i==0 else 'no_depth')), dpi=args.dpi)
                plt.clf()
                plt.close() 


    ### Plots
    #   channel depths
    if args.plot_channel_depths == 1:
        plt.figure()
        colors_depth_plot = []
        plt.bar(np.arange(0, len(channel_depths)), channel_depths)
        plt.xticks(np.arange(0,len(channel_depths)), labels=CHANNELS, rotation=-65)
        plt.ylabel('Depth')
        plt.tight_layout()
        plt.show()

