import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='whitegrid')
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Autoencoder.dsc_clustering_ng import convert_cluster_set_to_labels, read_cluster_sets, perform_hierarchical_clustering, create_dendrogram_plot
from plotlib.style import COLORS


def find_inference_folders(parent_folder, postfix="DSCNet_Inference"):
    folders_run = os.listdir(parent_folder)
    inference_folders = []
    for folder in folders_run:
        path_folder = os.path.join(parent_folder, folder)
        if not os.path.isdir(path_folder):
            continue

        for subfolder in os.listdir(path_folder):
            if not subfolder.endswith(postfix):
                continue

            path_inference_folder = os.path.join(path_folder, subfolder)
            inference_folders.append(path_inference_folder)
    return inference_folders


def find_files_in_inference_folders(path_folders, name_template="raw_losses_class_<REPLACE>.csv", replacement_values=["1","2","3"]):
    check_names = [name_template.replace("<REPLACE>", substitute) for substitute in replacement_values]

    loss_files = []
    for check_name in check_names:
        loss_files.append([])
        for path_folder in path_folders:
            path_file = os.path.join(path_folder, check_name)
            if os.path.isfile(path_file):            
                loss_files[-1].append(path_file)
    
    return loss_files


def find_cluster_infos(path_folders, folder_target_name="clustering", file_target_name="cluster_levels_norm_division_single.txt"):
    cluster_files = []
    for path_folder in path_folders:
        path_cluster_file = os.path.join(os.path.dirname(path_folder), "clustering", file_target_name)
        if os.path.isfile(path_cluster_file):
            cluster_files.append(path_cluster_file)
    return cluster_files


def parse_loss_values(path_loss_file, func=np.median):
    df = pd.read_csv(path_loss_file)
    n = df.values.shape[0]
    stat = func(df.values[:,:10], axis=0)
    return stat, n


def extract_class(loss_lists, class_code=0):
    if isinstance(class_code, int) and class_code <= len(loss_lists):
        return loss_lists[class_code]
    else:
        raise Exception("Error")


def mean_over_class(loss_lists, func=np.mean):
    stat_avg_over_class = []
    for i in range(len(stat_lists[0])):
        vals = []
        for cla in range(len(stat_lists)):
            vals.append(stat_lists[cla][i])
        vals = func(np.stack(vals), axis=0)
        stat_avg_over_class.append(vals)
    return stat_avg_over_class


def plot_stat(values, mean_values=None, min_values=None, max_values=None, color="indianred", color_minmax="black", fig=None, label="Noname", region=False, alpha=1, xticklabels=None):
    if fig is None:
        fig = plt.figure(figsize=(6,4))

    if region:
        plt.fill_between(np.arange(10), min_values, max_values, color=color_minmax, alpha=0.1, label=None)
        plt.plot(np.arange(10), values, color=color, alpha=alpha, linewidth=None, label=label)
    else:
        yerr = np.zeros(shape=(2, 10))
        yerr[0,:] = np.array(values) - np.array(min_values)
        yerr[1,:] = np.array(max_values) - np.array(values)
        plt.errorbar(np.arange(10), values, yerr=yerr, fmt='o', color=color, alpha=alpha, label=label)
    plt.xlabel("Channel")
    plt.ylabel("Loss")
    plt.xticks(ticks=np.arange(10), labels=xticklabels, rotation=-65)
    plt.tight_layout()
    return fig


def reorder_series(series, ordering=None):
    if ordering is None:
        return series
    else:
        series_new = [None] * len(series)
        for i in range(len(series)):
            idx_new = ordering[i]
            series_new[idx_new] = series[i]
        return np.array(series_new)


def choose_channels(channel_losses, clusters):
    ranking_inds = list(reversed([idx for idx in np.argsort(channel_losses)]))

    selections = []
    N_limit = 10 - len(clusters)
    for N in range(1, 10):
        if N <= N_limit:
            cluster_info = clusters[0]
        else:
            cluster_info = clusters[N-N_limit]
        
        inds_chosen = []
        clusters_included = []
        for candidate in ranking_inds:
            # Find cluster idx
            cand_cluster_idx = None
            for cluster_idx, cluster in enumerate(cluster_info):
                if candidate in cluster:
                    cand_cluster_idx = cluster_idx
                    break
            
            if cand_cluster_idx in clusters_included:
                continue
            else:
                inds_chosen.append(candidate)
                clusters_included.append(cand_cluster_idx)
                if len(inds_chosen) == N:
                    break
        selections.append(sorted(inds_chosen))
    return selections


PATH_ROOT = r"D:\Speciale\Code\output\DSC_New\Runs_3264_relu"
CHANNELS = ["Aperture", "ApodizedAP", "BrightField", "DarkField", "DFIOpen", "DFIPhase", "DPI", "iSSC", "Phase", "UVPhase"]
ORDER = lambda series: reorder_series(series, ordering=[5,3,2,0,8,9,7,1,4,6])
K_MIN = 5
SELECT_IDX = 0


if __name__ == "__main__":
    # Find inference files
    path_folders = find_inference_folders(PATH_ROOT)
    path_loss_files = find_files_in_inference_folders(path_folders)
    path_cluster_infos = find_cluster_infos(path_folders)

    # Parse loss values and calculate statistic (np.median)
    stat_lists = []
    n_lists = []
    for class_list in path_loss_files:
        stat_lists.append([])
        n_lists.append([])
        for path_file in class_list:
            stat, n = parse_loss_values(path_file)
            stat_lists[-1].append(stat)
            n_lists[-1].append(n)

    # plot for losses for each class
    fig = None
    for c in range(3):
        losses = extract_class(stat_lists, class_code=c)
        stat_min_all = np.min(np.stack(losses), axis=0)
        stat_avg_all = np.mean(np.stack(losses), axis=0)
        stat_max_all = np.max(np.stack(losses), axis=0)
        fig = plot_stat(
            values=ORDER(losses[SELECT_IDX]), 
            mean_values=ORDER(stat_avg_all), 
            min_values=ORDER(stat_min_all), 
            max_values=ORDER(stat_max_all), 
            color=COLORS[c], 
            color_minmax=COLORS[c],
            alpha=1.0,
            fig=fig,
            label=["Healthy", "Apoptosis", "Dead"][c],
            xticklabels=ORDER(CHANNELS))

    mean_losses = np.stack(mean_over_class(stat_lists))
    losses_selected = mean_losses[SELECT_IDX]
    plt.bar(np.arange(10), ORDER(losses_selected), color="black", alpha=0.35, label="Class average")
    plt.legend()
    plt.savefig("dsc_loss_by_channel.pdf")
    plt.savefig("dsc_loss_by_channel.png", dpi=250)
    
    # Figure showing rankings
    plt.figure(figsize=(6,4))
    sns.heatmap(mean_losses)
    print("L SEL: ", losses_selected)
    mean_losses_reordered = np.zeros_like(mean_losses)
    for i in range(mean_losses.shape[1]):
        idx_new = [5,3,2,0,8,9,7,1,4,6][i]
        mean_losses_reordered[:,idx_new] = mean_losses[:,i]
    

    plt.figure(figsize=(6,4))
    sns.heatmap(mean_losses_reordered)
    #sns.heatmap(mean_losses_reordered, cbar_kws={'format': ''})
    plt.xticks(ticks=np.arange(10)+0.5, labels=ORDER(CHANNELS), rotation=-65)
    plt.yticks(ticks=np.arange(26,step=2) + 0.5, labels=np.arange(26, step=2), rotation=0)
    plt.xlabel("Channel")
    plt.ylabel("Run")
    plt.tight_layout()
    plt.savefig("dsc_loss_by_run.pdf")
    plt.savefig("dsc_loss_by_run.png", dpi=250)

    cluster_set = list(reversed(read_cluster_sets(path_cluster_infos[SELECT_IDX])))[K_MIN-1:]
    reference_selections = choose_channels(losses_selected, cluster_set)
    print(reference_selections)

    agree_count = [0] * len(reference_selections)
    for i in range(mean_losses.shape[0]):
        cluster_sets = list(reversed(read_cluster_sets(path_cluster_infos[i])))[K_MIN-1:]
        losses = mean_losses[i]
        selections = choose_channels(losses, cluster_sets)

        for j, (ref, test) in enumerate(zip(reference_selections, selections)):
            agree = 1
            for ch in ref:
                if ch not in test:
                    agree = 0
                    # if j <= 2:
                    #     print(test)
                    break
            
            agree_count[j] += agree/mean_losses.shape[0]
    
    print(agree_count)


    plt.show()
