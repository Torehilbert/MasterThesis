import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Autoencoder.dsc_clustering_ng import read_cluster_sets

parser = argparse.ArgumentParser()
parser.add_argument("-root_folder", required=False, type=str, default=None)
parser.add_argument("-foldername_clustering", required=False, type=str, default="clustering")
parser.add_argument("-filename_clusters", required=False, type=str, default="clusters_single.txt")
parser.add_argument("-filename_z_matrix", required=False, type=str, default="Z.csv")


def compute_similarity_scores(N, Z):
    # get candidate list
    cands = generate_combinations_recursively(N, [], [i for i in range(Z.shape[0])])
      
    # get scores
    S1 = np.zeros(shape=(len(cands)))
    S2 = np.zeros(shape=(len(cands)))
    for i,cand in enumerate(cands):
        # get non-candidate list
        non_cands = []
        for mem in range(Z.shape[0]):
            if mem not in cand:
                non_cands.append(mem)

        # sim score
        sim_score = 0
        dissim_score = 0
        for mem in cand:
            for mem2 in cand:
                if mem != mem2:
                    sim_score += Z[mem, mem2]
            for mem2 in non_cands:
                dissim_score += Z[mem, mem2]

        S1[i] = sim_score
        S2[i] = dissim_score
    S1 = (S1 - np.min(S1))/(np.max(S1) - np.min(S1)) + 1
    S2 = (S2 - np.min(S2))/(np.max(S2) - np.min(S2)) + 1
    scores = S2/S1

    # sort by scores
    sort_idx = np.array(list(reversed(np.argsort(scores))))
    S1 = np.array([S1[idx] for idx in sort_idx])
    S2 = np.array([S2[idx] for idx in sort_idx])
    scores = np.array([scores[idx] for idx in sort_idx])

    # return
    return S1, S2, scores, cands


def generate_combinations_recursively(N, selections, choices):    
    if len(selections) == N:
        return [selections]

    combs = []
    for i,choice in enumerate(choices):
        subselection = selections + [choice]
        subcombs = generate_combinations_recursively(N, subselection, choices[i+1:])
        combs.extend(subcombs)

    return combs


def convert_cands_to_strings(cands):
    strs = []
    for cand in cands:
        strs.append("-".join([str(i) for i in cand]))
    return strs


def fetch_performance_series(path_folder, cand_str, metric='val_acc'):
    path_performance_cand = os.path.join(path_folder, cand_str.replace("-", ""), 'series.txt')
    return np.max(pd.read_csv(path_performance_cand).loc[:,'val_acc'])


def smooth_array(arr, radius):
    n = len(arr)
    arr_smoothed = np.zeros(shape=(n,))
    for i in range(n):
        elements_to_average = []
        for j in range(i - radius, i + radius + 1):
            if j >= 0 and j < len(arr):
                elements_to_average.append(arr[j])
        arr_smoothed[i] = np.mean(elements_to_average)
    return arr_smoothed


PATH_RAND2 = r"D:\Speciale\Code\output\Performance Trainings\CRANDOM_2"
PATH_Z = r"D:\Speciale\Code\output\DSC_New\Runs_3264_relu\AggregationAnalysis\matrices\Z.csv"


if __name__ == "__main__":
    S1, S2, scores, cands = compute_similarity_scores(2, np.loadtxt(PATH_Z, delimiter=','))   
    performances = [fetch_performance_series(PATH_RAND2, cstr) for cstr in convert_cands_to_strings(cands)]
    performances_smoothed = smooth_array(performances, radius=7)

    ### Plots
    # scores
    plt.figure(figsize=(4,3))
    for i,(arr, ylabel) in enumerate(zip([S1, S2, scores], ["$S_1$", "$S_2$", "Score ($\\frac{S_2}{S_1}$)"])):
        plt.subplot(3,1,i+1)
        plt.plot(np.arange(0, len(arr)), arr, '.')
        plt.xticks(ticks=[])
        plt.ylabel(ylabel)
    plt.tight_layout()

    plt.figure(figsize=(5,3))
    plt.plot(np.arange(0, len(scores)), performances, 'o')
    plt.plot(np.arange(0, len(scores)), performances_smoothed, '-r')
    plt.xticks(ticks=[])
    plt.xlabel("Channel subset (sorted by score)")
    plt.ylabel("Validation accuracy")
    plt.ylim([np.min(performances) * 0.995, 1.005 * np.max(performances)])
    plt.legend(['Raw', 'Smoothed'])
    plt.grid()
    plt.tight_layout()

    plt.show()
