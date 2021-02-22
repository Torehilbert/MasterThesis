import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='whitegrid')
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from plot_accuracy_by_channel_subset import ensure_enough_list_elements
from subset_combinations import get_possible_channel_combinations, subset_str2list, extract_accuracy_for_subset, extract_min_max_for_subset


def extract_accuracies_for_combinations(path_data, n, choices, target_file="test_acc.txt"):
    combs = get_possible_channel_combinations(n, choices)
    accuracies = []
    accuracy_stds = []
    for comb in combs:
        subset_str = "".join([str(num) for num in comb])
        try:
            acc_mean, acc_std = extract_accuracy_for_subset(path_data, subset_str, target_file=target_file, look_in_random_folder=True)
            accuracies.append(acc_mean)
            accuracy_stds.append(acc_std)
        except NotADirectoryError:
            print("WARNING: Could not find subset: ", subset_str)

    return accuracies, accuracy_stds


def extract_min_max_for_combinations(path_data, n, choices, target_file="test_acc.txt", look_in_random_folder=False):
    combs = get_possible_channel_combinations(n, choices)
    acc_min = []
    acc_max = []
    for comb in combs:
        subset_str = "".join([str(num) for num in comb])
        try:
            a_min, a_max = extract_min_max_for_subset(path_data, subset_str, target_file=target_file, look_in_random_folder=look_in_random_folder)
            acc_min.append(a_min)
            acc_max.append(a_max)
        except NotADirectoryError:
            print("WARNING: Could not find subset: ", subset_str)

    return acc_min, acc_max


colors = sns.color_palette("tab10")

parser = argparse.ArgumentParser()
parser.add_argument("-data", required=False, type=str, default=r"D:\Speciale\Code\output\Performance Trainings")
parser.add_argument("-channel_choices", required=False, type=str, nargs="+", default=["246", "256", "028", "089", "048", "025", "278"])
parser.add_argument("-best_choices_ref", required=False, type=str, nargs="+", default=["2", "24", "247"])
parser.add_argument("-best_choice", required=False, type=str, default="2478")
#parser.add_argument("-channel_choices", required=False, type=str, nargs="+", default=["246", "256", "089", "048"])
parser.add_argument("-target_file", required=False, type=str, default="test_acc.txt")
parser.add_argument("-error_bar_type", required=False, type=str, default="minmax")  # minmax, std
parser.add_argument("-colors", required=False, type=str, nargs="+", default=colors)
parser.add_argument("-legends", required=False, type=str, nargs="+", default=["Entropy", "Correlation", "CO-Zero", "CO-Avergae", "Teacher-student", "DSC-sim", "DSC-loss"])
parser.add_argument("-figsize", required=False, type=float , nargs="+", default=[6,4])
parser.add_argument("-err_linewidths", required=False, type=float, nargs="+", default=[0.5])
parser.add_argument("-markersizes", required=False, type=float, nargs="+", default=[5])
parser.add_argument("-markertypes", required=False, type=str, nargs="+", default=['o', 'o', 'o', 'o', 'o', 'o', 'o'])
parser.add_argument("-ylim", required=False, type=float, nargs="+", default=[0.90, 0.975])
parser.add_argument("-save", required=False, type=int, default=1)
parser.add_argument("-show", required=False, type=int, default=1)

labels = ["Entropy", "Correlation", "CO-Zero", "CO-Avergae", "Teacher-Student", "DSC-Sim", "DSC-Loss"]
color_idx = [0,0,1,1,2,3,3]
markers = ['o','^','o','^','o','o','^']
markersizes = [7]*7

SUBSETS_ENT =       "6-26-246"
SUBSETS_CORR =      "6-26-256"
SUBSET_CHO_ZERO =   "0-02-028"
SUBSET_CHO_AVG =    "9-09-089"
SUBSET_TS =         "4-48-048"
SUBSET_DSC_S =      "0-04-025"
SUBSET_DSC_L =      "7-27-278"
ALL_SUBSETS = [SUBSETS_ENT, SUBSETS_CORR, SUBSET_CHO_ZERO, SUBSET_CHO_AVG, SUBSET_TS, SUBSET_DSC_S, SUBSET_DSC_L]


if __name__ == "__main__":
    args = parser.parse_args()
    
    N = len(args.channel_choices)
    # Convert from string subsets to list of channel indices
    all_choices = []
    for subset_str in args.channel_choices:
        subset = subset_str2list(subset_str)
        all_choices.append(subset)

    plt.figure(figsize=(8,8))
    # Best reference
    for i, subset_str in enumerate(args.best_choices_ref):
        acc_mean, acc_std = extract_accuracy_for_subset(args.data, subset_str, target_file=args.target_file, look_in_random_folder=False)
        plt.plot([i+1 - 0.4, i+1 + 0.4], [acc_mean]*2, color='black', markersize=3, label='Ref-Best' if i==0 else None, linestyle='solid')

    # Random reference
    for i, subset_str in enumerate(["RANDOM_1", "RANDOM_2", "RANDOM_3"]):
        acc_mean, acc_std = extract_accuracy_for_subset(args.data, subset_str, target_file=args.target_file, look_in_random_folder=False)
        plt.plot([i+1 - 0.4, i+1 + 0.4], [acc_mean]*2, color='gray', markersize=3, label='Ref-Random' if i==0 else None, linestyle='solid')

    N = len(all_choices)
    for i,choices in enumerate(all_choices):
        true_subsets = ALL_SUBSETS[i].split("-")
        for n in range(len(choices)):
            acc, stds = extract_accuracies_for_combinations(args.data, n+1, choices, target_file=args.target_file)    
            x = np.array([n+1]*len(acc)) + 0.40*(i - N//2)/(N)

            if len(true_subsets) > 1:
                subset = true_subsets[n]      
                acc_mean, acc_std = extract_accuracy_for_subset(args.data, subset, target_file=args.target_file, look_in_random_folder=False)                
                plt.plot(x[0], acc_mean, color=[0.2,0.2,0.2], marker="o", markersize=2, linestyle="None", zorder=300)

            # if len(true_subsets) == 1 and n!=2:
            #     x = []
            #     acc = []
            plt.plot(x, acc, color=sns.color_palette()[color_idx[i]], linestyle='solid', marker=markers[i], markersize=markersizes[i], label=labels[i] if n==0 else None)
            




    plt.xticks(ticks=[1,2,3])
    plt.xlabel("Sub-combination size")
    plt.ylabel("Test accuracy")
    plt.legend()
    plt.grid(axis='both')
    plt.tight_layout()

    plt.savefig("results_combinatory_performance_ng.pdf")
    plt.savefig("results_combinatory_performance_ng.png", dpi=250)
    plt.show()
