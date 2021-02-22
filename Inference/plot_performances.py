import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='whitegrid')
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from subset_combinations import extract_accuracy_for_subset, extract_min_max_for_subset



def extract_accuracies_for_subsets(path_data, subsets, target_file="test_acc.txt", look_in_random_folder=False):
    acc_means = []
    acc_stds = []

    for ss in subsets:
        acc_m, acc_s = extract_accuracy_for_subset(path_data, ss, target_file=target_file, look_in_random_folder=look_in_random_folder)
        acc_means.append(acc_m)
        acc_stds.append(acc_s)
    
    return acc_means, acc_stds


def extract_min_max_acc_for_subsets(path_data, subsets, target_file="test_acc.txt", look_in_random_folder=False):
    acc_min =[]
    acc_max = []

    for ss in subsets:
        a_min, a_max = extract_min_max_for_subset(path_data, ss, target_file=target_file, look_in_random_folder=look_in_random_folder)
        acc_min.append(a_min)
        acc_max.append(a_max)

    return acc_min, acc_max


PATH_DATA = r"D:\Speciale\Code\output\Performance Trainings"
SUBSETS_RAND =      "RANDOM_1-RANDOM_2-RANDOM_3-RANDOM_4"
SUBSETS_BEST =      "2-24-247-2478"
SUBSETS_ENT =       "6-26-246-2468-24568-124568-1245678-01245678-012456789"
SUBSETS_CORR =      "6-26-256-2568-23568-023568-0235678-01235678-012356789" #"6-26-256-2568-23568-023568-0235678-01235678-012356789"

SUBSET_CHO_AVG =    "9-09-089-0689-01689-014689" #-0124689-01246789-012456789"
SUBSET_CHO_ZERO =   "0-02-028-0289-02789-024789-0124789-01245789-012345789"
SUBSET_TS =         "4-48-048-0478-02478-024678-0245678-02345678-023456789"
SUBSET_DSC_S =      "0-04-025-0235-14678-146789-1346789-12346789-123456789" #"0-04-025-0235-14678-146789-1346789-12346789-123456789"
SUBSET_DSC_L =      "7-27-278-2478-23478-123478-1234678-01234678-012345678"
SUBSET_CHEM =       "13467"
SUBSET_CHEM_NEW =   "23489"

colors = sns.color_palette("tab10")
colors.insert(0, (0.0,0.0,0.0))
colors.insert(0, (0.5,0.5,0.5))


if __name__ == "__main__":
    subset_series = [SUBSETS_RAND, SUBSETS_BEST, SUBSETS_ENT, SUBSETS_CORR, SUBSET_CHO_ZERO, SUBSET_CHO_AVG, SUBSET_TS, SUBSET_DSC_S, SUBSET_DSC_L]
    subset_labels = ["Ref. (Random)", "Ref. (Best)", "Entropy", "Correlation", "CO-Zero", "CO-Average", "Teacher-Student", "DSC-Sim", "DSC-Loss"]
    subset_color_idx = [0, 1, 2, 2, 3, 3, 4, 5, 5]
    subset_linestyle = ["-","-","-","--","-","--","-","-","--"]

    #plt.figure(figsize=(6,4))
    plt.figure(figsize=(8, 5.3))

    # Chemometec reference
    for i, subset in enumerate([SUBSET_CHEM, SUBSET_CHEM_NEW]):
        acc_chem_min, acc_chem_max = extract_min_max_acc_for_subsets(
                        path_data=PATH_DATA, 
                        subsets=[subset], 
                        target_file="test_acc.txt", 
                        look_in_random_folder=False)
        acc_chem_means, _ = extract_accuracies_for_subsets(
                        path_data=PATH_DATA, 
                        subsets=[subset], 
                        target_file="test_acc.txt", 
                        look_in_random_folder=False)
        
        plt.plot(len(subset)-1, acc_chem_means, color="black", zorder=20, alpha=0.5, linestyle="none", linewidth=1, marker="*" if i==0 else "P", markersize=6, label="Ref. (ChemoMetec%s)" % ("-v2" if i==1 else "")) 

    for i, ss in enumerate(subset_series):
        subsets = ss.split("-")
        acc_min, acc_max = extract_min_max_acc_for_subsets(
            path_data=PATH_DATA, 
            subsets=subsets, 
            target_file="test_acc.txt", 
            look_in_random_folder=False)
        plt.fill_between(np.arange(len(acc_min)), acc_min, acc_max, facecolor=colors[subset_color_idx[i]], alpha=0.15 if i!=0 else 0.075, linewidth=0)
    

    for i,(ss,label) in enumerate(zip(subset_series, subset_labels)):
        subsets = ss.split("-")
        acc_means, _ = extract_accuracies_for_subsets(
            path_data=PATH_DATA, 
            subsets=subsets, 
            target_file="test_acc.txt", 
            look_in_random_folder=False)
        plt.plot(np.arange(len(acc_means)), acc_means, color=colors[subset_color_idx[i]], label=label, linestyle=subset_linestyle[i], linewidth=1 if i<=1 else None)

        # _, acc_max = extract_min_max_acc_for_subsets(
        #     path_data=PATH_DATA, 
        #     subsets=subsets, 
        #     target_file="test_acc.txt", 
        #     look_in_random_folder=False)
        # plt.plot(np.arange(len(acc_max)), acc_max, color=colors[subset_color_idx[i]], label=label, linestyle=subset_linestyle[i], linewidth=1 if i<=1 else None)

    plt.legend()
    plt.xlabel("Number of channels (N)")
    plt.ylabel("Test accuracy")
    plt.ylim([0.92,0.974])
    plt.xticks(ticks=np.arange(9), labels=[i+1 for i in np.arange(9)])
    plt.tight_layout()
    plt.savefig("results_subset_performance_L.pdf")
    plt.savefig("results_subset_performance_L.png", dpi=250)
    plt.show()