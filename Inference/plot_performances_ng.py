import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

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


def subplot_on_edge(idx, rows, cols):
    first_column = (idx % cols == 0)
    last_row = (idx // cols) == (rows - 1)
    return first_column, last_row


PATH_DATA = r"D:\Speciale\Code\output\Performance Trainings"
SUBSETS_RAND =      "RANDOM_1-RANDOM_2-RANDOM_3-RANDOM_4"
SUBSETS_BEST =      "2-24-247-2478"
SUBSETS_ENT =       "6-26-246-2468-24568-124568-1245678-01245678-012456789"
SUBSETS_CORR =      "6-26-256-2568-23568-023568-0235678-01235678-012356789" #"6-26-256-2568-23568-023568-0235678-01235678-012356789"

SUBSET_CHO_AVG =    "9-09-089-0689-01689-014689-0124689-01246789-012456789"
SUBSET_CHO_ZERO =   "0-02-028-0289-02789-024789-0124789-01245789-012345789" #-024789-0124789-01245789-012345789"
SUBSET_TS =         "4-48-048-0478-02478-024678-0245678-02345678-023456789"
SUBSET_DSC_S =      "0-04-025-0235-14678-146789-1346789-12346789-123456789" #"0-04-025-0235-14678-146789-1346789-12346789-123456789"
SUBSET_DSC_L =      "7-27-278-2478-23478-123478-1234678-01234678-012345678"
SUBSET_CHEM =       "13467"
SUBSET_CHEM_NEW =   "23489"

colors = sns.color_palette("tab10")
colors.insert(0, (0.0,0.0,0.0))
colors.insert(0, (0.5,0.5,0.5))

SUBPLOT_ROWS = 3
SUBPLOT_COLS = 3
YLIM = [0.93, 0.975]

subset_series = [SUBSETS_RAND, SUBSETS_BEST, SUBSETS_ENT, SUBSETS_CORR, SUBSET_CHO_ZERO, SUBSET_CHO_AVG, SUBSET_TS, SUBSET_DSC_S, SUBSET_DSC_L]
subset_labels = ["Ref-Random", "Ref-Best", "Entropy", "Correlation", "CO-Zero", "CO-Average","Teacher-Student", "DSC-Sim", "DSC-Loss"]
subset_color_idx = [0, 1, 2, 2, 3, 3, 4, 5, 5]
subset_linestyle = ["--","dotted","-","-","-","-","-","-","-"]
subset_marker = ['o','o','o','^','o','^','o','o','^']
subset_markersize = [7,7,7,7,7,7,7,7,7]
chem_markersizes = [7,7]

if __name__ == "__main__":


    # A) LOAD DATA
    #   Load chemometec reference values
    acc_chem_min, acc_chem_max = extract_min_max_acc_for_subsets(path_data=PATH_DATA, subsets=[SUBSET_CHEM, SUBSET_CHEM_NEW], target_file="test_acc.txt", look_in_random_folder=False)
    acc_chem_means, _ = extract_accuracies_for_subsets(path_data=PATH_DATA, subsets=[SUBSET_CHEM, SUBSET_CHEM_NEW], target_file="test_acc.txt", look_in_random_folder=False)
    
    #   Load subset series values
    accs_min = []
    accs_max = []
    accs_mean = []
    accs_yerrs = []
    for i,ss in enumerate(subset_series):
        subset = ss.split('-')
        a_min, a_max = extract_min_max_acc_for_subsets(path_data=PATH_DATA, subsets=subset, target_file="test_acc.txt", look_in_random_folder=False)
        a_mean, _ = extract_accuracies_for_subsets(path_data=PATH_DATA, subsets=subset, target_file="test_acc.txt", look_in_random_folder=False)
        yerr = np.zeros(shape=(2,len(a_mean)))
        yerr[0,:] = np.array(a_mean) - np.array(a_min)
        yerr[1,:] = np.array(a_max) - np.array(a_mean)

        accs_mean.append(a_mean)
        accs_min.append(a_min)
        accs_max.append(a_max)
        accs_yerrs.append(yerr)

    # B) Construct figure
    plt.figure(figsize=(8, 8))
    for i,ss_main in enumerate(subset_series[1:]):
        subsets_main = ss_main.split("-")
        subplot_idx = i+1
        ax = plt.subplot(SUBPLOT_ROWS, SUBPLOT_COLS, subplot_idx)

        #   background gray lines
        for j in range(len(subset_series)):
            plt.plot(np.arange(len(accs_mean[j])), accs_mean[j], color=[0.9,0.9,0.9],linewidth=1, linestyle=subset_linestyle[j], zorder=-32)    
        #   chemometec references
        plt.plot(len(SUBSET_CHEM)-1, acc_chem_means[0], color="black", alpha=0.15, linestyle="none", linewidth=1, marker="*", markersize=chem_markersizes[0])
        plt.plot(len(SUBSET_CHEM_NEW)-1, acc_chem_means[1], color="black", alpha=0.25, linestyle="none", linewidth=1, marker="+", markersize=chem_markersizes[1])

        #   main subset line
        series_index = i+1
        plt.errorbar(
            x=np.arange(len(accs_mean[series_index])), 
            y=accs_mean[series_index], 
            yerr=accs_yerrs[series_index], 
            color=colors[subset_color_idx[series_index]], 
            marker=subset_marker[series_index], 
            markersize=subset_markersize[series_index], 
            markeredgecolor='white', 
            markeredgewidth=0.5, 
            linestyle=subset_linestyle[series_index], 
            label=subset_labels[series_index])
        
        #   insert random reference and chemometec references in first plot
        if i==0:
            plt.errorbar(
                x=np.arange(len(accs_mean[0])), 
                y=accs_mean[0], 
                yerr=accs_yerrs[0], 
                elinewidth=1, 
                color=colors[subset_color_idx[0]], 
                marker=subset_marker[0], 
                markersize=subset_markersize[0], 
                markeredgecolor='white', 
                markeredgewidth=0.5, 
                linestyle=subset_linestyle[0], 
                label=subset_labels[0])

            plt.plot(len(SUBSET_CHEM)-1, acc_chem_means[0], color="black", alpha=0.75, linestyle="none", linewidth=1, marker="*", markersize=chem_markersizes[0], label="Ref-Chem")
            plt.plot(len(SUBSET_CHEM_NEW)-1, acc_chem_means[1], color="black", alpha=0.75, linestyle="none", linewidth=1, marker="+", markersize=chem_markersizes[1], label="Ref-Chem-v2")

        #   check for border plot
        on_edge_left, on_edge_bottom = subplot_on_edge(subplot_idx-1, SUBPLOT_ROWS, SUBPLOT_COLS)

        #   formatting
        plt.yticks(ticks=(np.arange(0.94,0.98, 0.01) if on_edge_left else []))
        plt.xticks(ticks=(np.arange(0,9) if on_edge_bottom else []), labels=(np.arange(1,10) if on_edge_bottom else []))
        plt.xlabel('Channels (N)' if on_edge_bottom else None)
        plt.ylabel('Test accuracy' if on_edge_left else None)
        plt.grid()
        plt.ylim(YLIM)
        plt.legend()
        plt.tick_params(axis='y', which='major', left=True, length=2, width=0.5)
        plt.tick_params(axis='x', which='major', bottom=True, length=2, width=0.5)
        
    # C) Save and show polt
    plt.tight_layout()
    plt.savefig("results_subset_performance_NG.pdf")
    plt.savefig("results_subset_performance_NG.png", dpi=250)
    plt.show()