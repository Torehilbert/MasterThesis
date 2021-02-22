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
SUBSET_CHO_ZERO =   "0-02-028-0289-02789-024789-0124789-01245789-012345789"
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

subset_series = [SUBSETS_ENT, SUBSETS_CORR, SUBSET_CHO_ZERO, SUBSET_CHO_AVG, SUBSET_TS, SUBSET_DSC_S, SUBSET_DSC_L]
#subset_series = ["0123456789-2-24-247-2478-13467-23489"]
subset_labels = ["Entropy", "Correlation", "CO-Zero", "CO-Average","Teacher-Student", "DSC-Sim", "DSC-Loss"]
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
    accs_std = []
    accs_yerrs = []
    for i,ss in enumerate(subset_series):
        subset = ss.split('-')
        a_min, a_max = extract_min_max_acc_for_subsets(path_data=PATH_DATA, subsets=subset, target_file="test_acc.txt", look_in_random_folder=False)
        a_mean, a_std = extract_accuracies_for_subsets(path_data=PATH_DATA, subsets=subset, target_file="test_acc.txt", look_in_random_folder=False)
        yerr = np.zeros(shape=(2,len(a_mean)))
        yerr[0,:] = np.array(a_mean) - np.array(a_min)
        yerr[1,:] = np.array(a_max) - np.array(a_mean)

        accs_mean.append(a_mean)
        accs_std.append(a_std)
        accs_min.append(a_min)
        accs_max.append(a_max)
        accs_yerrs.append(yerr)
    
    metric_used = accs_max
    idx_max_by_n = []
    for n in range(9):
        idx_max = -1
        val_max = -1
        for i, accs in enumerate(metric_used):
            if len(accs) > n:
                if accs[n] > val_max:
                    val_max = accs[n]
                    idx_max = i

        idx_max_by_n.append(idx_max)


    #   print table content
    for n in range(9):
        print("%d" % (n+1), end="")
        for i, (accs, stds) in enumerate(zip(metric_used, accs_std)):            
            if len(accs) > n:
                str_std_deviation = "(\pm %s)" % ("%.2f" % (100*stds[n])).lstrip('0')
                if i==idx_max_by_n[n]:
                    print(" & $\mathbf{%.2f}\; %s$" % (100*accs[n], str_std_deviation), end="")
                else:
                    print(" & $%.2f\; %s$" % (100*accs[n], str_std_deviation), end="")
            else:
                print(" & n/a", end="")
        print(r"\\", end="")
        print("\n", end="")
