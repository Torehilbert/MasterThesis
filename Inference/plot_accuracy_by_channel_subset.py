import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='whitegrid')
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plotlib.style import COLORS_NO_CLASS
from subset_combinations import extract_accuracy_for_subset, extract_min_max_for_subset, ensure_enough_list_elements


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


SUBSET_ML_CORRELATION = "7-07-057-0357"
SUBSET_ML_ENTROPY = "6-26-268-2468"
SUBSET_TEACHER_STUDENT = "4-48-048-0478"    # done
SUBSET_CHANNEL_OCCLUSION = "9-09-089-0689"  # done
ZERO_CHANNEL_ACCURACY = 0.65398         # 23897/36541 in test set



parser= argparse.ArgumentParser()
parser.add_argument("-data", required=False, type=str, default=r"D:\Speciale\Code\output\Performance Trainings")
parser.add_argument("-method", required=False, type=str, default="minmax")  # minmax, mean, meanstd
parser.add_argument("-target_file", required=False, type=str, default="test_acc.txt")
parser.add_argument("-subset_series", required=False, type=str, nargs="+", default=["RANDOM_1", SUBSET_ML_CORRELATION, SUBSET_ML_ENTROPY, SUBSET_CHANNEL_OCCLUSION, SUBSET_TEACHER_STUDENT])
parser.add_argument("-legends", required=False, type=str, nargs="+", default=["Random", "Correlation Measure", "Entropy", "Channel Occlusion", "Teacher Student"])
parser.add_argument("-also_look_in_random_folders", required=False, type=int, nargs="+", default=[])
parser.add_argument("-xlabel", required=False, type=str, default="Number of channels selected")
parser.add_argument("-ylabel", required=False, type=str, default="Test Accuracy")
parser.add_argument("-ylim", required=False, type=float, nargs="+", default=None)
parser.add_argument("-xtickadd", required=False, type=float, default=1)
parser.add_argument("-figsize", required=False, type=float , nargs="+", default=[6.4, 4.8])

parser.add_argument("-use_error_bar", required=False, type=int, nargs="+", default=[1])
parser.add_argument("-fmt", required=False, type=str, default=None)
parser.add_argument("-linewidths", required=False, type=float, nargs="+", default=[2.5])
parser.add_argument("-markersizes", required=False, type=float, nargs="+", default=[7.5])
parser.add_argument("-err_linewidths", required=False, type=float, nargs="+", default=[1.5])
parser.add_argument("-colors", required=False, type=str, nargs="+", default=["black"])

parser.add_argument("-save", required=False, type=int, default=0)
parser.add_argument("-show", required=False, type=int, default=1)


if __name__ == "__main__":
    args= parser.parse_args()
    N = len(args.subset_series)

    args.fmt = args.fmt.split(" ") if args.fmt is not None else None

    args.colors = ensure_enough_list_elements(args.colors, N)
    args.use_error_bar = ensure_enough_list_elements(args.use_error_bar, N)
    args.linewidths = ensure_enough_list_elements(args.linewidths, N)
    args.markersizes = ensure_enough_list_elements(args.markersizes, N)
    args.err_linewidths = ensure_enough_list_elements(args.err_linewidths, N)


    # Setup two axes figure to enable broken y-axis
    HEIGHT_RATIO = 10
    f, (ax, ax2) = plt.subplots(2,1,figsize=args.figsize, gridspec_kw={'height_ratios': [HEIGHT_RATIO, 1]})
    ax.set_ylim(args.ylim[0], args.ylim[1])
    ax2.set_ylim(0, 0.075)
    ax.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax.set_xticks(ticks=[])
    ax2.set_yticks(ticks=[0])

    # Perform plotting
    len_max = 0
    for i,sss in enumerate(args.subset_series):
        subsets = sss.split("-")

        if args.method == "meanstd":
            acc_means, acc_stds = extract_accuracies_for_subsets(args.data, subsets, target_file=args.target_file, look_in_random_folder=args.also_look_in_random_folders[i])
            
            X = np.arange(len(acc_means)) + args.xtickadd
            Y = acc_means
            YERR = acc_stds if args.use_error_bar[i] == 1 else None

        elif args.method == "minmax":
            acc_means, _ = extract_accuracies_for_subsets(args.data, subsets, target_file=args.target_file, look_in_random_folder=args.also_look_in_random_folders[i])
            acc_min, acc_max = extract_min_max_acc_for_subsets(args.data, subsets, target_file=args.target_file, look_in_random_folder=args.also_look_in_random_folders[i])
            
            X = np.arange(len(acc_min)) + args.xtickadd
            Y = acc_means
            if args.use_error_bar[i] == 1:
                YERR = np.zeros(shape=(2,len(acc_means)))
                YERR[0,:] = np.array(acc_means) - np.array(acc_min)
                YERR[1,:] = np.array(acc_max) - np.array(acc_means)
            else:
                YERR = None


        ax.errorbar(
                x=X, 
                y=Y, 
                yerr=YERR, 
                fmt=args.fmt[i] if len(args.fmt) > i else args.fmt[-1], 
                ecolor=args.colors[i], 
                linewidth=args.linewidths[i],
                markersize=args.markersizes[i],
                elinewidth=args.err_linewidths[i], 
                color=args.colors[i])
        
        len_max = len(acc_means) if len(acc_means) > len_max else len_max

    ax2.set_xticks(ticks=np.arange(len_max) + args.xtickadd)
    ax2.set_xlabel(args.xlabel)
    ax.set_ylabel(args.ylabel)
    ax.legend(args.legends)
    ax2.set_xlim(ax.get_xlim())

    # Paint broken axis indicators
    d=0.015
    kwargs = dict(transform=ax.transAxes, color='k', clip_on=False, linewidth=0.6)
    ax.plot((-d, +d), (-d, +d), **kwargs)
    ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)
    kwargs.update(transform=ax2.transAxes)
    ax2.plot((-d, +d), (1 - HEIGHT_RATIO*d, 1 + HEIGHT_RATIO*d), **kwargs)
    ax2.plot((1 - d, 1 + d), (1 - HEIGHT_RATIO*d, 1 + HEIGHT_RATIO*d), **kwargs)


    plt.tight_layout()

    # Save and shot plot
    if args.save==1:
        plt.savefig("results_ranking_performances.png", bbox_inches = 'tight', pad_inches=0, dpi=500)

    if args.show==1:
        plt.show()



