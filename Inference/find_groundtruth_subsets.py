import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from plot_accuracy_by_channel_subset import extract_accuracy_for_subset
from subset_combinations import get_possible_channel_combinations, extract_accuracy_for_subset, subset_str2list




def find_best_subset(n, target_file="test_acc.txt", path_data=r"D:\Speciale\Code\output\Performance Trainings"):
    CHOICES = [0,1,2,3,4,5,6,7,8,9]

    combs = get_possible_channel_combinations(n, CHOICES)

    accuracies = []
    for comb in combs:
        comb_str = "".join([str(num) for num in comb])
        try:
            acc, _ = extract_accuracy_for_subset(path_data=path_data, subset=comb_str,target_file=target_file, look_in_random_folder=True)
            accuracies.append(acc)
        except NotADirectoryError:
            print("WARNING: Couldn't find directory for subset: ", comb)
            accuracies.append(0)
    
    accuracies = np.array(accuracies)
    best_subset_index = np.argmax(accuracies)
    best_subset = combs[best_subset_index]
    return best_subset, accuracies[best_subset_index]


# parser = argparse.ArgumentParser()
# parser.add_argument("-data", required=False, type=str, default=r"D:\Speciale\Code\output\Performance Trainings")
# parser.add_argument("-target_file", required=False, type=str, default="test_acc.txt")
# parser.add_argument("-colors", required=False, type=str, nargs="+", default=["tan", "darkgray","steelblue","salmon"])
# parser.add_argument("-legends", required=False, type=str, nargs="+", default=["Entropy", "Correlation", "Channel Occlusion", "Teacher-Student"])
# parser.add_argument("-figsize", required=False, type=float , nargs="+", default=[7.4, 4.0])
# parser.add_argument("-err_linewidths", required=False, type=float, nargs="+", default=[0.5])
# parser.add_argument("-markersizes", required=False, type=float, nargs="+", default=[5])
# parser.add_argument("-markertypes", required=False, type=str, nargs="+", default=['D', 's', '^', 'o'])
# parser.add_argument("-ylim", required=False, type=float, nargs="+", default=[0.90, 0.975])
# parser.add_argument("-save", required=False, type=int, default=1)
# parser.add_argument("-show", required=False, type=int, default=1)

if __name__ == "__main__":

    plt.figure()
    accs = []
    subsets = []
    for n in range(1, 11):
        subset, acc = find_best_subset(n)
        accs.append(acc)
        subsets.append(subset)
    x = np.arange(1,11)
    plt.plot(x, accs, '-ok')
    plt.xticks(ticks=x, labels=subsets, rotation=90)
    plt.tight_layout()
    plt.show()

