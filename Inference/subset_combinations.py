import os
import numpy as np


def get_possible_channel_combinations(n, choices):
    # Expecting choices to list of integers, and n to be max size

    # Ensure choices are sorted (then combination channel order are correctly sorted)
    choices = sorted(choices)

    # Finding all combinations
    combs = []
    for ch in choices:
        new_combs = [[ch]]
        for existing_combination in combs:
            new_combination = existing_combination + [ch]
            new_combs.append(new_combination)
        
        for new_combination in new_combs:
            combs.append(new_combination)   

    # Remove combinations that doesn't contain n elements
    for i in reversed(range(len(combs))):
        comb = combs[i]
        if len(comb) != n:
            del combs[i]
    
    combs = sorted(combs, key=len)

    # Returning result
    return combs


def extract_accuracy_for_subset(path_data, subset, target_file="test_acc.txt", look_in_random_folder=False):
    filepaths = _filepaths_for_subset(path_data, subset, target_file, look_in_random_folder)
    accs = _fetch_accuracies_for_subset(filepaths)
    return np.mean(accs), np.std(accs, ddof=1) if len(accs) > 1 else 0


def extract_min_max_for_subset(path_data, subset, target_file="test_acc.txt", look_in_random_folder=False):
    filepaths = _filepaths_for_subset(path_data, subset, target_file, look_in_random_folder)     
    accs = _fetch_accuracies_for_subset(filepaths)
    return np.min(accs), np.max(accs)


def _filepaths_for_subset(path_data, subset, target_file="test_acc.txt", look_in_random_folder=False):
    # Generate filepaths
    filepaths = []
    path_subset_directory = os.path.join(path_data, "C"+subset)
    if os.path.isdir(path_subset_directory):
        for run in os.listdir(path_subset_directory):
            path_file = os.path.join(path_subset_directory, run, target_file)
            if os.path.isfile(path_file):
                filepaths.append(path_file)
            else:
                raise Exception("Could not find file: %s" % path_file)
    else:
        if look_in_random_folder:
            path_folder = os.path.join(path_data, "CRANDOM_"+str(len(subset)), subset)
            if os.path.isdir(path_folder):
                path_file = os.path.join(path_folder, target_file)
                if os.path.isfile(path_file):
                    filepaths.append(path_file)
                else:
                    raise Exception("Could not find file: %s" % path_file)
            else:
                raise NotADirectoryError("Could not find directory: %s or %s" % (path_subset_directory,path_folder))
        else:
            raise NotADirectoryError("Could not find directory: %s" % path_subset_directory)   
    return filepaths


def _fetch_accuracies_for_subset(filepaths):
    accs = []
    for fpath in filepaths:
        with open(fpath, 'r') as f:
            accs.append(float(f.readline().split(" ")[0]))
    return accs


def subset_str2list(subset_str):
    subset = []
    for c in subset_str:
        subset.append(int(c))
    return subset


def ensure_enough_list_elements(element_list, n):
    while len(element_list) < n:
        element_list.append(element_list[-1])
    return element_list
