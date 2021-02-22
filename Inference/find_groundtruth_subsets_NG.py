import os
import numpy as np
import matplotlib.pyplot as plt

PATH_TOP = r"D:\Speciale\Code\output\Performance Trainings"
PATH_RUNS = r"D:\Speciale\Code\output\Performance Trainings\CRANDOM_2"


def read_performance_of_run(path_run):
    path_test_acc = os.path.join(path_run, 'test_acc.txt')
    with open(path_test_acc, 'r') as f:
        acc = float(f.readline().split(" ")[0])
    return acc


if __name__ == "__main__":
    subfolders = os.listdir(PATH_RUNS)
    accs = []
    accs_max = []
    accs_min = []
    for folder in subfolders:
        path_run = os.path.join(PATH_RUNS, folder)
        acc = read_performance_of_run(path_run)
        accs.append(acc)

        reproduce_folder_name = "C" + folder
        path_reproduce = os.path.join(PATH_TOP, reproduce_folder_name)
        if os.path.isdir(path_reproduce):
            sub_accs = [read_performance_of_run(os.path.join(path_reproduce, run)) for run in os.listdir(path_reproduce)]
            accs_min.append(min(np.min(sub_accs), acc))
            accs_max.append(max(np.max(sub_accs), acc))
        else:
            accs_max.append(acc)
            accs_min.append(acc)
   
    accs = np.array(accs)
    accs_max = np.array(accs_max)
    accs_min = np.array(accs_min)
    yerrs = np.vstack((np.expand_dims(accs_min, axis=0), np.expand_dims(accs_max, axis=0)))

    idx_sort = np.array(list(reversed([idx for idx in np.argsort(accs)])))
    accs = accs[idx_sort]
    yerrs_min = np.expand_dims(accs, axis=0) - yerrs[0,idx_sort]
    yerrs_max = yerrs[1,idx_sort] - np.expand_dims(accs, axis=0)
    yerrs = np.vstack((yerrs_min, yerrs_max))


    # acc_max = np.max(accs)
    # folder_max = subfolders[np.argmax(accs)]
    # print(folder_max, "with an test accuracy of %.2f%%" % acc_max)
    print(yerrs_min.shape)
    
    plt.errorbar(np.arange(len(accs)), accs, yerr=yerrs, fmt='-ok')
    #plt.plot(np.array(accs)[idx_sort], '-ok')
    plt.xticks(ticks=np.arange(0, len(accs)), labels=[subfolders[idx] for idx in idx_sort], rotation=-90)
    plt.show()
