import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='whitegrid')
import os
import numpy as np


def read_acc(path_file):
    with open(path_file, 'r') as f:
        acc = float(f.readline().split(" ")[0])
    return acc


def get_accuracies(path_folder):
    subfolder = os.listdir(path_folder)
    accs = []
    for folder in subfolder:
        path_file = os.path.join(path_folder, folder, NAME_FILE)
        accs.append(read_acc(path_file))
    return np.array(accs)


def smooth_series(series, weights=[1,1,1]):
    N = len(series)
    order = len(weights)

    new_series = []
    for i in range(N):
        val = 0
        weight_sum = 0

        for j in range(order):
            idx_aim = i + (j - order//2)
            if idx_aim >= 0 and idx_aim < N-1:
                val += series[idx_aim] * weights[j]
                weight_sum += weights[j]
   
        new_series.append(val/weight_sum)
    return np.array(new_series)

PATH_BASE = r"D:\Speciale\Code\output\Performance Trainings"
END_FOLDERS = ["CRANDOM_1", "CRANDOM_2", "CRANDOM_3", "CRANDOM_4"]
mean_densities = [14.81, 50.27, 90.02, 120.50]
NAME_FILE = 'test_acc.txt'

ENTROPY = [('6', 21),('26',58.5), ('246',61.10),('2468',51.71)]
CORRELATION = [('6', 19), ('26', 53),('256', 101), ('2568', 129)]
CHANNEL_OCC_ZERO = [('0',8),('02',45.9),('028',35.4),('0289',128.8)]
CHANNEL_OCC_AVG = [('9',22.37),('09',43.48),('089',92.89),('0689',126)]
TEACHER_STUDENT = [('4',23.5),('48',47),('048',67.28),('0478',84.5)]
DSC_SIM = [('0',5.5),('04',54.7),('025',79),('0235',103)]
DSC_LOSS = [('7',8.8),('27',39.5),('278',28), ('2478',23)]


LABELS = ['Entropy', 'Correlation', 'CO-Zero', 'CO-Average', 'Teacher-Student', 'DSC-Sim', 'DSC-Loss']
COLORS = [sns.color_palette()[0], sns.color_palette()[0], sns.color_palette()[1], sns.color_palette()[1], sns.color_palette()[2], sns.color_palette()[3], sns.color_palette()[3]] 
MARKERS = ['o', '^', 'o', '^', 'o', 'o', '^']

if __name__ == "__main__":
    bins = [20, 20, 20, 20]
    widths = [0.003, 0.0019, 0.00115, 0.0009]
    smooths = [0.2, 0.2, 0.2, 0.2]

    for i,endfolder in enumerate(END_FOLDERS):
        path_folder = os.path.join(PATH_BASE, endfolder)
        accs = get_accuracies(path_folder)
        mean = np.mean(accs)

        plt.figure(figsize=(8,3))
        sns.distplot(accs, bins=25, color='gray')
        plt.plot(mean, mean_densities[i], 'ok', label='Average')

        for j,pair_series in enumerate([ENTROPY, CORRELATION, CHANNEL_OCC_ZERO, CHANNEL_OCC_AVG, TEACHER_STUDENT, DSC_SIM, DSC_LOSS]):
            strkey = pair_series[i][0]
            density = pair_series[i][1]
            path_subset = os.path.join(PATH_BASE, "C"+strkey)
            if os.path.isdir(path_subset):
                acc = np.mean(get_accuracies(path_subset))
                plt.plot(acc, density, label=LABELS[j], color=COLORS[j], marker=MARKERS[j], linestyle='none')
            else:
                print("WARNING: %s does not exist!" % path_subset)

        #xticklocs, xticklabels = plt.xticks()
        xticklocs = np.linspace(0.91, 0.97, 7)
        plt.xticks(ticks=xticklocs, labels=["%.1f" % (100*val) for val in xticklocs])
        plt.xlabel("Test accuracy (%)")

        #plt.legend(prop={'size': 10},bbox_to_anchor=(1.05, 1),loc='upper left')
        plt.legend()
        plt.xlim([0.9, 0.975])
        plt.tight_layout()
        name = "results_distribution_n%d" % (i+1)
        plt.savefig(name +".pdf")
        plt.savefig(name +".png", dpi=250)
    plt.show()