import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='whitegrid')
import numpy as np
import argparse

parser = argparse.ArgumentParser()
#parser.add_argument("-path_file", required=False, type=str, default=r"D:\Speciale\Code\output\TeacherStudent\Real Student 16\Original Regularization\Teacher A\Run 1\importance_scores_progression.csv")
parser.add_argument("-path_file", required=False, type=str, default=r"D:\Speciale\Code\output\TeacherStudent\Real Student 16\New Regularization\Teacher B\Run 1\importance_scores_progression.csv")


CHANNELS = ["Aperture", "ApodizedAP", "BrightField", "DarkField", "DFIOpen", "DFIPhase", "DPI", "iSSC", "Phase", "UVPhase"]


if __name__ == "__main__":
    args = parser.parse_args()
    data = np.loadtxt(args.path_file, delimiter=',')
    row_sums = np.expand_dims(np.sum(data, axis=1), axis=1)

    data_normalized = data / row_sums
    print(data.shape)
    print(row_sums.shape)


    plt.figure(figsize=(7,3))
    for i in range(data.shape[1]):
        plt.plot(data_normalized[:,i])
    plt.ylim([0, 1.1 * np.max(data_normalized)])
    plt.xlabel("Epoch")
    plt.ylabel("Normalized score")
    plt.legend(CHANNELS, loc='upper left', bbox_to_anchor=(1.0, 1.05))
    plt.tight_layout()
    plt.savefig("teacher_student_importance_score_progress_new.pdf")
    plt.savefig("teacher_student_importance_score_progress_new.png", dpi=250)
    
    plt.show()