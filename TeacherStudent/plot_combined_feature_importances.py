import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plotlib.style import COLORS


parser = argparse.ArgumentParser()
parser.add_argument("-path_input_folder", required=False, type=str, nargs="+", default=[r"D:\Speciale\Code\output\TeacherStudent\Phantom Student 16\New Regularization\Teacher A", r"D:\Speciale\Code\output\TeacherStudent\Phantom Student 16\New Regularization\Teacher B", r"D:\Speciale\Code\output\TeacherStudent\Phantom Student 16\New Regularization\Teacher C"])
parser.add_argument("-path_output_folder", required=False, type=str, default=None)
parser.add_argument("-use_phantom_labels", required=False, type=int, default=1)


REAL_LABELS = ['Aperture', 'ApodizedAP', 'BrightField', 'DarkField', 'DFIOpen', 'DFIPhase', 'DPI', 'iSSC', 'Phase', 'UVPhase']
PHANTOM_LABELS = ['DPI', 'Discr1', 'Discr2', 'Discr3', 'Blurr1', 'Blurr2','Blurr3', 'Noisy1', 'Noisy2', 'Noisy3','X1','X2']


if __name__ == "__main__":
    args = parser.parse_args()
    LABELS = PHANTOM_LABELS if args.use_phantom_labels else REAL_LABELS

    file_paths = []
    for top_folder in args.path_input_folder:
        subfolder = os.listdir(top_folder)
        file_paths.extend([os.path.join(top_folder, sub, 'importance_scores.csv') for sub in subfolder])

    scores = []
    for fpath in file_paths:
        scores.append(pd.read_csv(fpath, header='infer')['scores'].values)

    data = np.stack(scores)
    data_mean = np.mean(data, axis=0)
    data_sd = np.std(data, axis=0)
    
    data_sd_normalized = data_sd / np.sum(data_mean)
    data_mean_normalized = data_mean / np.sum(data_mean)


    plt.figure(figsize=(3.5,3))
    plt.bar(x=list(range(0, data.shape[1])), height=data_mean_normalized, color=COLORS[0], yerr=data_sd_normalized, capsize=5)
    plt.xticks(ticks=list(range(0,data.shape[1])), labels=LABELS, rotation=90, fontsize=11)
    plt.ylabel('Normalized importance scores')
    plt.tight_layout()

    plt.savefig("ts_importance_scores_phantom.png", dpi=500)
    plt.show()

    print(data.shape)
    print(data_mean.shape)
    print(data_sd.shape)