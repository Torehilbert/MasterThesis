import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='whitegrid')
import pandas as pd
import numpy as np
import argparse
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plotlib.style import COLORS_NO_CLASS, CMAP_NEGPOS
from plotlib.plot_channel_importance_scores import plot_channel_importance_scores


parser = argparse.ArgumentParser()
# parser.add_argument("-path_input_folder", required=False, type=str, nargs="+", default=[r"D:\Speciale\Code\output\TeacherStudent\Real Student 16\Original Regularization\Teacher A",
#                                                                                          r"D:\Speciale\Code\output\TeacherStudent\Real Student 16\Original Regularization\Teacher B", 
#                                                                                          r"D:\Speciale\Code\output\TeacherStudent\Real Student 16\Original Regularization\Teacher C"])
parser.add_argument("-path_input_folder", required=False, type=str, nargs="+", default=[r"D:\Speciale\Code\output\TeacherStudent\Real Student 16\New Regularization\Teacher A",
                                                                                         r"D:\Speciale\Code\output\TeacherStudent\Real Student 16\New Regularization\Teacher B", 
                                                                                         r"D:\Speciale\Code\output\TeacherStudent\Real Student 16\New Regularization\Teacher C"])
parser.add_argument("-output_path", required=False, type=str, default="teacher_student_importance_scores_new")
parser.add_argument("-use_phantom_labels", required=False, type=int, default=0)
parser.add_argument("-dpi", required=False, type=int, default=500)
parser.add_argument("-show", required=False, type=int, default=1)
parser.add_argument("-figsize", required=False, type=float, nargs="+", default=[7, 3.5])
parser.add_argument("-add_average", required=False, type=int, default=1)


REAL_LABELS = ['Aperture', 'ApodizedAP', 'BrightField', 'DarkField', 'DFIOpen', 'DFIPhase', 'DPI', 'iSSC', 'Phase', 'UVPhase']
PHANTOM_LABELS = ['DPI', 'Discr1', 'Discr2', 'Discr3', 'Blurr1', 'Blurr2','Blurr3', 'Noisy1', 'Noisy2', 'Noisy3','X1','X2']
GROUPS = ['A', 'B', 'C', 'D', 'E']

if __name__ == "__main__":
    args = parser.parse_args()
    LABELS = PHANTOM_LABELS if args.use_phantom_labels else REAL_LABELS
    bar_width = 0.15

    file_paths = []
    group_labels = []
    for i, top_folder in enumerate(args.path_input_folder):
        subfolder = os.listdir(top_folder)
        file_paths.extend([os.path.join(top_folder, sub, 'importance_scores.csv') for sub in subfolder])
        group_labels.extend(["%s-%d" % (GROUPS[i], 1+j) for j in range(len(subfolder))])

    # extract scores
    scores = []
    for fpath in file_paths:
        scores.append(pd.read_csv(fpath, header='infer')['scores'].values)
        scores[-1] = scores[-1] / np.sum(scores[-1])
    scores = np.stack(scores)

    # Plot
    plot_channel_importance_scores(
        scores=scores,
        channel_names=LABELS,
        y_labels=group_labels,
        add_average_row=(args.add_average != 0),
        figsize=args.figsize,
        show=(args.show != 0),
        save_path=args.output_path,
        save_dpi=args.dpi
    )
