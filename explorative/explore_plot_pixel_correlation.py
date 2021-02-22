import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plotlib.style import COLORS, CMAP_NEGPOS


PATH_SRC = r"D:\Speciale\Repos\cell crop phantom\output\Pixel Data\s_13932\data.csv"
modes = ['Aperture', 'ApodizedAP', 'BrightField', 'DarkField', 'DFIOpen', 'DFIPhase', 'DPI', 'iSSC', 'Phase', 'UVPhase']


if __name__ == "__main__":
    # Load data
    df = pd.read_csv(PATH_SRC)

    # Rename columns
    rename_dict = {'images_%s' % mode:'%s' % mode  for mode in modes}
    df = df.rename(columns=rename_dict)

    # Add individual class variables
    df['Class 1'] = df['class']==1
    df['Class 2'] = df['class']==2
    df['Class 3'] = df['class']==3

    # Reorder and remove (class)
    df = df[['Class 1', 'Class 2', 'Class 3'] + ['%s'%mode for mode in modes ]]
    
    # Correlation marix
    corr_matrix = df.corr()
    
    # Plot
    plt.figure(figsize=[10,9])
    sns.heatmap(corr_matrix, center=0, linewidths=.5, square=True, cmap=CMAP_NEGPOS, vmin=-1, vmax=1)
    plt.savefig("explore_pixel_corrmatrix.png", bbox_inches = 'tight', pad_inches=0)
    plt.clf()
    plt.close()
