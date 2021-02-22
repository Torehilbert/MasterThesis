import os
import sys
import random
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plotlib.scatter import scatter
from plotlib.style import COLORS, COLOR_NEGATIVE, COLOR_NEUTRAL, COLOR_POSITIVE

parser = argparse.ArgumentParser()
parser.add_argument("-src", required=False, type=str, default=r"D:\Speciale\Repos\cell crop phantom\output\ExploreSimpleStats\s_13932\data.csv")


BOXPLOTS = False
CORRMATRIX = True
PAIRPLOTS = False


if __name__ == "__main__":
    modes = ['Aperture', 'ApodizedAP', 'BrightField', 'DarkField', 'DFIOpen', 'DFIPhase', 'DPI', 'iSSC', 'Phase', 'UVPhase']
    args = parser.parse_args()
    
    df = pd.read_csv(args.src)

    # Rename columns
    vartypes = ["mean", "std", "n"]
    vartypes_new = ["$\mu_{cell}$", "$\sigma_{cell}$", "$size_{cell}$"]
    rename_dict = {}
    for mode in modes:
        for varname, varname_new in zip(vartypes, vartypes_new):
            rename_dict['%s (images_%s)' % (varname, mode)] = "%s (%s)" % (varname_new, mode)
    df = df.rename(columns=rename_dict)

    # Drop redundant size columns
    columns_to_drop = ["%s (%s)" % (vartypes_new[2], mode) for mode in modes]
    del columns_to_drop[0]
    df = df.drop(columns=columns_to_drop)
    df = df.rename(columns={('%s (%s)' % (vartypes_new[2], modes[0])):vartypes_new[2]})

    ### BOXPLOTS
    # Rearrange data for boxplots
    if BOXPLOTS:
        nbase = df.values.shape[0]
        data_boxplot = np.empty(shape=(nbase*10, 1+1+3), dtype=np.float64)
        
        cursor = 0
        for i in range(len(modes)):
            istart = cursor
            iend = cursor + nbase
            varnames = ["%s (%s)" % (vtype, modes[i]) for vtype in vartypes_new[:-1]]
            data_boxplot[istart:iend, 0] = df['class'].values
            data_boxplot[istart:iend, 1] = i
            for j, vname in enumerate(varnames):
                data_boxplot[istart:iend, 2 + j] = df[vname].values
            data_boxplot[istart:iend, 4] = df['$size_{cell}$']
            cursor += nbase

        df_boxplot = pd.DataFrame(
            data=data_boxplot, 
            columns=['class', 'channel', vartypes_new[0], vartypes_new[1], vartypes_new[2]])
        df_boxplot['class'] = df_boxplot['class'].astype(int)
        df_boxplot['channel'] = df_boxplot['channel'].astype(int)
        df_boxplot[vartypes_new[2]] = df_boxplot[vartypes_new[2]].astype(int)

        # Means
        plt.figure(figsize=(12,4))
        sns.boxplot(
            x='channel', 
            y=vartypes_new[0], 
            hue='class', 
            data=df_boxplot, 
            palette=COLORS)
        plt.xticks(ticks=list(range(0,10)), labels=modes)
        plt.savefig("explore_box_mean.png", bbox_inches = 'tight', pad_inches=0)
        plt.clf()
        plt.close()

        # Stds
        plt.figure(figsize=(12,4))
        sns.boxplot(
            x='channel', 
            y=vartypes_new[1], 
            hue='class', 
            data=df_boxplot, 
            palette=COLORS)
        plt.xticks(ticks=list(range(0,10)), labels=modes)
        plt.savefig("explore_box_std.png", bbox_inches = 'tight', pad_inches=0)
        plt.clf()
        plt.close()

        # Sizes
        plt.figure(figsize=(2,2))
        sns.displot(df, 
            x=vartypes_new[2], 
            hue="class", 
            kind="kde", 
            palette=COLORS, 
            fill=True,
            height=3,
            aspect=3,
            legend=True)
        plt.savefig("explore_box_size.png", bbox_inches = 'tight', pad_inches=0)
        plt.clf()
        plt.close()

    ### CORRELATION MATRIX
    # Construct matrix
    if CORRMATRIX:
        cmap_corrmatrix = LinearSegmentedColormap.from_list("mymap", colors=[COLOR_NEGATIVE, COLOR_NEUTRAL, COLOR_POSITIVE])
        df_corrmatrix = df
        df_corrmatrix['Class 1'] = df['class']==1
        df_corrmatrix['Class 2'] = df['class']==2
        df_corrmatrix['Class 3'] = df['class']==3

        df_corrmatrix = df[['Class 1', 'Class 2', 'Class 3'] + [vartypes_new[2]] + ["%s (%s)" % (vartypes_new[0], mode) for mode in modes] + ["%s (%s)" % (vartypes_new[1], mode) for mode in modes]]
        corr_matrix = df_corrmatrix.corr()
        
        # Plot
        plt.figure(figsize=[10,9])
        sns.heatmap(corr_matrix, center=0, linewidths=.5, square=True, cmap=cmap_corrmatrix, vmin=-1, vmax=1)
        plt.savefig("explore_corrmatrix.png", bbox_inches = 'tight', pad_inches=0)
        plt.clf()
        plt.close()

    ### CORRELATION PLOTS
    if PAIRPLOTS:
        g = sns.pairplot(
            data=df, 
            hue='class', 
            x_vars=["%s (%s)" % (vartypes_new[0], mode) for mode in modes], 
            y_vars=["%s (%s)" % (vartypes_new[0], mode) for mode in modes] + [vartypes_new[2]], 
            plot_kws=dict(marker="o", linewidth=0.5, alpha=1.0, size=0.1), 
            palette=COLORS,
            height=1.5)

        plt.savefig("app_explore_n_mean.png")

