import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
from .style import COLORS


def plot_training_curve_from_file(path_to_file, xkey='epoch', ykeys=['loss'], color_ids=[0], linewidths=[1], linestyles=['solid'], show=True, save_path=None):
    training_data = pd.read_csv(path_to_file, header='infer')

    plt.figure()
    for i, ykey in enumerate(ykeys):
        X = training_data[xkey]
        Y = training_data[ykey]
        plt.plot(X,Y,
            color=COLORS[color_ids[i]] if len(color_ids) > i else color_ids[-1], 
            linewidth=linewidths[i] if len(linewidths) > i else linewidths[-1], 
            linestyle=linestyles[i] if len(linestyles) > i else linestyles[-1]
        )
    plt.legend(ykeys)
    plt.xlabel(xkey)
    
    # Save plot
    if save_path is not None:
        plt.savefig(save_path, dpi=500, bbox_inches='tight', pad_inches=0)
    
    # Show plot
    if show:
        plt.show()
    
    # Clean up
    plt.clf()
    plt.close()


if __name__ == "__main__":
    pass