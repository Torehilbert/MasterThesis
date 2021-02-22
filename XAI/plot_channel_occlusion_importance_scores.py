import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
import seaborn as sns
import argparse
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plotlib.plot_channel_importance_scores import plot_channel_importance_scores

parser = argparse.ArgumentParser()
parser.add_argument("-files", required=False, type=str, nargs="+", default=[r"D:\Speciale\Code\output\ChannelOcclusion\Real\Run1\raw_stats.csv", r"D:\Speciale\Code\output\ChannelOcclusion\Real\Run2\raw_stats.csv", r"D:\Speciale\Code\output\ChannelOcclusion\Real\Run1\raw_stats.csv"])
parser.add_argument("-output_path", required=False, type=str, default="co_importance_scores_real")
parser.add_argument("-dpi", required=False, type=int, default=500)
parser.add_argument("-show", required=False, type=int, default=1)
parser.add_argument("-figsize", required=False, type=float, nargs="+", default=[7, 3.5])
parser.add_argument("-add_average", required=False, type=int, default=1)


if __name__ == "__main__":
    args = parser.parse_args()

    n_runs = len(args.files)
    n_channels = 10


    values = np.zeros(shape=(n_runs, n_channels))
    channels = None
    for i,f in enumerate(args.files):
        df = pd.read_csv(f)
        subset = df[df['class']=='all']
        values[i,:] = subset['dprob mean']
        channels = subset['channel']
    
    values = values/np.expand_dims(np.sum(values, axis=1),axis=1)

    plot_channel_importance_scores(
        scores=values,
        channel_names=channels,
        y_labels=["Net %d" % (i+1) for i in range(n_runs)],
        add_average_row=(args.add_average != 0),
        figsize=args.figsize,
        show=(args.show != 0),
        save_path=args.output_path,
        save_dpi=args.dpi
    )

