import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_channel_importance_scores(scores, channel_names, y_labels, add_average_row=True, figsize=(6.4, 4.8), show=True, save_path=None, save_dpi=500):
    # expecting scores to have shape (runs, channels)
    annotations = None
    fmt = None
    
    if add_average_row:
        scores_mean = np.expand_dims(np.mean(scores, axis=0), axis=0)
        scores = np.concatenate((scores, scores_mean), axis=0)

        # annotations
        annotations =  [["" for _ in range(scores.shape[1])] for _ in range(scores.shape[0])]
        annotations[-1] = ["%d" % (val+1) for val in np.argsort(np.flip(np.argsort(scores_mean)))[0]]
        fmt = 's'

        # add average labels to y
        y_labels.append("Average")

    # Plot
    plt.figure(figsize=figsize) 
    sns.heatmap(100*scores, cmap='Reds', annot=annotations, fmt=fmt, vmin=0, vmax=100, linewidths=2)

    if add_average_row:
        plt.hlines([scores.shape[0]-1], 0, scores.shape[1], linestyles='dashed')
    
    x = np.arange(scores.shape[1]) + 0.5
    plt.xticks(ticks=x+0.15, labels=channel_names, rotation=-65)
    plt.yticks(ticks=np.arange(len(y_labels)) + 0.5, labels=y_labels, rotation=0)
    plt.ylabel('Repetitions', fontsize=12)
    #plt.xlabel('Channel', fontsize=12)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path + ".pdf")
        plt.savefig(save_path + ".png", dpi=save_dpi)
    