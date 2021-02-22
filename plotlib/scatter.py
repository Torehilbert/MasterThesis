import os
import matplotlib.pyplot as plt
from .style import COLORS, SCATTER_MARKER_SIZE

def scatter(x, y, color_index=0, title=None, show=True, save_path=None, xlabel=None, ylabel=None, append=False, legends=None):
    # create figure
    if not append:
        plt.figure()
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

    # plot
    plt.scatter(x, y, color=COLORS[color_index], s=SCATTER_MARKER_SIZE, alpha=0.5)

    # add legends
    if legends is not None:
        plt.legend(legends)

    # save plot
    if save_path is not None:
        if os.path.isdir(os.path.dirname(save_path)):
            plt.savefig(save_path)
        else:
            print("ERROR: Something is wrong with save_path: %s" % save_path)

    # show plot
    if show:
        plt.show()


