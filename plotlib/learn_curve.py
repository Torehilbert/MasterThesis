import matplotlib.pyplot as plt
import pandas as pd
import os
from .style import COLORS


def plot_learn_curve(x, acc, val_acc, color_index=0, use_error_rate=False, title=None, show=True, save_path=None):
    plt.figure()

    y_train = 100*acc if not use_error_rate else 100 - 100*acc
    y_val = 100*val_acc if not use_error_rate else 100 - 100*val_acc

    plt.plot(x, y_train, color=COLORS[color_index], linestyle='dashed', linewidth=0.75)
    plt.plot(x, y_val, color=COLORS[color_index])
    plt.legend(['Train', 'Validation'])

    plt.title(title)
    plt.xlabel('Iter.')
    plt.ylabel('%s' % ('Accuracy' if not use_error_rate else 'Error') + ' (%)' )
    plt.ylim([0, 100])
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

    if save_path is not None:
        if os.path.isdir(os.path.dirname(save_path)):
            plt.savefig(save_path)
        else:
            print("ERROR: Something is wrong with save_path: %s" % save_path)
    
    if show:
        plt.show()

    
def plot_accuracy_curve_from_file(path_to_file, x_name='iter', y_names=['acc', 'val_acc'], legends=['Train', 'Validation'], xlabel='iter.', ylabel='Accuracy', title=None, show=True, save=False, filename='accuracy.png'):
    df = pd.read_csv(path_to_file, header='infer')
    
    plt.figure()
    plt.plot(df[x_name], 100*df[y_names[0]], color=COLORS[0], linestyle='dashed', linewidth=0.75)
    plt.plot(df[x_name], 100*df[y_names[1]], color=COLORS[0])
    plt.legend(legends)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.ylim([0, 100])
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

    if save:
        filepath = os.path.join(os.path.dirname(path_to_file), filename)
        plt.savefig(filepath)
    
    if show:
        plt.show()


if __name__ == "__main__":
    path =r""
    plot_accuracy_curve_from_file(
        path_to_file=path,
        save=True,
        show=True)
