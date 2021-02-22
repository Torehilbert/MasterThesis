import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from style import COLORS
import argparse
import seaborn as sns
sns.set(style="whitegrid")


def plot_trainings(train_directories, x_id='epoch', y_id='val_acc', groups='AUTO', ylim=None, xlabel=None, ylabel=None, title=None, legends=None, summary=False, filename="noname"):
    paths_series = [os.path.join(directory, 'series.txt') for directory in train_directories]
    paths_args = [os.path.join(directory, 'args.txt') for directory in train_directories]
    

    # Error checking
    for i, directory in enumerate(train_directories):
        if not os.path.isdir(directory):
            raise Exception("Directory %s is not valid!" % directory)
        if not os.path.isfile(paths_series[i]):
            raise Exception("File %s does not exist" % paths_series[i])
        if not os.path.isfile(paths_args[i]):
            raise Exception("File %s does not exist" % paths_args[i])
    
    # Determining groups        
    if groups == 'AUTO' and len(train_directories) > 1:
        arg_dicts = [load_args(path) for path in paths_args]
        if not check_same_keys(arg_dicts):
            raise Exception("There is a mismatch in argument keys for the different series!")

        group_members, keys = auto_grouping(arg_dicts, args.keys_to_ignore)
        groups = {k:v for (k, v) in sorted(group_members.items())}
    else:
        keys = None
        groups = {k:[k] for k in range(len(paths_series))}

    # Plot
    keys_label = ",".join([key for key in keys])
    plt.figure(figsize=(6,3))
    ax = plt.subplot(1,1,1)
    legend_lines = []
    legend_names = []

    for i, (k,members) in enumerate(groups.items()):
        print("Group: ", i, len(members))
        X = None
        Y = None
        Ytraining = None
        for j, mem in enumerate(members):
            series = pd.read_csv(paths_series[mem], header='infer')

            # Instantiate X and Y matrices
            if X is None:       
                if args.max_x is not None:
                    Y = np.empty(shape=(args.max_x, len(members)))
                    X = np.linspace(1, args.max_x, args.max_x, endpoint=True)
                    Y[:] = np.nan
                else:
                    Y = np.empty(shape=(series[y_id].values.shape[0], len(members)))
                    X = series[x_id].values
                    Y[:] = np.nan
                if args.training == 1:
                    Ytraining = np.empty_like(Y)
                    Ytraining[:] = np.nan

            # Insert Y series in Y matrix
            n_series = series[y_id].values.shape[0]
            if n_series == Y.shape[0]:
                Y[:,j] = series[y_id].values
                if Ytraining is not None:
                    Ytraining[:,j] = series[args.training_key].values
            elif n_series > Y.shape[0]:
                Y[:,j] = series[y_id].values[0:Y.shape[0]]
                if Ytraining is not None:
                    Ytraining[:,j] = series[args.training_key].values[0:Y.shape[0]]
            else:
                Y[0:n_series,j] = series[y_id].values
                if Ytraining is not None:
                    Ytraining[0:n_series,j] = series[args.training_key].values
        
        # Plot summary
        if summary:
            if args.training == 1:
                y_train_mean = np.nanmean(Ytraining, axis=1)
                mask_nonnan = np.invert(np.isnan(y_train_mean))
                plt.plot(X[mask_nonnan], y_train_mean[mask_nonnan], color=COLORS[i], linewidth=0.5, linestyle='dashed')

            y_mean = np.nanmean(Y, axis=1)
            y_std = np.nanstd(Y, axis=1)
            mask_nonnan = np.invert(np.isnan(y_mean))
            ax.fill_between(X[mask_nonnan], y_mean[mask_nonnan] - y_std[mask_nonnan], y_mean[mask_nonnan] + y_std[mask_nonnan], color=COLORS[i], alpha=0.2)
            plt.plot(X[mask_nonnan], y_mean[mask_nonnan], color=COLORS[i], linewidth=2)
        else:
            for j in range(Y.shape[1]):
                plt.plot(X, Y[:,j], color=COLORS[i], linewidth=1)
        
        legend_lines.append(Line2D([0], [0], color=COLORS[i], lw=2))
        legend_names.append("%s=%s" % (try_translate(keys_label), k))    

    # # grid
    # plt.grid(alpha=0.2)

    # legends
    if keys is not None:
        if legends is not None:
            if len(legends) != len(legend_names):
                raise Exception("Specified legends have length %d, but found %d groups!" % (len(legends), len(legend_names)))
            plt.legend(legend_lines, legends)
        else:
            plt.legend(legend_lines, legend_names)

    # y-limits
    if ylim is not None:
        plt.ylim(ylim)

    # x-label
    if xlabel is not None:
        plt.xlabel(xlabel)
    else:
        plt.xlabel(try_translate(x_id))

    # y-label
    if ylabel is not None:
        plt.ylabel(ylabel)
    else:
        plt.ylabel(try_translate(y_id))

    # title
    if title is not None:
        plt.title(title)
    elif keys is not None:
        plt.title("%s for different %s" % (try_translate(y_id), try_translate(keys_label)))
    else:
        plt.title("%s" % try_translate(y_id))
    
    plt.tight_layout()

    plt.savefig("%s.pdf" % filename)
    plt.savefig("%s.png" % filename, dpi=250)
    plt.show()
    

def load_args(path_to_args):
    args = {}
    with open(path_to_args, 'r') as f:
        lines = f.readlines()
        for line in lines:
            split = line.split("=")
            args[split[0]] = split[1]
    return args


def check_same_keys(arg_dicts):
    key_lists = [list(d.keys()) for d in arg_dicts]
    ns = [len(l) for l in key_lists]

    # check same length of args
    for n in ns:
        if n!=ns[0]:
            return False

    # check same keys
    for key in key_lists[0]:
        for list_sec in key_lists:
            if key not in list_sec:
                return False
    
    return True


def auto_grouping(arg_dicts, keys_to_ignore=None):
    # Find keys with a difference in values
    keys = []
    for key in arg_dicts[0].keys():
        # skip this key
        if keys_to_ignore is not None and key in keys_to_ignore:
            continue
        # check other 
        val_ref = arg_dicts[0][key]
        for arg_d in arg_dicts:
            if arg_d[key] != val_ref:
                keys.append(key)
                break
    
    # Error check
    if len(keys) == 0:
        return None, None
    # elif len(keys) > 1:
    #     raise Exception("Differences in multiple keys are not yet supported!")

    # Group based on differences
    group_members = {}  # structure should be {0: [1,3,7], 1: [6,2,4], 2:[5]} 
    # key = keys_with_dif[0]
    # for i, arg_d in enumerate(arg_dicts):
    #     val = arg_d[key].replace("\n", "")
    #     if val in group_members.keys():
    #         group_members[val].append(i)
    #     else:
    #         group_members[val] = [i]

    # DEV: Group based on multiple-key differences
    for i, arg_d in enumerate(arg_dicts):
        val = ",".join([arg_d[key].replace("\n", "") for key in keys]) 
        if val in group_members.keys():
            group_members[val].append(i)
        else:
            group_members[val] = [i]

    # Return group member dict
    return group_members, keys
    

def try_translate(key):
    if key == 'acc':
        return 'accuracy'
    elif key == 'val_acc':
        return 'Validation accuracy'
    elif key == 'val_loss':
        return 'loss (val)'
    elif key == 'aug_noise':
        return 'noise level'
    elif key == 'epoch':
        return "Epoch"
    else:
        return key


def get_subdirs_in_dir(directory):
    dirs = [os.path.join(directory, folder) for folder in os.listdir(directory)]
    del_list = []
    for i, d in enumerate(dirs):
        if not os.path.isdir(d):
            del_list.append(i)

    for i in reversed(del_list):
        del dirs[i]
    
    return dirs


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-paths", required=False, type=str, nargs='+', default=None)
    parser.add_argument("-master_path", required=False, type=str, default=r"D:\Speciale\Code\output\TeacherStudent\Real Teacher 32")
    parser.add_argument("-legends", required=False, type=str, nargs='+', default=None)
    parser.add_argument("-title", required=False, type=str, default=None)
    parser.add_argument("-ylim", required=False, type=float, nargs='+', default=None)
    parser.add_argument("-summary", required=False, type=int, default=0)
    parser.add_argument("-max_x", required=False, type=int, default=None)
    parser.add_argument("-training", required=False, type=int, default=0)
    parser.add_argument("-training_key", required=False, type=str, default="acc")
    parser.add_argument("-keys_to_ignore", required=False, type=str, nargs="+", default=["path_training_data", "path_validation_data", "epochs"])
    parser.add_argument("-filename", required=False, type=str, default="noname")
    args = parser.parse_args()

    # Error checks
    if args.paths is None and args.master_path is None:
        raise Exception("The arguments \"-paths\" or \"-master_path\" must be supplied!")
    
    if args.paths is not None and args.master_path is not None:
        raise Exception("The arguments \"-paths\" or \"-master_path\" cannot both be supplied!")
   
    # Extract paths inside directory if master path is supplied
    if args.master_path is not None:
        args.paths = get_subdirs_in_dir(args.master_path)

    # Perform plotting
    plot_trainings(args.paths, title=args.title, legends=args.legends, ylim=args.ylim, summary=bool(args.summary), filename=args.filename)


    # dirs = [
    #     r"D:\Speciale\Repos\cell crop phantom\output\2020-09-19--09-46-04_Training",
    #     r"D:\Speciale\Repos\cell crop phantom\output\2020-09-19--10-28-44_Training",
    #     r"D:\Speciale\Repos\cell crop phantom\output\2020-09-19--11-16-02_Training",
    #     r"D:\Speciale\Repos\cell crop phantom\output\2020-09-19--12-03-29_Training",
    #     r"D:\Speciale\Repos\cell crop phantom\output\2020-09-19--13-08-35_Training"
    # ]

    # #dirs = get_subdirs_in_dir(r"D:\Speciale\Repos\cell crop phantom\output\EXP Noise level")
    # dirs = get_subdirs_in_dir(r"D:\Speciale\Repos\cell crop phantom\output\EXP ResNet architecture")
