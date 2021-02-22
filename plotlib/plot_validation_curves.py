import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
import argparse


def recursive_find(paths_root, target_filename='series.txt'):
    all_paths = []
    for root_path in paths_root:
        path_file = os.path.join(root_path, target_filename)
        if os.path.isfile(path_file):
            all_paths.append(root_path)
            continue
        else:
            subcontent = os.listdir(root_path)
            for cont in subcontent:
                path_sub = os.path.join(root_path, cont)
                if os.path.isdir(path_sub):
                    all_paths.extend(recursive_find([os.path.join(root_path, cont)]))
    return all_paths


parser = argparse.ArgumentParser()
parser.add_argument("-g1", required=False, type=str, nargs="+", default=[r"D:\Speciale\Code\output\DSC_New\Runs_3264_relu"])
parser.add_argument("-g2", required=False, type=str, nargs="+", default=None)
parser.add_argument("-g3", required=False, type=str, nargs="+", default=None)
parser.add_argument("-g4", required=False, type=str, nargs="+", default=None)
parser.add_argument("-g5", required=False, type=str, nargs="+", default=None)
parser.add_argument("-filename", required=False, type=str, default="series.txt")
parser.add_argument("-ymetric", required=False, type=str, default="val_loss")
parser.add_argument("-xmetric", required=False, type=str, default="epoch")
parser.add_argument("-colors", required=False, type=str, nargs="+", default=["steelblue", "salmon", "orchid", "green"])
parser.add_argument("-gnames", required=False, type=str, nargs="+", default=None)
parser.add_argument("-summarize", required=False, type=int, default=0)
parser.add_argument("-highlight_index", required=False, type=int, default=None)
parser.add_argument("-ylim", required=False, type=float, nargs="+", default=None)
parser.add_argument("-ylabel", required=False, type=str, default=None)
parser.add_argument("-xlabel", required=False, type=str, default=None)



if __name__ == "__main__":
    args = parser.parse_args()

    paths_groups = []
    if args.g1 is not None:
        args.g1 = recursive_find(args.g1, target_filename=args.filename)
        paths_groups.append(args.g1)
    if args.g2 is not None:
        args.g2 = recursive_find(args.g2, target_filename=args.filename)
        paths_groups.append(args.g2)
    if args.g3 is not None:
        args.g3 = recursive_find(args.g3, target_filename=args.filename)
        paths_groups.append(args.g3)
    if args.g4 is not None:
        args.g4 = recursive_find(args.g4, target_filename=args.filename)
        paths_groups.append(args.g4)
    if args.g5 is not None:
        args.g5 = recursive_find(args.g5, target_filename=args.filename)
        paths_groups.append(args.g5)


    plt.figure(figsize=(4.5, 3.5))

    if args.summarize==0:
        for i,group in enumerate(paths_groups):
            for j,path in enumerate(group):
                df = pd.read_csv(os.path.join(path, args.filename))
                Y = df.loc[:, args.ymetric]
                X = df.loc[:, args.xmetric]
                gname = "%d" % i if args.gnames is None else args.gnames[i]
                plt.plot(X,Y, color=args.colors[i], label="%s (%d)" % (gname,j))
        plt.legend()
        plt.xlabel(args.xmetric if args.xlabel is None else args.xlabel)
        plt.ylabel(args.ymetric if args.ylabel is None else args.ylabel)
    else:
        for i, group in enumerate(paths_groups):
            Y_all = np.zeros(shape=(500, len(group)))
            Y_all[:,:] = np.nan
            X_all = np.arange(0, 500)
            for j,path in enumerate(group):
                df = pd.read_csv(os.path.join(path, args.filename))
                Y = df.loc[:,args.ymetric]
                X = df.loc[:,args.xmetric]
                print(len(Y))
                Y_all[:len(Y), j] = Y
            
            Y_mean = np.nanmean(Y_all, axis=1)
            Y_min = np.nanmin(Y_all, axis=1)
            Y_max = np.nanmax(Y_all, axis=1)
            plt.fill_between(X_all, Y_min, Y_max, color=args.colors[i], alpha=0.25)
            plt.plot(X_all, Y_mean, color=args.colors[i])
            if args.highlight_index is not None:
                plt.plot(X_all, Y_all[:,args.highlight_index], color='k')
        plt.xlabel('Epoch' if args.xmetric=='epoch' else args.xmetric)
        plt.ylabel('Validation loss' if args.ymetric=='val_loss' else args.ymetric)    

        legend_labels = ['Mean over all runs', 'Min-Max over all runs']
        if args.highlight_index is not None:
            legend_labels.insert(1, 'Selected run')
        plt.legend(legend_labels)
    
    if args.ylim is not None:
        plt.ylim(args.ylim)
    plt.tight_layout()
    plt.savefig("dsc_validation_loss.png", dpi=250)
    plt.show()