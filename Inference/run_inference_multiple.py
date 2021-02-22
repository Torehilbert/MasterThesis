import argparse
import tensorflow as tf
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from inference import add_inference_file_to_training


def recursive_search_for_trainings(path_master, criteria_subdir="model_best"):
    if os.path.isdir(os.path.join(path_master, criteria_subdir)):
        return [path_master]

    trainings = []

    conts = os.listdir(path_master)
    for cont in conts:
        subpath = os.path.join(path_master, cont)
        if os.path.isdir(subpath):
            trainings.extend(recursive_search_for_trainings(subpath))

    return trainings


parser= argparse.ArgumentParser()
parser.add_argument("-data", required=False, type=str, default=r"D:\Speciale\Code\output\Performance Trainings")
#parser.add_argument("-data", required=False, type=str, default=r"D:\Speciale\Code\output\TeacherStudent\Real Teacher 16")
parser.add_argument("-test_data", required=False, type=str, default=r"E:\test32")
parser.add_argument("-net_select", required=False, type=str, default=r"model_best")
parser.add_argument("-out_name", required=False, type=str, default="test_acc.txt")
parser.add_argument("-override", required=False, type=int, default=0)
parser.add_argument("-class_equalize", required=False, type=int, default=0)


if __name__ == "__main__":
    args = parser.parse_args()

    
    trainings = recursive_search_for_trainings(args.data, criteria_subdir=args.net_select)

    for i,training in enumerate(trainings):
        print("(%d/%d) %s" % (i+1, len(trainings), training))
        add_inference_file_to_training(path_training=training, path_test_data=args.test_data, net_selection=args.net_select.split("_")[1], override=(args.override != 0), out_name=args.out_name, class_equalize=(args.class_equalize != 0))
