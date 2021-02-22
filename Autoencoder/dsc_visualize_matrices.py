import argparse
import numpy as np
import matplotlib.pyplot as plt
import os


def load_matrix(path):
    return np.loadtxt(path, delimiter=",")


parser = argparse.ArgumentParser()
# parser.add_argument("-path_matrices", required=False, type=str, nargs="+", default=[
#     r"D:\Speciale\Code\output\DSC_New\2020-12-04--20-53-36_DSC_3264", 
#     r"D:\Speciale\Code\output\DSC_New\2020-12-04--23-00-57_DSC_3264", 
#     r"D:\Speciale\Code\output\DSC_New\2020-12-05--02-26-12_DSC_3264",
#     r"D:\Speciale\Code\output\DSC_New\2020-12-05--11-31-20_DSC_3264",
#     r"D:\Speciale\Code\output\DSC_New\2020-12-05--11-32-05_DSC_3264"])
parser.add_argument("-path_matrices", required=False, type=str, nargs="+", default=[
    r"D:\Speciale\Code\output\DSC_New\2020-12-05--11-37-21_DSC_3264_bs32", 
    r"D:\Speciale\Code\output\DSC_New\2020-12-05--11-37-21_DSC_3264_bs32", 
    r"D:\Speciale\Code\output\DSC_New\2020-12-06--14-17-32_DSC_3264_bs32",
    r"D:\Speciale\Code\output\DSC_New\2020-12-06--16-17-49_DSC_3264_bs64",
    r"D:\Speciale\Code\output\DSC_New\2020-12-06--18-10-56_DSC_3264_bs64",
    r"D:\Speciale\Code\output\DSC_New\2020-12-07--05-53-35_DSC_3264_bs64"])
parser.add_argument("-filename", required=False, type=str, default='D.csv')


if __name__ == "__main__":
    args = parser.parse_args()

    plt.figure()

    for i,path_matrix in enumerate(args.path_matrices):
        plt.subplot(len(args.path_matrices), 1, i+1)
        C = load_matrix(os.path.join(path_matrix, args.filename))
        plt.imshow(np.abs(C))
    
    plt.show()
