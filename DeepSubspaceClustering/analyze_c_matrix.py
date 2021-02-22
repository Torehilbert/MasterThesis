import numpy as np
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument("-path_matrix", required=False, type=str, default=r"D:\Speciale\Code\output\2020-11-06--10-46-08_DSCNetTraining\Matrices\cm_7.npy")


if __name__ == "__main__":
    args = parser.parse_args()

    data = np.load(args.path_matrix)

    # Weight sums
    row_sums = np.sum(data, axis=1)
    print("Row sums of matrix are: ", row_sums)



    time.sleep(10)