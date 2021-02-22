import tensorflow as tf
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("-root_folder", required=False, type=str, default=r"D:\Speciale\Code\output\DSC_New\Runs_LONG")
parser.add_argument("-model_name", required=False, type=str, default="model_best")
parser.add_argument("-overwrite", required=False, type=int, default=1)


OUTPUT_FOLDER_NAME = 'matrices'


def load_c_matrix(path_model):
    model = tf.keras.models.load_model(path_model)
    return model.get_layer('selfexp').get_weights()[0]

def calculate_z(C, normalize):
    Cabs = np.abs(C)
    if normalize:
        Cabs = Cabs / np.sum(Cabs, axis=1, keepdims=True)
    Z = Cabs + np.transpose(Cabs)
    return Z

def calculate_d(Z, method='subtract'):
    if method=='division':
        D = 1/Z
    elif method=='subtract':
        D = np.max(Z) - Z
        
    np.fill_diagonal(D, 0)
    return D

def save_matrix(matrix, path):
    np.savetxt(fname=path, X=matrix, delimiter=",")


if __name__ == "__main__":
    args = parser.parse_args()
    conts = os.listdir(args.root_folder)

    for i,folder in enumerate(conts):
        print("%d/%d : %s" % (i+1, len(conts), folder))
        path_folder = os.path.join(args.root_folder, folder)     
        path_model = os.path.join(path_folder, args.model_name)
        if not os.path.isdir(path_model):
            continue
        
        # output folder
        path_output_folder = os.path.join(path_folder, OUTPUT_FOLDER_NAME)
        if not os.path.isdir(path_output_folder):
            os.makedirs(path_output_folder)
        else:
            if args.overwrite==0:
                print("Warning: matrices folder already exist - while overwrite is DISABLED!")
                continue

        # extract matrices      
        C = load_c_matrix(path_model)
        save_matrix(C, os.path.join(path_output_folder, 'C.csv'))
        for row_normalize, label_normalize in zip([False, True], ['', '_norm']):
            Z = calculate_z(C, normalize=row_normalize)
            save_matrix(Z, os.path.join(path_output_folder, 'Z%s.csv' % (label_normalize)))

            for d_method in ['subtract', 'division']:
                D = calculate_d(Z, method=d_method)
                save_matrix(D, os.path.join(path_output_folder, 'D%s%s.csv' % (label_normalize, "_"+d_method)))


