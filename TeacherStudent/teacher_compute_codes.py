import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
import numpy as np
import argparse
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import wfutils.log
import wfutils.infofile
import wfutils.mmap
import wfutils.idxfunc

parser = argparse.ArgumentParser()
parser.add_argument('-path_data', required=False, type=str, default=r"E:\full32_redist")
parser.add_argument('-path_teacher_encoder', required=False, type=str, default=r"D:\Speciale\Code\output\TeacherStudent\Real Teacher 16\Teacher C\encoder_end")
parser.add_argument('-n_max', required=False, type=int, default=4000)
parser.add_argument('-code_dim', required=False, type=int, default=16)
args = parser.parse_args()


if __name__ == "__main__":
    teacher_encoder = tf.keras.models.load_model(args.path_teacher_encoder)

    #### CONVERT TRAINING DATA TO CODES
    #   read training data
    shapes, data_dtype = wfutils.infofile.read_info(wfutils.infofile.get_path_to_infofile(args.path_data))
    mmaps = wfutils.mmap.get_class_mmaps_read(args.path_data, shapes, data_dtype)
    
    #   select subset indices
    idx_lists = []
    for i in range(3):
        idx_lists.append(wfutils.idxfunc.random_sample_indices(shapes[i][0], args.n_max))

    #   create output folder and log arguments
    path_output_folder = wfutils.log.create_output_folder("TeacherCodes")
    wfutils.log.log_arguments(path_output_folder, args)
    
    #   create output map
    shapes = [(3*args.n_max, shapes[0][1], shapes[0][2], shapes[0][3]), (3*args.n_max, args.code_dim)]
    data_dtype_output = "float32"
    mmap_train = np.memmap(os.path.join(path_output_folder, 'X.npy'), shape=shapes[0], dtype=data_dtype_output, mode='w+')
    mmap_codes = np.memmap(os.path.join(path_output_folder, 'Y.npy'), shape=shapes[1], dtype=data_dtype_output, mode='w+')
    wfutils.infofile.create_info_codes(wfutils.infofile.get_path_to_infofile(path_output_folder), shapes=shapes, data_dtype=data_dtype_output)

    #   convert training to codes
    for CLASS in range(3):
        idx_list = idx_lists[CLASS]
        for i, idx in enumerate(idx_list):
            image = np.expand_dims(mmaps[CLASS][idx], axis=0)
            mmap_train[i + args.n_max * CLASS] = image
            codes = teacher_encoder(image)
            mmap_codes[i + args.n_max * CLASS] = codes
    
    #### NORMALIZE CODES
    vmin = mmap_codes.min(axis=0)
    vmax = mmap_codes.max(axis=0)
    mmap_codes[:] = (mmap_codes[:] - vmin)/(vmax - vmin)
    mmap_codes.flush()

    #### VERIFY NORMALIZATION
    mmap_verify = np.memmap(os.path.join(path_output_folder, 'Y.npy'), shape=(3*args.n_max, args.code_dim), dtype=data_dtype_output, mode='r')
    vmin_verify = mmap_verify.min(axis=0)
    vmax_verify = mmap_verify.max(axis=0)

    print("Are there minimum values above 0.0: ", (vmin_verify > 0.01).any())
    print("Are there maximum values above 1.0: ", (vmax_verify > 1.01).any())
    print("Are there maximum values below 1.0: ", (vmax_verify < 0.99).any())

    
    


    
    