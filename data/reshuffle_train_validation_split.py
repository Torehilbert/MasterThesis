import numpy as np
import os
import math
import random
import shutil

PATH_TRAINING_OLD = r"E:\full32"
PATH_VALIDATION_OLD = r"E:\validate32"

PATH_TRAINING_NEW = r"E:\full32_redist"
PATH_VALIDATION_NEW = r"E:\validate32_redist"

VALIDATION_FRACTION = 0.1
TRAINING_FRACTION = 1.0 - VALIDATION_FRACTION  # 7/8 

DATA_DTYPE = "float32"


if __name__ == "__main__":
    # Paths olds
    path_info_train_old = os.path.join(PATH_TRAINING_OLD, 'info.txt')
    path_keys_train_old = os.path.join(PATH_TRAINING_OLD, 'keys.csv')
    paths_class_train_old = [os.path.join(PATH_TRAINING_OLD, '%d.npy') % (i+1) for i in range(0,3)]

    path_info_val_old = os.path.join(PATH_VALIDATION_OLD, 'info.txt')
    path_keys_val_old = os.path.join(PATH_VALIDATION_OLD, 'keys.csv')
    paths_class_val_old = [os.path.join(PATH_VALIDATION_OLD, '%d.npy') % (i+1) for i in range(0,3)]

    # Error checks
    for file_path in [path_info_train_old, path_keys_train_old] + paths_class_train_old + [path_info_val_old, path_keys_val_old] + paths_class_val_old:
        if not os.path.isfile(file_path):
            raise Exception("Could not find file: %s" % file_path)
    
    # Find shapes file
    train_old_shapes = []
    train_old_dtype = None
    with open(path_info_train_old, 'r') as f:
        for i in range(3):
            train_old_shapes.append(tuple((int(s) for s in f.readline().split(","))))
        train_old_dtype = f.readline()
    print("In current training files:", train_old_shapes, " of type:", train_old_dtype)

    val_old_shapes = []
    val_old_dtype = None
    with open(path_info_val_old, 'r') as f:
        for i in range(3):
            val_old_shapes.append(tuple((int(s) for s in f.readline().split(","))))
        val_old_dtype = f.readline()  
    print("In current validation files:", val_old_shapes, " of type:", val_old_dtype)


    ns_train_to_train = [math.ceil(TRAINING_FRACTION * train_old_shapes[i][0]) for i in range(3)]
    ns_val_to_train = [math.ceil(TRAINING_FRACTION * val_old_shapes[i][0]) for i in range(3)]
    
    ns_train_to_val = [math.floor(VALIDATION_FRACTION * train_old_shapes[i][0]) for i in range(3)]
    ns_val_to_val = [math.floor(VALIDATION_FRACTION * val_old_shapes[i][0]) for i in range(3)]

    ns_train = [ns_train_to_train[i] + ns_val_to_train[i] for i in range(3)]
    ns_val = [ns_train_to_val[i] + ns_val_to_val[i] for i in range(3)]
    
    idx_train_old = [np.linspace(0, train_old_shapes[i][0], train_old_shapes[i][0], endpoint=False, dtype=int) for i in range(3)]
    idx_val_old = [np.linspace(0, val_old_shapes[i][0], val_old_shapes[i][0], endpoint=False, dtype=int) for i in range(3)]

    # Shuffle
    for i in range(3):
        np.random.shuffle(idx_train_old[i])
        np.random.shuffle(idx_val_old[i]) 

    # Create new folders
    os.makedirs(PATH_TRAINING_NEW)
    os.makedirs(PATH_VALIDATION_NEW)

    cursor_train_old = [0,0,0]
    cursor_val_old = [0,0,0]

    # Determine indices
    train_new_shapes = []
    val_new_shapes = []
    for cell_class in range(3):
        print("Cell class: %d" % (cell_class + 1))

        # Create training dataset
        train_shape = (ns_train[cell_class],64,64,10)
        train_new_shapes.append(train_shape)
        trainset_new = np.memmap(
            filename=os.path.join(PATH_TRAINING_NEW, '%d.npy' % (cell_class+1)),
            shape=train_shape, 
            dtype=DATA_DTYPE, 
            mode='w+')
        print("  created new training file!")

        # Create validation dataset
        val_shape = (ns_val[cell_class],64,64,10)
        val_new_shapes.append(val_shape)
        valset_new = np.memmap(
            filename=os.path.join(PATH_VALIDATION_NEW, '%d.npy' % (cell_class+1)),
            shape=val_shape, 
            dtype=DATA_DTYPE, 
            mode='w+')
        print("  created new validation file!")

        # Open old training and validation datasets
        trainset_old = np.memmap(
            filename=paths_class_train_old[cell_class],
            shape=train_old_shapes[cell_class],
            dtype=train_old_dtype,
            mode='r') 
        valset_old = np.memmap(
            filename=paths_class_val_old[cell_class],
            shape=val_old_shapes[cell_class],
            dtype=val_old_dtype,
            mode='r')
        print("  opened old training and validation files!")

  
        # Transfer data to new training set (part class)
        trainset_new[0:ns_train_to_train[cell_class]] = trainset_old[idx_train_old[cell_class][0:ns_train_to_train[cell_class]]]
        print("  transfered old training part to new training file!")
        trainset_new[ns_train_to_train[cell_class]:] = valset_old[idx_val_old[cell_class][0:ns_val_to_train[cell_class]]]
        print("  transfered old validation part to new training file!")

        # Transfer data to new validation set (part class)
        valset_new[0:ns_train_to_val[cell_class]] = trainset_old[idx_train_old[cell_class][ns_train_to_train[cell_class]:]]
        print("  transfered old training part to new validation file!")
        valset_new[ns_train_to_val[cell_class]:] = valset_old[idx_val_old[cell_class][ns_val_to_train[cell_class]:]]
        print("  transfered old validation part to new validation file!")


    # Create info file with shape information in training folder
    with open(os.path.join(PATH_TRAINING_NEW, "info.txt"), 'w') as info_train:
        for i in range(3):
            info_train.write("%d,%d,%d,%d\n" % (ns_train[i], 64, 64, 10))
        info_train.write("%s" % DATA_DTYPE)
    shutil.copy(
        src=os.path.join(PATH_TRAINING_OLD, 'channel_order.txt'), 
        dst=os.path.join(PATH_TRAINING_NEW, 'channel_order.txt'))

    # Create info file with shape information in validation folder
    with open(os.path.join(PATH_VALIDATION_NEW, "info.txt"), 'w') as info_val:
        for i in range(3):
            info_val.write("%d,%d,%d,%d\n" % (ns_val[i], 64, 64, 10))
        info_val.write("%s" % DATA_DTYPE)
    shutil.copy(
        src=os.path.join(PATH_VALIDATION_OLD, 'channel_order.txt'), 
        dst=os.path.join(PATH_VALIDATION_NEW, 'channel_order.txt'))