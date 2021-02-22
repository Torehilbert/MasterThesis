import os
import numpy as np


DIRECTORY = r"E:\original_test_data"
OUTPUT_DIRECTORY = r"E:\test32"
LABEL_FOLDER_ID = 'labels'
CLASSES = [1,2,3]
DTYPE = "float32"
IMAGE_DIMENSION = 64
NUM_CHANNELS = 10

EXPECTED_FOLDERS = ["images_Aperture", "images_ApodizedAP", "images_BrightField", "images_DarkField", "images_DFIOpen", "images_DFIPhase", "images_DPI", "images_iSSC", "images_Phase", "images_UVPhase"]

if __name__ == "__main__":
    folders = os.listdir(DIRECTORY)
    
    for exp_fold in EXPECTED_FOLDERS:
        if exp_fold not in folders:
            raise Exception("Could not find %s in directory" % exp_fold)

    if LABEL_FOLDER_ID not in folders:
        raise Exception("A \"%s\" folder is required in directory: %s" % (LABEL_FOLDER_ID, DIRECTORY))
    
    if len(folders) != 11:
        raise Exception("Too many folders in directory!")
    
    folders = EXPECTED_FOLDERS

    path_labels_folder = os.path.join(DIRECTORY, LABEL_FOLDER_ID)
    print("Found %d images modes and labels folder!" % len(folders))
    
    # Found out how many samples from each class exist (takes quite sometime...)
    print("Counting class count...")
    sample_names = os.listdir(path_labels_folder)
    n_images = len(sample_names)
    labels = np.zeros(shape=(n_images,), dtype=np.uint64)
    prog_int = max(n_images // 20, 1)
    for i, name in enumerate(sample_names):
        path_sample = os.path.join(path_labels_folder, name)
        labels[i] = np.load(path_sample)[0]
        if i % prog_int == 0:
            print("  %.0f %%" % (100 * i/n_images))
    
    unqs, n_within_classes = np.unique(labels, return_counts=True)
    print("Counted numbers within classes:", n_within_classes)


    # Constructing mmaps
    print("Constructing memory-maps...")
    os.makedirs(OUTPUT_DIRECTORY)
    mmaps = []
    print("  0 %")
    for i in range(len(CLASSES)):
        memmap = np.memmap(os.path.join(OUTPUT_DIRECTORY, str(CLASSES[i]) + ".npy"), dtype=DTYPE, mode='write', shape=(n_within_classes[i], IMAGE_DIMENSION, IMAGE_DIMENSION, NUM_CHANNELS))
        mmaps.append(memmap)
        print("  %.0f %%" % (100*(i+1)/3))

    # Create info file with shape information
    with open(os.path.join(OUTPUT_DIRECTORY, "info.txt"), 'w') as infofile:
        for i in range(len(CLASSES)):
            infofile.write("%d,%d,%d,%d\n" % (n_within_classes[i], IMAGE_DIMENSION, IMAGE_DIMENSION, NUM_CHANNELS))
        infofile.write("%s" % mmaps[0].dtype)
    
    # Open keyfile
    keyfile = open(os.path.join(OUTPUT_DIRECTORY, "keys.csv"), 'w')
    keyfile.write("image,class,memory_map_entry\n")

    # Transfer and merge data
    print("Transfer data...")
    cursors = 3*[0]
    for i, name in enumerate(sample_names):
        class_idx = int(labels[i] - 1)
        for j, folder in enumerate(folders):
            path_sample = os.path.join(DIRECTORY, folder, name)
            mmaps[class_idx][cursors[class_idx],:,:,j] = np.load(path_sample)
        
        keyfile.write("%s,%d,%d\n" % (name, CLASSES[class_idx], cursors[class_idx]))
        cursors[class_idx] += 1

        if i % prog_int == 0:
            print("  %.0f%%" % (100 * i/n_images))

    # Close keyfile
    keyfile.close()

    # Logs
    path_mode_order_log = os.path.join(OUTPUT_DIRECTORY, 'channel_order.txt')
    with open(path_mode_order_log, 'w') as f:
        for mode in folders:
            f.write(mode + "\n")

    print("Done!")