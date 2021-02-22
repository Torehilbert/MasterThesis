import os
import sys
import shutil


PATH_DATA = r"E:\original"
LABEL_FOLDER_ID = 'labels'
IGNORE_STRING = 'aug'
EXPECTED_MODE_COUNT = 10

if __name__ == "__main__":
    # check for "labels" directory¨
    path_label_folder = os.path.join(PATH_DATA, LABEL_FOLDER_ID)
    if not os.path.isdir(path_label_folder):
        raise Exception("Could not find \"%s\" folder inside data directory: %s" % (LABEL_FOLDER_ID, PATH_DATA))

    # extract mode folder paths
    modes = os.listdir(PATH_DATA)
    flag_discarded_label_folder = False
    for i, mode in enumerate(modes):
        if mode == LABEL_FOLDER_ID:
            del modes[i]
            flag_discarded_label_folder = True
            break
    
    if not flag_discarded_label_folder:
        raise Exception("Did not remove a \"%s\" folder from modes" % LABEL_FOLDER_ID)

    if len(modes)!=EXPECTED_MODE_COUNT:
        raise Exception("Expected number of mode folders to be %d, it is %d." % (EXPECTED_MODE_COUNT, len(modes)))

    path_mode_folders = [os.path.join(PATH_DATA, mode) for mode in modes]
    
    # delete augment files
    fnames = os.listdir(path_label_folder)
    N = len(fnames)
    N_kept = 0
    N_removed = 0
    prog_int = max(N // 100, 1)  # 0%, 10%, 20%...
    for i, name in enumerate(fnames):
        identifier = name.split(".")[0]  # removing file extension .npy from name
        if IGNORE_STRING in identifier:
            N_removed += 1

            for j in range(EXPECTED_MODE_COUNT):
                path_file_src = os.path.join(path_mode_folders[j], name)
                if os.path.isfile(path_file_src):
                    os.remove(path_file_src)
                else:
                    print("Warning: Could not find file %s" % path_file_src)
            
            path_label_file_src = os.path.join(path_label_folder, name)
            if os.path.isfile(path_label_file_src):
                os.remove(path_label_file_src)
            else:
                print("Warning: Could not find file %s" % path_label_file_src)

        else:
            N_kept += 1


        if i % prog_int == 0:
            print("%.0f%%" % (100*i/N))

    print("Done removing %d files (x%d) while keeping %d (x%d)." % (N_removed, EXPECTED_MODE_COUNT + 1, N_kept, EXPECTED_MODE_COUNT + 1))

    