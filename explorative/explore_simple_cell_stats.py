import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import argparse
import random
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plotlib.scatter import scatter
import wfutils


parser = argparse.ArgumentParser()
parser.add_argument('-src', required=False, type=str, default=r"D:\Speciale\Data\Real\Validation (split-by-class)")
parser.add_argument('-samples', required=False, type=int, default=2800)
parser.add_argument('-seed', required=False, type=int, default=-1)


SCRIPT_IDENTIFIER = "ExploreSimpleStats"


if __name__ == "__main__":
    # assumes that images are sorted into class-specific folders under args.src path.
    args = parser.parse_args()
    
    # create outputfolder
    path_output_folder = wfutils.create_output_folder(SCRIPT_IDENTIFIER)
    wfutils.log_arguments(path_output_folder, args)


    modes = os.listdir(args.src)
    for mode in modes:
        path_mode = os.path.join(args.src, mode)

        # Error checks
        if not os.path.isdir(path_mode):
            raise Exception("Supplied source is not a directory: %s" % path_mode)
        
        # Find number of classes
        path_class_folders = [os.path.join(path_mode, s) for s in os.listdir(path_mode)]
        n_classes = len(path_class_folders)

        if n_classes <= 1:
            print("WARNING: Found %d classes" % n_classes)
        
        # Allocate array
        example_im = np.load(os.path.join(path_class_folders[0], os.listdir(path_class_folders[0])[0]))
        all_images = np.empty(shape=(n_classes, args.samples, example_im.shape[0], example_im.shape[1]), dtype=float)

        # Set seed for random sample
        if args.seed != -1:
            random.seed(args.seed)

        # Load images
        for class_idx in range(n_classes):
            print("Class %d" % class_idx)
            image_names = os.listdir(path_class_folders[class_idx])
            print(" sampling %d image names for class %s..." % (args.samples, class_idx))
            chosen_image_names = random.sample(image_names, args.samples)
            print(" loading images for class %s..." % class_idx)
            for i in range(len(chosen_image_names)):
                all_images[class_idx, i, :, :] = np.load(os.path.join(path_class_folders[class_idx], chosen_image_names[i]))


        # Calculate stats
        for class_idx in range(n_classes):
            # only use cell pixels (no background)
            arr_masked = np.ma.masked_array(all_images[class_idx], mask=all_images[class_idx] == 0)

            # compute image-wise mean and stds
            mus = arr_masked.mean(axis=(1,2))
            sds = arr_masked.std(axis=(1,2))

            # plot
            scatter(x=mus, y=sds, 
                    color_index=class_idx, 
                    show=False,
                    title=mode.split('_')[1], 
                    xlabel="Mean intensity", 
                    ylabel="Standard deviation of intensity", 
                    append=(not wfutils.true_first(class_idx)),
                    legends=wfutils.value_last(class_idx, n_classes, [str(0), str(1), str(2)]),
                    save_path=os.path.join(path_output_folder, mode + ".png"))
