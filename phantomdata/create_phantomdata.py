import numpy as np
import os
import sys
import random
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import wfutils.infofile
import wfutils.idxfunc
import wfutils.mmap
import wfutils.log
import degrade
import interaction_feature


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-path_src", required=False, type=str, default=r"E:\full32_redist")
    parser.add_argument("-path_dst", required=False, type=str, default=r"E:\Phantom")
    parser.add_argument("-src_channel", required=False, type=int, default=-1)
    parser.add_argument("-samples", required=False, type=int, default=10)
    parser.add_argument("-degrade_discretize_coefficients", required=False, type=float, nargs="+", default=[1.0, 0.5, 0.35])
    parser.add_argument("-degrade_blur_kernel_sizes", required=False, type=int, nargs="+", default=[6,8,16])
    parser.add_argument("-degrade_noise_levels", required=False, type=float, nargs="+", default=[2,4,8])
    parser.add_argument("-add_interactive_class_features", required=False, type=int, default=1)
    args = parser.parse_args()
    print("Creating phantom dataset with %d samples of each class out of channel %d in source data:\n  %s\nwith output in:\n  %s" % (args.samples, args.src_channel, args.path_src, args.path_dst))

    n_channels_dst = 1 + len(args.degrade_discretize_coefficients) + len(args.degrade_blur_kernel_sizes) + len(args.degrade_noise_levels) + (2 if bool(args.add_interactive_class_features) else 0)
    print("Output images have a total of %d channels" % n_channels_dst)

    # Load info file
    shapes, data_dtype = wfutils.infofile.read_info(os.path.join(args.path_src, 'info.txt'))

    # Error checking
    min_count = min([s[0] for s in shapes])
    if args.samples > min_count:
        raise Exception("Cannot extract %d samples from a root file with only %d samples available!" % (args.samples, min_count))
    
    # Open source mmaps
    mmaps_src = wfutils.mmap.get_class_mmaps_read(args.path_src, shapes=shapes, dtype=data_dtype)
    print("Opened source memory maps!")

    # Shuffle and extract indices for source maps
    idx_lists = [wfutils.idxfunc.random_sample_indices(shapes[i][0], args.samples) for i in range(3)]
    channel_order_src = wfutils.infofile.read_channel_order(os.path.join(args.path_src, 'channel_order.txt'))
    if args.src_channel != -1:
        channels = [args.src_channel]
    else:
        channels = np.linspace(0, shapes[0][3], shapes[0][3], endpoint=False, dtype=np.uint64)
    print("Obtained shuffled indices!")

    os.makedirs(args.path_dst)
    wfutils.log.log_arguments(args.path_dst, args)

    for channel in channels:
        print("Channel %s..." % channel_order_src[channel])
        path_channel_output = os.path.join(args.path_dst, channel_order_src[channel])
        os.makedirs(path_channel_output)

        # Open destination mmaps 
        shapes_dst = [(args.samples, shapes[0][1], shapes[0][2], n_channels_dst) for i in range(3)]
        mmaps_dst = wfutils.mmap.get_class_mmaps_write(path_channel_output, shapes=shapes_dst, dtype=data_dtype)
        print("Create destination memory maps!")

        # Transfer data
        for c in range(3):
            print("Begin transfering and creating data for class %d:" % (c+1))
            idxlist = idx_lists[c]
            mmapsrc = mmaps_src[c]
            mmapdst = mmaps_dst[c]

            if not bool(args.add_interactive_class_features):
                for i in range(args.samples):
                    mmapdst[i] = degrade.append_degraded_channels(
                        image=mmapsrc[idxlist[i],:,:,channel], 
                        degrade_discretize=args.degrade_discretize_coefficients,
                        degrade_blur=args.degrade_blur_kernel_sizes,
                        degrade_noise=args.degrade_noise_levels)
            else:
                X = interaction_feature.get_features(c * np.ones(shape=(len(idxlist),), dtype=int), number_of_classes=3, multiplier=6)
                for i in range(args.samples):
                    mmapdst[i,:,:,:-2] = degrade.append_degraded_channels(
                        image=mmapsrc[idxlist[i],:,:,channel], 
                        degrade_discretize=args.degrade_discretize_coefficients,
                        degrade_blur=args.degrade_blur_kernel_sizes,
                        degrade_noise=args.degrade_noise_levels)
                    mmapdst[i,:,:,-2:] = interaction_feature.create_channels_images(mmapsrc[idxlist[i],:,:,channel], X[i,0], X[i,1])

        # Create info file
        wfutils.infofile.create_info(os.path.join(path_channel_output, 'info.txt'), shapes=shapes_dst, data_dtype=data_dtype)
        
        # Create channel order file
        src_channel_name = channel_order_src[channel]
        channel_labels = [src_channel_name]

        for disc in args.degrade_discretize_coefficients:
            channel_labels.append("Discretize %.2f" % disc)
        
        for bsize in args.degrade_blur_kernel_sizes:
            channel_labels.append("Blurred %d" % bsize)

        for nlevel in args.degrade_noise_levels:
            channel_labels.append("Noisy %.2f" % nlevel)

        if bool(args.add_interactive_class_features):
            channel_labels.append("X1")
            channel_labels.append("X2")
            
        wfutils.infofile.create_channel_order(
            path_order_file=os.path.join(path_channel_output, 'channel_order.txt'), 
            labels=channel_labels)


    print("Done!")
