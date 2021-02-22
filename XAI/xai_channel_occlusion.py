import os
import sys
import tensorflow as tf
import numpy as np
import argparse
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import wfutils.infofile
import wfutils.mmap
import wfutils.log
from plotlib.style import COLORS
import plotlib.channel_visualization
import pandas as pd
import seaborn


def perform_channel_occlusion(model, inp, reference_class, method='average', inp_average=None, noise_magnitude=1):
    if method=='average' and inp_average is None:
        raise Exception("Expecting argument \"inp_average\" to not be None when method is: %s" % method)
    
    if method=='average':
        return _channel_occlusion_average(model, inp, reference_class, average_inp=inp_average)
    elif method=='noise':
        return _channel_occlusion_noise(model, inp, reference_class, noise_magnitude)
    elif method=='zero':
        return _channel_occlusion_zero(model, inp, reference_class)
    else:
        raise Exception("Unrecognized method %s!" % method)


def average_over_inputs(memorymap_class1, memorymap_class2, memorymap_class3, return_total_average=True, weight_average_by_class_n=False):
    if return_total_average:
        if weight_average_by_class_n:
            n1,n2,n3 = memorymap_class1.shape[0], memorymap_class2.shape[0], memorymap_class3.shape[0]
            w1,w2,w3 = n1/(n1+n2+n3), n2/(n1+n2+n3), n3/(n1+n2+n3)
            return w1 * memorymap_class1.mean(axis=0) + w2 * memorymap_class2.mean(axis=0) + w3 * memorymap_class3.mean(axis=0)
        else:
            return (memorymap_class1.mean(axis=0) + memorymap_class2.mean(axis=0) + memorymap_class3.mean(axis=0))/3

    else:
        return memorymap_class1.mean(axis=0), memorymap_class2.mean(axis=0), memorymap_class3.mean(axis=0)


def _channel_occlusion_average(model, inp, reference_class, average_inp):
    if len(inp.shape) != 4:
        raise Exception("Expecting inp to have 4D shape: (batch, row, column, channel)!")

    n_channels = inp.shape[3]

    prob_base = model(inp)[0, reference_class].numpy()
    probs = []
    for ch in range(n_channels):
        inp_ocl = np.array(inp, copy=True)
        inp_ocl[:,:,:,ch] = average_inp[:,:,ch]

        probs.append(model(inp_ocl)[0, reference_class].numpy())
    
    return np.stack(probs) - prob_base, prob_base 


def _channel_occlusion_noise(model, inp, reference_class, noise_magnitude=1):
    if len(inp.shape) != 4:
        raise Exception("Expecting inp to have 4D shape: (batch, row, column, channel)!")

    n_channels = inp.shape[3]

    prob_base = model(inp)[0, reference_class].numpy()
    probs = []
    nmap = noise_magnitude*np.random.randn(inp.shape[0], inp.shape[1], inp.shape[2])
    for ch in range(n_channels):
        inp_ocl = np.array(inp, copy=True)
        inp_ocl[:,:,:,ch] = nmap

        probs.append(model(inp_ocl)[0, reference_class].numpy())
    
    return np.stack(probs) - prob_base, prob_base 


def _channel_occlusion_zero(model, inp, reference_class):
    if len(inp.shape) != 4:
        raise Exception("Expecting inp to have 4D shape: (batch, row, column, channel)!")

    n_channels = inp.shape[3]

    prob_base = model(inp)[0, reference_class].numpy()
    probs = []
    for ch in range(n_channels):
        inp_ocl = np.array(inp, copy=True)
        inp_ocl[:,:,:,ch] = 0

        probs.append(model(inp_ocl)[0, reference_class].numpy())
    
    return np.stack(probs) - prob_base, prob_base


def _rename_labels(labels):
    renamedict = {
        'Discretize 1.00':'Discr1', 
        'Discretize 0.50':'Discr2', 
        'Discretize 0.35':'Discr3', 
        'Blurred 6':'Blurr1', 
        'Blurred 8':'Blurr2', 
        'Blurred 16':'Blurr3',
        'Noisy 4.00':'Noisy1',
        'Noisy 8.00':'Noisy2',
        'Noisy 16.00':'Noisy3'
    }
    new_labels = []
    for lab in labels:
        if lab in renamedict.keys():
            new_labels.append(renamedict[lab])
        else:
            new_labels.append(lab)
    return new_labels


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Arguments
    parser = argparse.ArgumentParser()

    #parser.add_argument("-path_model", required=False, type=str, default=r"D:\Speciale\Code\output\Performance Trainings\NONOISE\2021-01-17--13-48-18_Training_57580\model_end")
    parser.add_argument("-path_model", required=False, type=str, default=r"D:\Speciale\Code\output\Performance Trainings\C0123456789\C0123456789_Run1\model_best")

    parser.add_argument("-evaluation_data", required=False, type=str, default=r"E:\validate32_redist")

    parser.add_argument("-weighting_method", required=False, type=str, default="constant")  # "constant", "linear", "power"
    parser.add_argument("-replacement_method", required=False, type=str, default="average_class_wise")  # "zero", "noise", "average", "average_class_wise"
    parser.add_argument("-n_max", required=False, type=int, default=50)
    parser.add_argument("-y_lim", required=False, type=int, nargs="+", default=None)
    args = parser.parse_args()
    
    # i) Error checkings
    if not os.path.isdir(args.path_model):
        raise Exception("Path %s is not a directory!" % args.path_model)
    
    if not os.path.isdir(args.evaluation_data):
        raise Exception("Path %s is not a directory!" % args.evaluation_data)

    acceptable_weighting_methods = ["constant", "linear", "power"]
    if args.weighting_method not in acceptable_weighting_methods:
        raise Exception("Unknown weighting method: %s!" % args.weighting_method)

    accpetable_replacement_methods = ["average_class_wise", "average", "zero", "noise"]
    if args.replacement_method not in accpetable_replacement_methods:
        raise Exception("Unknown replacement method: %s!" % args.replacement_method)

    if args.n_max <= 0:
        raise Exception("Number of samples must be greater than zero!")

    # ii) Create output folder
    path_output_folder = wfutils.log.create_output_folder("XAI_ChannelOcclusion")
    wfutils.log.log_arguments(path_output_folder, args)

    # A) Load model
    model = tf.keras.models.load_model(args.path_model)

    # B) Load inference data
    path_info_file = wfutils.infofile.get_path_to_infofile(args.evaluation_data)
    shapes, data_dtype = wfutils.infofile.read_info(path_info_file)
    channel_order = wfutils.infofile.read_channel_order(wfutils.infofile.get_path_to_channel_order(args.evaluation_data), exclude_prefix=True)

    mmaps = wfutils.mmap.get_class_mmaps_read(args.evaluation_data, shapes, data_dtype)
    
    # B2) Compute means
    if args.replacement_method=='average' or args.replacement_method=="average_class_wise":
        avg_inps = average_over_inputs(mmaps[0], mmaps[1], mmaps[2], return_total_average=False)
        average_input = (avg_inps[0] + avg_inps[1] + avg_inps[2])/3
        channel_names = wfutils.infofile.read_channel_order(wfutils.infofile.get_path_to_channel_order(args.evaluation_data), exclude_prefix=True)

        if args.replacement_method=="average_class_wise":
            for i in range(3):
                plotlib.channel_visualization.visualize_channels(
                    image=avg_inps[i], 
                    save_path=os.path.join(path_output_folder, 'average_input_class%d.png' % (i+1)), 
                    show=False,
                    channel_names=channel_names,
                    figsize=(6,3))
        else:
            plotlib.channel_visualization.visualize_channels(
                image=average_input, 
                save_path=os.path.join(path_output_folder, 'average_input.png'), 
                show=False,
                channel_names=channel_names)


    # C) Run inference to gather channel-wise importance
    dprob_stats = []
    for CLASS in range(3):
        print("Class %d" % (CLASS + 1))
        delta_probs_all = []
        n_images = shapes[CLASS][0]
        final_number_of_images = min(n_images, args.n_max)
        print_progress_interval = max(final_number_of_images//10, 1)
        for i in range(n_images):
            if i % print_progress_interval == 0:
                print("%.0f%%" % (100*i/final_number_of_images))

            if args.replacement_method=="average_class_wise":
                delta_probs = None
                prob_base = None
                for CLASS_SECOND in range(3):
                    if CLASS==CLASS_SECOND:
                        continue

                    delta_probs_subpart, prob_base_subpart = perform_channel_occlusion(
                        model=model, 
                        inp=np.expand_dims(mmaps[CLASS][i], axis=0), 
                        reference_class=CLASS, 
                        method='average',  # replaces "average_class_wise" with "average"
                        inp_average=avg_inps[CLASS_SECOND])
                    
                    if delta_probs is not None:
                        delta_probs += delta_probs_subpart
                        prob_base += prob_base_subpart
                    else:
                        delta_probs = delta_probs_subpart
                        prob_base = prob_base_subpart
                
                delta_probs = delta_probs/2
                prob_base = prob_base/2

            else:
                delta_probs, prob_base = perform_channel_occlusion(
                    model=model, 
                    inp=np.expand_dims(mmaps[CLASS][i], axis=0), 
                    reference_class=CLASS, 
                    method=args.replacement_method, 
                    inp_average=average_input if args.replacement_method=='average' else None)

            if args.weighting_method == "constant":
                importance = delta_probs
            elif args.weighting_method == "linear":
                importance = prob_base*delta_probs
            elif args.weighting_method == "power":
                importance = prob_base*prob_base*delta_probs

            delta_probs_all.append(importance)
            if i == args.n_max:
                break
    
        delta_probs_all = 100*np.stack(delta_probs_all)
        delta_probs_mean = np.mean(delta_probs_all, axis=0)  
        delta_probs_se = np.std(delta_probs_all, axis=0) / (np.sqrt(args.n_max))
        dprob_stats.append([delta_probs_mean, delta_probs_se])
    dprob_stats = np.array(dprob_stats)  # shape = [class, stat = mean/se, channel]

    #   append aggregated estimate
    aggregated_mean = np.mean(dprob_stats[:,0,:], axis=0)
    aggregated_se = (1/3) * np.sqrt(np.sum(np.power(dprob_stats[:,1,:], 2), axis=0))
    dprob_stats_aggregated = np.expand_dims(np.stack((aggregated_mean, aggregated_se), axis=0), axis=0)
    dprob_stats = np.concatenate((dprob_stats, dprob_stats_aggregated), axis=0)

    ### SAVE RESULTS
    #   A) Save raw stats
    num_classes = dprob_stats.shape[0]
    with open(os.path.join(path_output_folder, 'raw_stats.csv'), 'w') as f:
        f.write("class,channel,dprob mean,dprob standard error\n")
        for class_id in range(num_classes):
            for channel_id in range(dprob_stats.shape[2]):
                stats = dprob_stats[class_id,:,channel_id]
                f.write("%s,%s,%f,%f\n" % (("%d" % (class_id+1)) if class_id <= 2 else "all", channel_order[channel_id], stats[0], stats[1]))

    #   B) Save ranking analysis
    num_classes = dprob_stats.shape[0]
    num_channels = dprob_stats.shape[2]
    with open(os.path.join(path_output_folder, 'ranking.csv'), 'w') as f:
        header = ",".join(["rank (class %d)" % (i+1) for i in range(num_classes - 1)]) + ",rank (all)\n"
        f.write(header)
        ranks = [np.argsort(dprob_stats[class_id, 0, :]) for class_id in range(num_classes)]
        
        for i in range(num_channels):
            line = ",".join(["%d" % (ranking[i]) for ranking in ranks]) + "\n"
            f.write(line)      

    #   C) Plot
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    #       find y limits
    if args.y_lim is None:
        args.y_lim = [np.min(dprob_stats[:,0,:] - dprob_stats[:,1,:]), np.max(dprob_stats[:,0,:] + dprob_stats[:,1,:])]
    
    fig = plt.figure(figsize=(10,3))
    for i in range(num_classes):
        color = COLORS[i] if i <= 2 else 'black'
        plt.subplot(1,4,i+1)
        plt.hlines(0, 0, shapes[0][3], linestyle='dashed', color='gray', linewidth=0.5)
        plt.errorbar(
            x=[j for j in range(0, len(dprob_stats[i,0,:]))],
            y=dprob_stats[i,0,:],
            yerr=dprob_stats[i,1,:],
            fmt='.', color=color, ms=10)
        plt.ylim(args.y_lim)
        if i==0:
            plt.ylabel("Procentage point change \nin correct class probability")
        if i==1:
            pass #plt.title("Channel Occlusion - %s - %s" % (args.replacement_method, args.weighting_method))
        if i==3:
            # Create legend
            custom_legend_lines = [Line2D([0], [0], color=COLORS[k], lw=1) for k in range(3)] + [Line2D([0], [0], color='black', lw=1)]
            custom_legend_names = ['Healthy', 'Apoptosis', 'Dead', 'Aggregated']
            plt.legend(custom_legend_lines, custom_legend_names)
        plt.xticks(ticks=list(range(0, len(channel_order))), labels=_rename_labels(channel_order), rotation=90)

    #plt.suptitle("Channel occlusion by %s" % args.replacement_method)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(path_output_folder, 'plot.png'), dpi=500)
    plt.show()