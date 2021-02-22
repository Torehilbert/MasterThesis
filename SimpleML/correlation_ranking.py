import numpy as np
import argparse
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import wfutils.mmap
import wfutils.infofile
import wfutils.log

parser = argparse.ArgumentParser()
parser.add_argument('-path_data', required=False, type=str, default=r"E:\Phantom_v3\train\images_DPI")


def correlation_measure(image):
    n_channels = image.shape[2]
    image = np.reshape(image, newshape=(image.shape[0]*image.shape[1], n_channels))
    image = image[image[:,0] != 0]

    mus = np.mean(image, axis=0)
    stds = np.std(image, axis=0)

    correlation_matrix = np.zeros(shape=(n_channels, n_channels))
    for c1 in range(n_channels):
        for c2 in range(n_channels):
            numerator = np.mean((image[:,c1] - mus[c1])*(image[:,c2] - mus[c2]))
            denominator = stds[c1] * stds[c2]
            if numerator == 0:
                correlation_matrix[c1,c2] = 0
            else: 
                correlation_matrix[c1,c2] = numerator/denominator

    return correlation_matrix


def channel_selection(correlation_matrix, method='max'):
    use_max = True if method=='max' else False

    correlation_matrix = np.array(correlation_matrix, copy=True)
    np.fill_diagonal(correlation_matrix, 0)
    correlation_matrix = np.abs(correlation_matrix)

    ranking = []
    channel_pool = list(range(correlation_matrix.shape[0]))

    temporary_rank = np.argsort(np.max(correlation_matrix, axis=0) if use_max else np.mean(correlation_matrix, axis=0))
    ranking.append(temporary_rank[0])
    channel_pool.remove(temporary_rank[0])

    while len(channel_pool) > 0:
        scores = []
        for ch in channel_pool:
            vals = []
            for ch2 in ranking:
                vals.append(correlation_matrix[ch, ch2])
            scores.append(max(vals) if use_max else sum(vals)/len(vals))
        chosen = channel_pool[np.argsort(scores)[0]]
        ranking.append(chosen)
        channel_pool.remove(chosen)

    return ranking


if __name__ == "__main__":
    args = parser.parse_args()
    path_output_folder = wfutils.log.create_output_folder("SimpleML Ranking Correlation")
    wfutils.log.log_arguments(path_output_folder, args)

    # Load data
    channels = wfutils.infofile.read_channel_order(wfutils.infofile.get_path_to_channel_order(args.path_data))
    shapes, data_dtype = wfutils.infofile.read_info(wfutils.infofile.get_path_to_infofile(args.path_data))
    mmaps = wfutils.mmap.get_class_mmaps_read(args.path_data, shapes, data_dtype)

    # Compute correlations for all classes
    class_correlations = []
    for clas in range(3):
        print("Class %d" % (clas+1))
        correlations = []
        print_progress_interval = shapes[clas][0]//100
        for r in range(shapes[clas][0]):
            if(r % print_progress_interval==0):
                print("  %d%%" % (100*r/shapes[clas][0]))
            correlations.append(correlation_measure(mmaps[clas][r]))
        class_correlations.append(np.mean(np.stack(correlations), axis=0))
    
    class_correlations =  np.mean(np.stack(class_correlations), axis=0)
    ranking_sum = channel_selection(class_correlations, method='sum')
    ranking_max = channel_selection(class_correlations, method='max')

    # Save correlations
    f = open(os.path.join(path_output_folder, 'correlations.csv'), 'w')
    f.write(" ,")
    for c in range(len(channels)):
        f.write(channels[c])
        if c != len(channels)-1:
            f.write(",")

    f.write("\n")
    for c1 in range(class_correlations.shape[0]):
        f.write(channels[c1] + ",")
        for c2 in range(class_correlations.shape[1]):
            f.write("%f" % class_correlations[c1,c2])
            if c2 != class_correlations.shape[1] - 1:
                f.write(",")
        f.write("\n")
    f.close()

    # Save ranking
    f = open(os.path.join(path_output_folder, 'ranking.csv'), 'w')
    fn = open(os.path.join(path_output_folder, 'ranking_names.csv'), 'w')
    f.write("sum method, max mathod\n")
    fn.write("sum method, max mathod\n")
    for i in range(len(ranking_sum)):
        f.write("%d,%d\n" % (ranking_sum[i], ranking_max[i]))
        fn.write("%s,%s\n" % (channels[ranking_sum[i]], channels[ranking_max[i]]))
    f.close()
    fn.close()
    #np.savetxt(os.path.join(path_output_folder, 'ranking.csv'), ranking, delimiter=",", fmt='%d')

    # Save ranking with names
    # f = open(os.path.join(path_output_folder, 'ranking_names.csv'), 'w')
    # for i in range(len(ranking)):
    #     f.write(channels[ranking[i]] + "\n")
    # f.close()    