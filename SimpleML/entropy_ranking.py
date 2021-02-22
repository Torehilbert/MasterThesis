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


# 1. Load class data mmaps
# 2. Compute and aggregate entropies for all images
# 3. Construct ranking
def entropy_measure(image):
    # expecting image of shape (r,r,channels)
    HIST_MIN = -10
    HIST_MAX = 10
    HIST_COUNT = 40

    entropies = np.zeros(shape=(image.shape[2],))

    image = np.reshape(image, (image.shape[0]*image.shape[1], image.shape[2]))
    bins = np.linspace(HIST_MIN, HIST_MAX, HIST_COUNT,endpoint=True)
    for i in range(image.shape[1]):
        hist, _ = np.histogram(image[:,i], bins=bins)
        hist = hist / np.sum(hist)
        hist[hist==0] = 1  # 
        entropies[i] = -np.nansum(hist * np.log(hist))

    return entropies


def entropy_measure_cell_only(image):
    HIST_MIN = -10
    HIST_MAX = 10
    HIST_COUNT = 40
    entropies = np.zeros(shape=(image.shape[2],))

    image = np.reshape(image, (image.shape[0]*image.shape[1], image.shape[2]))
    bins = np.linspace(HIST_MIN, HIST_MAX, HIST_COUNT,endpoint=True)
    for i in range(image.shape[1]):
        pixels = image[image[:,i]!=0,i]
        hist, _ = np.histogram(pixels, bins=bins)
        hist = hist / np.sum(hist)
        hist[hist==0] = 1  # 
        entropies[i] = -np.nansum(hist * np.log(hist))

    return entropies  


def test_reshape(image):
    return np.reshape(image, (image.shape[0]*image.shape[1], image.shape[2]))


if __name__ == "__main__":
    args = parser.parse_args()
    path_output_folder = wfutils.log.create_output_folder("SimpleML Ranking Entropy")
    wfutils.log.log_arguments(path_output_folder, args)

    # Load Data
    channels = wfutils.infofile.read_channel_order(wfutils.infofile.get_path_to_channel_order(args.path_data))
    shapes, data_dtype = wfutils.infofile.read_info(wfutils.infofile.get_path_to_infofile(args.path_data))
    mmaps = wfutils.mmap.get_class_mmaps_read(args.path_data, shapes, data_dtype)

    # Compute entropies for all classes
    class_entropies = []
    for clas in range(3):
        print("Class %d" % (clas+1))
        datamap = mmaps[clas]
        entropies = []
        print_progress_interval = shapes[clas][0]//100
        for r in range(shapes[clas][0]):
            if(r%print_progress_interval==0):
                print("  %d%%" % (100*r/shapes[clas][0]))
            ent = entropy_measure_cell_only(datamap[r])
            entropies.append(ent)

        entropies = np.stack(entropies)
        entropies = np.mean(entropies, axis=0)
        class_entropies.append(entropies)
    
    entropy_scores = np.mean(np.stack(class_entropies), axis=0)
    ranking = np.flip(np.argsort(entropy_scores))

    np.savetxt(os.path.join(path_output_folder, 'entropy_scores.csv'), entropy_scores, delimiter=",")
    np.savetxt(os.path.join(path_output_folder, 'ranking.csv'), ranking, delimiter=",")

    f = open(os.path.join(path_output_folder, 'ranking_names.csv'), 'w')
    for i in range(len(ranking)):
        f.write(channels[ranking[i]] + "\n")
    f.close()    
 
    print(ranking)
    print(entropy_scores)




