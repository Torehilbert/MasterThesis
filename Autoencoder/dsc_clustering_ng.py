import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, set_link_color_palette
from sklearn.metrics import silhouette_score
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import wfutils.log
from wfutils import CHANNELS


def get_cluster_model(D, linkage):
    return AgglomerativeClustering(
        affinity='precomputed',
        distance_threshold=0,
        n_clusters=None,
        compute_full_tree=True,
        linkage=linkage).fit(D)


def assemble_cluster_sets(model):
    n_clusters = model.n_clusters_

    cluster_sets = []
    
    clusters = {}
    next_cluster_id = n_clusters
    for i in range(n_clusters):
        clusters[i] = [i]
    
    for merge in model.children_:
        # export to output list
        cluster_sets.append([])
        for (key,value) in clusters.items():
            cluster_sets[-1].append(tuple(value))
        
        new_cluster = []
        new_cluster.extend(clusters[merge[0]])
        new_cluster.extend(clusters[merge[1]])

        del clusters[merge[0]]
        del clusters[merge[1]]
        clusters[next_cluster_id] = new_cluster
        next_cluster_id += 1
    
    # add final cluster (complete) not included in model.children_
    cluster_sets.append([tuple([i for i in range(n_clusters)])])

    return cluster_sets


def write_cluster_sets(path_file, cluster_sets):
    with open(path_file, 'w') as f:
        for cluster_set in cluster_sets:
            cluster_strings = []
            for cluster in cluster_set:
                cluster_strings.append("-".join(["%d" % ch for ch in cluster]))
            cluster_set_string = ",".join(cluster_strings)
            f.write(cluster_set_string + "\n")


def read_cluster_sets(path_file):
    with open(path_file, 'r') as f:
        cluster_set_lines = f.readlines()
        cluster_sets = []
        for cluster_set_string in cluster_set_lines:
            cluster_set = []
            cluster_strings = cluster_set_string.split(",")
            for cstring in cluster_strings:
                cluster_set.append(tuple([int(ch_string) for ch_string in cstring.split("-")]))
            cluster_sets.append(cluster_set)
    return cluster_sets


def create_dendrogram_plot(linkage_matrix, threshold, channel_labels, ylabel='Distance'):
    BIAS_ADD = 0.01
    set_link_color_palette(['k']*20)
    fig = plt.figure()
    color_threshold = threshold
    linkage_added_bias = linkage_matrix
    linkage_added_bias[:,2] += BIAS_ADD
    dendrogram(
        linkage_added_bias, 
        above_threshold_color='k',
        labels=channel_labels)
    plt.xticks(rotation=-65)
    plt.ylabel(ylabel)
    plt.hlines(color_threshold + BIAS_ADD, 0, 100, linestyles='dashed', colors='gray')

    yticks = np.linspace(0, np.max(linkage_matrix[:,2]), 5, endpoint=True)
    yticklabels = ["%.2f" % val for val in yticks]
    plt.yticks(ticks=yticks + BIAS_ADD, labels=yticklabels)
    plt.tight_layout()
    return fig


def save_dendrogram_plot(path_file, linkage_matrix, threshold, channel_labels, ylabel='Distance', dpi=250):
    fig = create_dendrogram_plot(linkage_matrix, threshold, channel_labels, ylabel)
    plt.savefig(path_file, dpi=dpi)
    plt.close(fig)


def calculate_linkage_matrix(model):
    ns = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        n_current = 0
        for child_idx in merge:
            if child_idx < n_samples:
                n_current += 1  # leaf node
            else:
                n_current += ns[child_idx - n_samples]
        ns[i] = n_current

    return np.column_stack([model.children_, model.distances_, ns]).astype(float)


def get_clusters(cluster_sets, linkage_matrix, confidence_threshold):
    n_merges = np.argmax(linkage_matrix[:,2] > confidence_threshold * np.max(linkage_matrix[:,2]))
    return cluster_sets[n_merges]


def perform_hierarchical_clustering(D, linkage_method):
    model = get_cluster_model(D, linkage_method)
    linkage_matrix = calculate_linkage_matrix(model)  
    cluster_sets = assemble_cluster_sets(model)
    return cluster_sets, linkage_matrix


# def elbow_cluster_count_determination(D, cluster_sets):
#     rev_cluster_sets = list(reversed(cluster_sets))

#     all_wss = np.zeros(shape=(len(rev_cluster_sets,)))
#     for k,cluster_set in enumerate(rev_cluster_sets):
#         wss = 0
#         for cluster in cluster_set:
#             wss_cluster = 0
#             for i in range(len(cluster)):
#                 for j in range(i+1, len(cluster)):
#                     ch1 = cluster[i]
#                     ch2 = cluster[j]
#                     dist = D[ch1,ch2]
#                     wss_cluster += dist*dist
#             wss += wss_cluster
#         all_wss[k] = wss
#     plt.figure()
#     plt.plot(all_wss, '-ok')
#     plt.show()

def extract_clusters(k, cluster_sets):
    rev_cluster_sets = list(reversed(cluster_sets))
    return rev_cluster_sets[k-1]


def convert_cluster_set_to_labels(cluster_set):
    N = sum([sum([1 for ch in cluster]) for cluster in cluster_set])  
    labels = np.ones(shape=(N,), dtype=int)
    for l, cluster in enumerate(cluster_set):
        for ch in cluster:
            labels[ch] = l
    return labels


def calc_silhouette_scores(D, cluster_sets):
    rev_cluster_sets = list(reversed(cluster_sets))
    
    silscores = []
    for i in range(1, len(rev_cluster_sets) - 1):
        cluster_set = rev_cluster_sets[i]      
        labels = convert_cluster_set_to_labels(cluster_set)
        silscores.append(silhouette_score(D, labels=labels, metric='precomputed'))
    return np.array(silscores)


def silhouette_k_select(D, cluster_sets):
    sil_scores = calc_silhouette_scores(D, cluster_sets)
    optimal_idx = np.argmax(sil_scores)
    optimal_k = optimal_idx + 2
    return optimal_k, sil_scores


parser = argparse.ArgumentParser()
parser.add_argument("-root_folder", required=False, type=str, default=r"D:\Speciale\Code\output\DSC_New\Runs_LONG")
parser.add_argument("-filename_d_matrix", required=False, type=str, nargs="+", default=["D_subtract.csv", "D_division.csv", "D_norm_subtract.csv", "D_norm_division.csv"])
parser.add_argument("-labels_distance_files", required=False, type=str, nargs="+", default=['subtract', 'division', 'norm_subtract', 'norm_division'])
parser.add_argument("-linkages", required=False, type=str, nargs="+", default=["average", "single", "complete"])
parser.add_argument("-dpi", required=False, type=int, default=250)
parser.add_argument("-overwrite_analysis", required=False, type=int, default=1)


if __name__ == "__main__":
    args = parser.parse_args()

    if args.root_folder is None:
        print("Root folder argument was not supplied as argument!")
        exit(1)

    conts = os.listdir(args.root_folder)
    for i, subfolder in enumerate(conts):
        print("%d/%d : %s" % (i+1, len(conts), subfolder))
        path_subfolder = os.path.join(args.root_folder, subfolder)
        paths_d_matrix = [os.path.join(path_subfolder, 'matrices', fname) for fname in args.filename_d_matrix]

        # Ensure directory and D matrix available
        if not os.path.isdir(path_subfolder):
            continue
        for path_d in paths_d_matrix:
            if not os.path.isfile(path_d):
                print("Warning: Could not find file %s!" % path_d)
                continue
        
        # Create output folder
        path_output_folder = os.path.join(path_subfolder, 'clustering')
        if os.path.isdir(path_output_folder):
            if args.overwrite_analysis==0:
                print("Warning: Clustering folder already exist - while overwrite is DISABLED!")
                continue
            else:
                pass
        else:
            os.makedirs(path_output_folder)

        # Log arguments
        wfutils.log.log_arguments(path_output_folder, args)

        # Load distance matrix
        for (path_d, label) in zip(paths_d_matrix, args.labels_distance_files):
            if not os.path.isfile(path_d):
                print("Warning: Could not find D file: %s" % path_d)
                continue

            D = np.loadtxt(path_d, delimiter=',')

            for j, linkage in enumerate(args.linkages):
                # Perform clustering
                cluster_sets, linkage_matrix = perform_hierarchical_clustering(D, linkage)
                
                # Cut dendrogram and select clusters
                optimal_k, sil_scores = silhouette_k_select(D, cluster_sets)                
                dendrogram_cut_value = (np.flipud(linkage_matrix)[optimal_k-2, 2] + np.flipud(linkage_matrix)[optimal_k-1, 2])/2
                clusters = extract_clusters(optimal_k, cluster_sets)

                # Save results
                write_cluster_sets(os.path.join(path_output_folder, 'cluster_levels_%s_%s.txt' % (label, linkage)), cluster_sets)
                write_cluster_sets(os.path.join(path_output_folder, 'clusters_%s_%s.txt' % (label, linkage)), [clusters])
                np.savetxt(os.path.join(path_output_folder, 'linkage_%s_%s.csv' % (label, linkage)), linkage_matrix, delimiter=',')
                np.savetxt(os.path.join(path_output_folder, 'optimal_k_%s_%s.txt' % (label, linkage)), [optimal_k], delimiter=',')
                np.savetxt(os.path.join(path_output_folder, 'silhouettes_%s_%s.txt' % (label, linkage)), sil_scores, delimiter=',')

                # Save plot
                save_dendrogram_plot(
                    path_file=os.path.join(path_output_folder, 'dendrogram_%s_%s.png' % (label, linkage)),
                    linkage_matrix=linkage_matrix,
                    threshold=dendrogram_cut_value,
                    channel_labels=CHANNELS,
                    ylabel='Distance (%s)' % linkage,
                    dpi=args.dpi
                )

                # Silhouette plot
                plt.figure()
                plt.plot(np.arange(2, 10, 1), sil_scores, '-ok')
                plt.plot(optimal_k, sil_scores[optimal_k-2], 'or')
                plt.ylabel("Average silhouette score")
                plt.xlabel("Num. clusters (k)")
                plt.tight_layout()
                plt.savefig(os.path.join(path_output_folder, 'silhouettes_%s_%s.png' % (label, linkage)), dpi=args.dpi)
                plt.close()