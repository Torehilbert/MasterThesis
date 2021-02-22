import numpy as np
import random


# def random_sample_indices_from_mmap(ns_pool, ns_extract):
#     idx_lists = [np.linspace(0, ns_pool[i], ns_pool[i], endpoint=True, dtype=np.uint64) for i in range(len(ns_pool))]
#     for i in range(3):
#         random.shuffle(idx_lists[i])
#         idx_lists[i] = idx_lists[i][:ns_extract[i]]
#     return idx_lists


def random_sample_indices(n_available, n_extract):
    idxlist = np.linspace(0, n_available, n_available, endpoint=False, dtype=np.uint64)
    random.shuffle(idxlist)
    return idxlist[:n_extract]