import os
import random
import time
import numpy as np
import tensorflow as tf
import multiprocessing as mp

QUEUE_LOAD_MAX_SIZE = 64
VALID_BATCH_SIZES = [4,8,16,32,64,128,256,512]

class ThreadedPipeline():
    def __init__(self, path_classes, batch_size=32, aug_translation=None, aug_rotation=None, aug_horizontal_flip=False, aug_noise=False, use_channels='all'):
        # Save input params
        self.batch_size = batch_size
        self.aug_trans = aug_translation
        self.aug_rot = aug_rotation
        self.aug_hor_flip = aug_horizontal_flip
        self.aug_noise = aug_noise
        self.path_classes = path_classes

        # Error checks
        if len(path_classes) != 3:
            raise Exception("Must provide 3 paths - one for each class!")
        for path in path_classes:
            if not isinstance(path, str):
                raise Exception("Path to class is not a string")
            if not os.path.isfile(path):
                raise Exception("Path to class file is not a file: %s" % path)

        if batch_size not in VALID_BATCH_SIZES:
            raise Exception("Batch size must be one of the following: " + ",".join([str(bs) for bs in VALID_BATCH_SIZES]))
        
        # Load info file
        self.shapes = []
        with open(os.path.join(os.path.dirname(path_classes[0]), 'info.txt'), 'r') as f:
            for i in range(3):
                self.shapes.append(tuple((int(s) for s in f.readline().split(","))))
            self.data_dtype = f.readline()
        self.n_channels_raw = self.shapes[0][3]
        if use_channels=='all':
            self.use_channels = np.linspace(0, self.n_channels_raw, self.n_channels_raw, endpoint=False, dtype=np.uint64)
        else:
            self.use_channels = np.array(use_channels)

        # Create queues
        self.loads = [mp.Queue(maxsize=QUEUE_LOAD_MAX_SIZE) for i in range(3)]
        self.augments = [mp.Queue(maxsize=QUEUE_LOAD_MAX_SIZE) for i in range(3)]

    def __iter__(self):
        # Calculate class batch sizes and number of batches
        self.bsizes = [self.batch_size // 3] * 3
        choices = random.sample([0,1,2], self.batch_size % 3)
        for ch in choices:
            self.bsizes[ch] += 1
        self.n_batches = min(self.shapes[0][0] // self.bsizes[0], self.shapes[1][0] // self.bsizes[1], self.shapes[2][0] // self.bsizes[2])

        # Calculate batch queue load order
        self.batch_load_order = np.zeros(shape=(self.batch_size,), dtype=np.uint64)
        normal_order_length = self.batch_size - self.batch_size % 3
        for i in range(normal_order_length):
            self.batch_load_order[i] = i % 3
        
        for i in range(len(choices)):
            self.batch_load_order[normal_order_length + i] = choices[i]

        # Create Y label-vector
        self.Y = np.empty(shape=(self.batch_size,), dtype=np.uint64)
        for i in range(self.batch_size):
            self.Y[i] = self.batch_load_order[i]
        # Starting loading processes
        # for i in range(3):
        #     p = mp.Process(
        #         target=worker_loader, 
        #         args=(self.path_classes[i], self.shapes[i], self.loads[i], self.n_batches * self.bsizes[i], self.data_dtype))
        #     p.start()
        p = mp.Process(
            target=worker_loader_single,
            args=(self.path_classes, self.shapes, self.loads, self.batch_load_order, self.n_batches, self.data_dtype, self.use_channels))
        p.start()

        # Starting augmentation processes
        for i in range(3):
            p = mp.Process(
                target=worker_augmenter, 
                args=(self.loads[i],self.augments[i], self.n_batches * self.bsizes[i], self.aug_trans, self.aug_hor_flip, self.aug_noise))
            p.start()

        # Reset counters
        self.batch_no = 0
        return self


    def __next__(self):
        if self.batch_no == self.n_batches:
            raise StopIteration
        
        X = np.empty(shape=(self.batch_size, 64, 64, len(self.use_channels)), dtype=np.float32)
        #Y = np.empty(shape=(self.batch_size,), dtype=np.uint64)
        for i in range(self.batch_size):
            X[i] = self.augments[self.batch_load_order[i]].get()
            #X[i] = self.loads[self.batch_load_order[i]].get()
            #Y[i] = self.batch_load_order[i]

        self.batch_no += 1
        return (X, self.Y)

# def worker_loader(path_data, shape, q, n_samples, dtype):
#     mmap = np.memmap(path_data, dtype=dtype, shape=shape, mode='r')
#     idx_list = np.linspace(0, shape[0], shape[0], endpoint=False, dtype=np.uint64)
#     random.shuffle(idx_list)
    
#     T_SPENT_OVER_READ = 0
#     T_SPENT_OVER_PUT = 0
#     for i in range(n_samples):
#         index = idx_list[i]
#         t0 = time.time()
#         X = mmap[index]
#         T_SPENT_OVER_READ += time.time() - t0
#         t0 = time.time()
#         q.put(X)
#         T_SPENT_OVER_PUT += time.time() - t0
#     print("<worker_loader>: Time spent reading is %f and over out-queue is %f" % (T_SPENT_OVER_READ, T_SPENT_OVER_PUT))


def worker_loader_single(paths_data, shapes, qs, batch_load_order, n_batches, dtype, use_channels):
    mmaps = [np.memmap(paths_data[i], dtype=dtype, shape=shapes[i], mode='r') for i in range(len(paths_data))]
    idx_lists = [np.linspace(0, shapes[i][0], shapes[i][0], endpoint=False, dtype=np.uint64) for i in range(len(paths_data))]
    for i in range(len(idx_lists)):
        random.shuffle(idx_lists[i])

    batch_size = len(batch_load_order)
    T_SPENT_OVER_PUT = 0
    cursors = [0] * 3
    for _ in range(n_batches):
        for sample in range(batch_size):
            class_index = batch_load_order[sample]
            mmap_entry = idx_lists[class_index][cursors[class_index]]
            #X = mmaps[class_index][mmap_entry]  # original
            X = mmaps[class_index][mmap_entry][:,:,use_channels]  # should work according to: https://stackoverflow.com/questions/52267542/numpy-array-changes-shape-when-accessing-with-indices
            #X = mmaps[class_index][mmap_entry,:,:,use_channels]  # buggy
            t0 = time.time()
            qs[class_index].put(X)
            T_SPENT_OVER_PUT += time.time() - t0
            cursors[class_index] += 1
    #print("<worker_loader_single>: Time spent over out-queue is %f" % (T_SPENT_OVER_PUT))


def worker_loader_single_lowmemory(paths_data, shapes, qs, batch_load_order, n_batches, dtype, use_channels):  
    idx_lists = [np.linspace(0, shapes[i][0], shapes[i][0], endpoint=False, dtype=np.uint64) for i in range(len(paths_data))]
    for i in range(len(idx_lists)):
        random.shuffle(idx_lists[i])

    batch_size = len(batch_load_order)
    T_SPENT_OVER_PUT = 0
    cursors = [0] * 3
    for _ in range(n_batches):
        mmaps = [np.memmap(paths_data[i], dtype=dtype, shape=shapes[i], mode='r') for i in range(len(paths_data))]
        for sample in range(batch_size):
            class_index = batch_load_order[sample]
            mmap_entry = idx_lists[class_index][cursors[class_index]]
            #X = mmaps[class_index][mmap_entry]  # original
            X = mmaps[class_index][mmap_entry][:,:,use_channels]  # should work according to: https://stackoverflow.com/questions/52267542/numpy-array-changes-shape-when-accessing-with-indices
            #X = mmaps[class_index][mmap_entry,:,:,use_channels]  # buggy
            t0 = time.time()
            qs[class_index].put(X)
            T_SPENT_OVER_PUT += time.time() - t0
            cursors[class_index] += 1
    #print("<worker_loader_single>: Time spent over out-queue is %f" % (T_SPENT_OVER_PUT))


def worker_augmenter(q_in, q_out, n_samples, aug_shift, aug_hor_flip, aug_noise):
    if aug_hor_flip:
        hor_flip_rolls = np.random.randint(low=0, high=2, size=n_samples)
    
    if aug_noise:
        noise_rolls = aug_noise * np.random.rand(n_samples)
    
    T_SPENT_OVER_GET = 0
    T_SPENT_OVER_PUT = 0
    for i in range(n_samples):
        t0 = time.time()
        X = q_in.get()
        T_SPENT_OVER_GET += time.time() - t0

        # horizontal flip
        if aug_hor_flip and hor_flip_rolls[i]:
            X = np.fliplr(X)
            # X = tf.image.flip_left_right(X)  # MEMORY-LEAK !?

        # translation
        if aug_shift is not None:
            X = tf.keras.preprocessing.image.random_shift(X, aug_shift, aug_shift)

        # noise
        if aug_noise:
            X += noise_rolls[i] * np.random.randn(X.shape[0], X.shape[1], X.shape[2])
        
        t0 = time.time()
        q_out.put(X)
        T_SPENT_OVER_PUT += time.time() - t0
    
    #print("<worker_augmenter>: Time spent over in-queue is %f while over out-queue is %f" % (T_SPENT_OVER_GET, T_SPENT_OVER_PUT))


if __name__ == "__main__":
    pipe = ThreadedPipeline(path_classes=[r"E:\phantom\validation\images_Aperture\1.npy", 
                                        r"E:\phantom\validation\images_Aperture\2.npy", 
                                        r"E:\phantom\validation\images_Aperture\3.npy"], 
                            batch_size=32,
                            aug_noise=1,
                            use_channels=[10,11])
    
    X = []
    Y = []
    for (x,y) in pipe:
        if len(X) < 1:
            X.append(x)
            Y.append(y)
    
    import matplotlib.pyplot as plt
    plt.figure()
    for i in range(15):
        plt.subplot(3, 5, 5 * (i%3) + i//3 + 1)
        plt.imshow(X[0][i,:,:,1], cmap='gray', vmin=-7, vmax=7)
    plt.show()


    # 0: 1