import os
import random
import numpy as np
import tensorflow as tf
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import wfutils.mmap
import wfutils.infofile


class AutoencoderDataPipe():
    def __init__(self, path_data, batch_size, use_separate_thread=False, rescale_mins=None, rescale_maxs=None, cropping=0, aug_noise_level=0.0, aug_random_flip=0, aug_random_shift=0.0):
        shapes, data_dtype = wfutils.infofile.read_info(wfutils.infofile.get_path_to_infofile(path_data))
        self.mmaps = wfutils.mmap.get_class_mmaps_read(path_data, shapes, data_dtype)
        self.batch_size = batch_size

        batch_size_each_class = self.batch_size//3
        batch_size_remainder = self.batch_size - 3*batch_size_each_class

        self.batch_size_class_wise = [batch_size_each_class, batch_size_each_class, batch_size_each_class + batch_size_remainder]
        self.number_of_batches = min(
            self.mmaps[0].shape[0] // self.batch_size_class_wise[0],
            self.mmaps[1].shape[0] // self.batch_size_class_wise[1],
            self.mmaps[2].shape[0] // self.batch_size_class_wise[2]
        )

        self.use_rescale = (rescale_mins is not None) and (rescale_maxs is not None)
        if self.use_rescale:
            self.rescale_mins = np.array(rescale_mins, dtype="float32")
            self.rescale_maxs = np.array(rescale_maxs, dtype="float32")

        self.use_cropping = cropping > 0
        self.cropping = tf.keras.layers.Cropping2D(cropping=cropping)

        self.use_augmentation = (aug_noise_level > 0.001) or (aug_random_flip==1) or (aug_random_shift > 0.001)
        self.aug_noise_level = aug_noise_level
        self.aug_random_flip = aug_random_flip
        self.aug_random_shift = aug_random_shift




    def __iter__(self):
        # Construct and shuffle image index order lists
        self.idx_list_1 = np.linspace(0, self.mmaps[0].shape[0], self.mmaps[0].shape[0], dtype=np.uint64, endpoint=False)
        self.idx_list_2 = np.linspace(0, self.mmaps[1].shape[0], self.mmaps[1].shape[0], dtype=np.uint64, endpoint=False)
        self.idx_list_3 = np.linspace(0, self.mmaps[2].shape[0], self.mmaps[2].shape[0], dtype=np.uint64, endpoint=False)
        random.shuffle(self.idx_list_1)
        random.shuffle(self.idx_list_2)
        random.shuffle(self.idx_list_3)

        # Reset cursors and batch number
        self.cursors = [0]*3
        self.batch_number = 0

        return self

    def __next__(self):
        if self.batch_number == self.number_of_batches:
            raise StopIteration   
        self.batch_number += 1

        return self._next()


    def _next(self):
        # Create batch placeholder
        X = np.empty(shape=(self.batch_size, self.mmaps[0].shape[1], self.mmaps[0].shape[2], self.mmaps[0].shape[-1]), dtype=np.float32)

        # extract indices
        idx_1 = self.idx_list_1[self.cursors[0]:(self.cursors[0] + self.batch_size_class_wise[0])]
        idx_2 = self.idx_list_2[self.cursors[1]:(self.cursors[1] + self.batch_size_class_wise[1])]
        idx_3 = self.idx_list_3[self.cursors[2]:(self.cursors[2] + self.batch_size_class_wise[2])]

        # Fill batch
        batch_cursor = 0
        for cl, idx in enumerate([idx_1, idx_2, idx_3]):
            ind_start = batch_cursor
            ind_end = batch_cursor + self.batch_size_class_wise[cl]
            X[ind_start:ind_end] = self.mmaps[cl][idx]
            batch_cursor = ind_end

        # increments
        self.cursors[0] += self.batch_size_class_wise[0]
        self.cursors[1] += self.batch_size_class_wise[1]
        self.cursors[2] += self.batch_size_class_wise[2]


        # Cropping
        if self.use_cropping:
            X = self.cropping(X)
        
        # rescale
        if self.use_rescale:
            X = (X-self.rescale_mins)/(self.rescale_maxs - self.rescale_mins)

        # augmentation
        if self.use_augmentation:
            X,Y = self.augment(X)
        else:
            Y = X

        # return
        return X,Y


    def augment(self, X):
        batch_size = X.shape[0]

        X = np.array(X, copy=True, dtype="float32")

        # Random flip
        if self.aug_random_flip==1:
            rolls_flip = np.random.rand(batch_size)
            for i in range(batch_size):
                if rolls_flip[i] > 0.5:
                    X[i] = tf.image.flip_left_right(X[i])

        # Random shift
        if self.aug_random_shift > 0.001:
            for i in range(batch_size):
                X[i] = tf.keras.preprocessing.image.random_shift(X[i], self.aug_random_shift, self.aug_random_shift)

        Y = np.array(X, copy=True, dtype="float32")

        # Random noise
        if(self.aug_noise_level is not None and self.aug_noise_level > 0.001):
            X += self.aug_noise_level * np.random.randn(X.shape[0], X.shape[1], X.shape[2], X.shape[3])

        # Return
        return X, Y  


if __name__ == "__main__":
    path_data = r"E:\full32_redist"
    pipe = AutoencoderDataPipe(path_data=path_data, batch_size=32)

    count = 0
    last_sum = None
    for x in pipe:
        count += 1
        s = np.sum(x)
        if last_sum is not None and s==last_sum:
            print("ERROR")

    print(count)
