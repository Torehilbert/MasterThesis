import os
import random
import time
import numpy as np
import tensorflow as tf


class MyDataPipeline():
    def __init__(self, path_classes, batch_size=32, translation_shift=None, aug_rotation=None, aug_flip_horizontal=False, use_channels='all'):
        self.batch_size = batch_size

        self.aug_translation = translation_shift
        self.aug_rotation = aug_rotation
        self.aug_flip_horizontal = aug_flip_horizontal

        if len(path_classes) != 3:
            raise Exception("Must provide 3 paths - one for each class!")

        self.paths = path_classes

        
        f = open(os.path.join(os.path.dirname(path_classes[0]), 'info.txt'), 'r')
        self.shapes = []
        for i in range(3):
            line = f.readline()
            self.shapes.append(tuple((int(s) for s in line.split(","))))
        self.data_dtype = f.readline()
        f.close()

        self.n_channels_raw = self.shapes[0][3]
        if use_channels=='all':
            self.use_channels = np.linspace(0, self.n_channels_raw, self.n_channels_raw, endpoint=False, dtype=np.uint64)
        else:
            self.use_channels = np.array(use_channels)

        self.mmaps = []
        for i, path in enumerate(path_classes):
            self.mmaps.append(np.memmap(path, dtype=self.data_dtype, shape=self.shapes[i]))

    def __iter__(self):
        # Calculating class batch-sizes
        self.bsizes = [self.batch_size // 3] * 3
        choices = np.array(random.sample([0,1,2], self.batch_size % 3), dtype=np.uint64)
        for ch in choices:
            self.bsizes[ch] += 1
        self.n_batches = min(self.shapes[0][0] // self.bsizes[0], self.shapes[1][0] // self.bsizes[1], self.shapes[2][0] // self.bsizes[2])

        self.splits = [self.bsizes[0], self.bsizes[0] + self.bsizes[1]]
        self.Y = np.empty(shape=(self.batch_size,), dtype=np.uint64)
        self.Y[:self.splits[0]] = 0
        self.Y[self.splits[0]:self.splits[1]] = 1
        self.Y[self.splits[1]:] = 2
        
        

        # Construct and shuffle image index order lists
        self.idx_list_1 = np.linspace(0, self.shapes[0][0], self.shapes[0][0], dtype=np.uint64, endpoint=False)
        self.idx_list_2 = np.linspace(0, self.shapes[1][0], self.shapes[1][0], dtype=np.uint64, endpoint=False)
        self.idx_list_3 = np.linspace(0, self.shapes[2][0], self.shapes[2][0], dtype=np.uint64, endpoint=False)
        random.shuffle(self.idx_list_1)
        random.shuffle(self.idx_list_2)
        random.shuffle(self.idx_list_3)
        
        # Reset cursors and batch number
        self.cursors = [0]*3
        self.batch_no = 0
        return self

    def __next__(self):
        if self.batch_no == self.n_batches:
            raise StopIteration
        
        X = np.empty(shape=(self.batch_size, 64, 64, len(self.use_channels)), dtype=np.float32)

        idx_1 = self.idx_list_1[self.cursors[0]:(self.cursors[0] + self.bsizes[0])]
        X[:self.splits[0]] = self.mmaps[0][idx_1][:,:,:,self.use_channels]

        idx_2 = self.idx_list_2[self.cursors[1]:(self.cursors[1] + self.bsizes[1])]
        X[self.splits[0]:self.splits[1]] = self.mmaps[1][idx_2][:,:,:,self.use_channels]

        idx_3 = self.idx_list_3[self.cursors[2]:(self.cursors[2] + self.bsizes[2])]
        X[self.splits[1]:] = self.mmaps[2][idx_3][:,:,:,self.use_channels]

        # increments
        self.cursors[0] += self.bsizes[0]
        self.cursors[1] += self.bsizes[1]
        self.cursors[2] += self.bsizes[2]
        self.batch_no += 1

        # augmentation translation
        with tf.device("/device:CPU:0"):
            if self.aug_flip_horizontal is not None:
                for i in range(self.batch_size):
                    if random.random() > 0.5:
                        X[i] = tf.image.flip_left_right(X[i])

            if self.aug_rotation is not None:
                for i in range(self.batch_size):
                    X[i] = tf.keras.preprocessing.image.random_rotation(X[i], self.aug_rotation)

            if self.aug_translation is not None:
                for i in range(self.batch_size):
                    X[i] = tf.keras.preprocessing.image.random_shift(X[i], self.aug_translation, self.aug_translation)
        return (X, self.Y)




if __name__ == "__main__":
    pipe = MyDataPipeline(path_classes=[r"E:\phantom\validation\images_Aperture\1.npy", 
                                        r"E:\phantom\validation\images_Aperture\2.npy", 
                                        r"E:\phantom\validation\images_Aperture\3.npy"], 
                        batch_size=32,
                        use_channels=[10,11])

    X = []
    Y = []
    for (x,y) in pipe:
        if len(X) < 1:
            X.append(x)
            Y.append(y)
    print(X[0].shape)
    import matplotlib.pyplot as plt
    plt.figure()
    for i in range(15):
        plt.subplot(3, 5, 5 * (i%3) + i//3 + 1)
        plt.imshow(X[0][i,:,:,1], cmap='gray', vmin=-7, vmax=7)
    plt.show()