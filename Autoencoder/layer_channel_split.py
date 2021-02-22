import tensorflow as tf


class ChannelSplit(tf.keras.layers.Layer):
    def __init__(self, keep_dim=True):
        super(ChannelSplit, self).__init__()
        self.keep_dim = keep_dim

    def build(self, input_shape):
        self.inp_shape = input_shape

    def call(self, x):
        # for encoder x.shape = (0:batch, 1:x, 2:y, 3:chs)
        # for decoder x.shape = (0:batch, 1:x, 2:y, 3:filters, 4:chs)
        splits = []
        
        if self.keep_dim:
            for i in range(x.shape[3]):
                splits.append(tf.expand_dims(x[:,:,:,i], axis=3))
        else:
            for i in range(x.shape[4]):
                splits.append(x[:,:,:,:,i])

        return splits



