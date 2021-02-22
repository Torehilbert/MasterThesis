import tensorflow as tf
import numpy as np


def DSCNet(input_shape, encode_filters=[32, 64], max_pool_strides=[2,2], l2=None, selfexpr_l1=None, selfexpr_l2=None, selfexpr_constraint_type="hardset", alpha=1.0):
    inp = tf.keras.layers.Input(input_shape)

    # *create encoder model
    encoder = Encoder(
        input_shape=(input_shape[0], input_shape[1], 1),
        filters=encode_filters,
        max_pool_strides=max_pool_strides,
        l2=l2,
        add_trailing_dimension=True,
    )

    # A) Encode channels individually
    CHS = ChannelSplitter(keep_dimension=True)(inp)
    codes = []
    for i in range(input_shape[-1]):
        codes.append(encoder(CHS[i]))
    codes_merged = tf.keras.layers.Concatenate(axis=4)(codes)

    # B) Perform self-expression on encodings
    codes_merged = SelfExpressive(
        constraint_type=selfexpr_constraint_type, 
        l1=selfexpr_l1, 
        l2=selfexpr_l2, 
        alpha=alpha,
        name='selfexp')(codes_merged)

    # *create decoder model
    decoder = Decoder(
        input_shape=(input_shape[0] // _prod(max_pool_strides), input_shape[1] // _prod(max_pool_strides), encode_filters[-1]), 
        filters=list(reversed(encode_filters[:-1])) + [1], 
        do_upsample=[stride==2 for stride in max_pool_strides],
        l2=l2)

    # C) Decode channels individually
    codes = ChannelSplitter(keep_dimension=False)(codes_merged)
    CHS = []
    for i in range(input_shape[-1]):
        CHS.append(decoder(codes[i]))
    
    # D) Return
    X_RECON = tf.keras.layers.Concatenate(axis=3)(CHS)
    return tf.keras.Model(inp, X_RECON)



def Encoder(input_shape, filters, max_pool_strides, l2=None, add_trailing_dimension=False):
    if(len(filters) != len(max_pool_strides)):
        raise Exception("Length of filters of max_pool_strides do not match!")

    
    inp = tf.keras.layers.Input(input_shape)
    for i, (filt, stride) in enumerate(zip(filters, max_pool_strides)):
        conv2d = tf.keras.layers.Conv2D(
            filters=filt,
            kernel_size=(3,3),
            strides=(1,1),
            padding='same',
            kernel_regularizer=tf.keras.regularizers.l2(l2) if l2 is not None else None,
            kernel_initializer='he_uniform'
        )

        flow = conv2d(inp if i==0 else flow)
        flow = tf.keras.layers.BatchNormalization()(flow)
        flow = tf.keras.layers.ReLU()(flow)
        flow = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(stride, stride), padding='same')(flow)
    

    if add_trailing_dimension:
        output_resolution = (input_shape[0] // _prod(max_pool_strides), input_shape[1] // _prod(max_pool_strides))
        flow = tf.keras.layers.Reshape((output_resolution[0], output_resolution[1], filters[-1], 1))(flow)

    return tf.keras.Model(inp, flow)



def Decoder(input_shape, filters, do_upsample, l2=None):
    inp = tf.keras.layers.Input(input_shape)
    flow = inp

    for i, filt in enumerate(filters):
        conv2d = tf.keras.layers.Conv2D(
            filters=filt, 
            kernel_size=(3,3), 
            padding='same', 
            kernel_regularizer=tf.keras.regularizers.l2(l2) if l2 is not None else None,
            kernel_initializer='he_uniform'
        )

        if do_upsample[i]:
            flow = tf.keras.layers.UpSampling2D((2,2))(flow)

        flow = conv2d(flow)

        if i != (len(filters) - 1):
            flow = tf.keras.layers.BatchNormalization()(flow)
            flow = tf.keras.layers.ReLU()(flow)
    
    return tf.keras.Model(inp, flow)



class SelfExpressive(tf.keras.layers.Layer):
    def __init__(self, constraint_type="hardset", l1=None, l2=None, alpha=1.0, name='selfexp'):
        super(SelfExpressive, self).__init__(name=name)
        self.constraint_type = constraint_type
        self.l1 = l1
        self.l2 = l2
        self.alpha = alpha
        self.mse = tf.keras.losses.MeanSquaredError()

    def build(self, input_shape):
        self.num_channels = input_shape[-1]

        # Regularizer
        if self.l1 is None and self.l2 is None:
            reg = None
        elif self.l2 is None:
            reg = tf.keras.regularizers.l1(self.l1)
        elif self.l1 is None:
            reg = tf.keras.regularizers.l2(self.l2)
        else:
            reg = tf.keras.regularizers.l1_l2(l1=self.l1, l2=self.l2)

        # Constraint
        if(self.constraint_type=="hardset"):
            con = DiagonalZeroConstraint(self.num_channels)
        else:
            con = None
        
        # Kernel
        self.kernel = self.add_weight(
            "kernel", 
            shape=(self.num_channels, self.num_channels), 
            regularizer=reg, 
            constraint=con
        )


    def call(self, x):
        out = []
        for i in range(self.num_channels):
            out.append(tf.reduce_sum(self.kernel[i] * x, axis=4, keepdims=True))
        xre = tf.concat(out, axis=4)
        loss = self.alpha * tf.reduce_mean(tf.keras.losses.mean_squared_error(x, xre))
        self.add_loss(loss)
        return xre


class ChannelSplitter(tf.keras.layers.Layer):
    def __init__(self, keep_dimension=False):
        super(ChannelSplitter, self).__init__()
        self.KEEP_DIM = keep_dimension


    def build(self, input_shape):
        self.SHAPE_LEN = len(input_shape)
        self.N_CHANNELS = input_shape[-1]


    def call(self, x):
        extracts = []

        if self.SHAPE_LEN == 4:

            if self.KEEP_DIM:
                for i in range(self.N_CHANNELS):
                    extracts.append(tf.expand_dims(x[:,:,:,i], axis=3))
            else:
                for i in range(self.N_CHANNELS):
                    extracts.append(x[:,:,:,i])

        elif self.SHAPE_LEN == 5:

            if self.KEEP_DIM:
                for i in range(self.N_CHANNELS):
                    extracts.append(tf.expand_dims(x[:,:,:,:,i], axis=4))
            else:
                for i in range(self.N_CHANNELS):
                    extracts.append(x[:,:,:,:,i])          
        else:
            raise Exception("Unsupported dimensions!")

        return extracts


class DiagonalZeroConstraint(tf.keras.constraints.Constraint):
    def __init__(self, size):
        self.diag = np.array([0]*size, dtype=np.float32)


    def __call__(self, w):
        diag = tf.linalg.diag(tf.linalg.diag_part(w))
        return w - diag


def _prod(l):
    p = 1
    for el in l:
        p = p*el
    return p
