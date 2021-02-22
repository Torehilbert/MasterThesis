import tensorflow as tf


def define_models(input_shape, layer_count=3, filters=[16,32,64], max_pools=[True, True, True], l2=None):
    encoders = []
    encoder_dimension = [input_shape]

    # Regularizer
    regularizer = tf.keras.regularizers.l2(l2) if l2 is not None else None

    # Encoders
    for l in range(layer_count):
        in_shape = encoder_dimension[l]
        enc, out = _enc_model(
            input_shape=in_shape, 
            filters=filters[l], 
            kernel_size=3, 
            bn=False, 
            activation=True, 
            max_pool=max_pools[l], 
            kernel_regularizer=regularizer)
        encoders.append(enc)
        encoder_dimension.append(out)

    # Decoders
    decoders = []
    rev_encoder_dimension = list(reversed(encoder_dimension))
    for l in range(layer_count):
        in_shape = rev_encoder_dimension[l]
        output_shape = rev_encoder_dimension[l+1]
        dec, out = _dec_model(
            input_shape=in_shape, 
            output_shape=output_shape, 
            kernel_size=3, 
            bn=False, 
            activation=True, 
            kernel_regularizer=regularizer)
        decoders.append(dec)

    # Returning
    return encoders, decoders, encoder_dimension


def assemble_model(input_shape, encs, decs, layer_count):
    pool_layer_count = len(encs)

    inp = tf.keras.layers.Input(input_shape)
    flow = inp
    #flow = tf.keras.layers.Cropping2D(cropping=())
    for l in range(pool_layer_count):
        if l < layer_count:
            flow = encs[l](flow)
    
    for l in range(pool_layer_count):
        if l >= (pool_layer_count - layer_count):
            flow = decs[l](flow)
    
    return tf.keras.Model(inp, flow)


def _enc_model(input_shape=(64,64,10), filters=16, kernel_size=3, bn=False, activation=True, max_pool=True, kernel_regularizer=None):
    inp = tf.keras.layers.Input(input_shape)
    x = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=1, padding='same', kernel_initializer="he_uniform", name='conv', kernel_regularizer=kernel_regularizer)(inp)
    x = tf.keras.layers.BatchNormalization()(x) if bn else x
    x = tf.keras.layers.ReLU()(x) if activation else x
    x = tf.keras.layers.MaxPooling2D(2)(x) if max_pool else x

    dim_reduction = 2 if max_pool else 1
    output_shape = (input_shape[0]//dim_reduction, input_shape[1]//dim_reduction, filters)
    return tf.keras.Model(inp, x), output_shape


def _dec_model(input_shape, output_shape, kernel_size, bn=False, activation=True, kernel_regularizer=None):
    perform_upsampling = input_shape[0] < output_shape[0]

    inp = tf.keras.layers.Input(input_shape)
    x = tf.keras.layers.UpSampling2D(size=2)(inp) if perform_upsampling else inp
    x = tf.keras.layers.Conv2D(filters=output_shape[2], kernel_size=kernel_size, strides=1, padding='same', kernel_initializer="he_uniform", name='conv', kernel_regularizer=kernel_regularizer)(x)
    x = tf.keras.layers.BatchNormalization()(x) if bn else x
    x = tf.keras.layers.ReLU()(x) if activation else x
    
    dim_multiplier = 2 if perform_upsampling else 1
    output_shape = (input_shape[0] * dim_multiplier, input_shape[1] * dim_multiplier, output_shape[2])   
    return tf.keras.Model(inp, x), output_shape


