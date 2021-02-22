import tensorflow as tf


def create_encoder(
        input_shape,
        filters,
        kernel_sizes=3,
        strides=2,
        batch_norm=False,
        max_pool=False,
        activations='relu',
        weight_decay=0.0001
        ):
    
    # manage arguments
    layer_count = len(filters)
    kernel_sizes = _ensure_listlen(kernel_sizes, layer_count)
    strides = _ensure_listlen(strides, layer_count)
    batch_norm = _ensure_listlen(batch_norm, layer_count)
    max_pool = _ensure_listlen(max_pool, layer_count)
    activations = _ensure_listlen(activations, layer_count)

    # create input
    inp = tf.keras.layers.Input(input_shape)
    x = inp

    # add convolutions
    regularizer = tf.keras.regularizers.l2(weight_decay)
    for i in range(layer_count):
        conv = tf.keras.layers.Conv2D(
            filters=filters[i], 
            kernel_size=kernel_sizes[i], 
            strides=strides[i], 
            padding='same',
            kernel_regularizer=regularizer)

        x = conv(x)
        if activations[i]!='none':
            x = tf.keras.layers.Activation(activation=activations[i])(x)
        if batch_norm[i]:
            x = tf.keras.layers.BatchNormalization()(x)
        if max_pool[i]:
            x = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=1, padding='same')(x)


    # calculate output shape
    output_shape = input_shape
    for i in range(layer_count):
        output_shape = (output_shape[0] // strides[i], output_shape[0] // strides[i], filters[i])

    # return
    return tf.keras.Model(inp, x, name='encoder'), output_shape


def _ensure_listlen(x, target_length):
    # Make list
    if not isinstance(x, list):
        x = [x]
    
    # Copy last if length is insufficient
    while len(x) < target_length:
        x.append(x[-1])
    
    return x
