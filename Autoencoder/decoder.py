import tensorflow as tf


def create_decoder(
        input_shape,
        channels_out,
        filters,
        kernel_sizes=3,
        strides=2,
        batch_norm=False,
        activations='relu',
        kernel_size_output=3,
        activation_output='sigmoid',
        weight_decay=0.0001
        ):

    # manage arguments
    layer_count = len(filters)
    kernel_sizes = _ensure_listlen(kernel_sizes, layer_count)
    strides = _ensure_listlen(strides, layer_count)
    batch_norm = _ensure_listlen(batch_norm, layer_count)
    activations = _ensure_listlen(activations, layer_count)

    # create input
    input_layer = tf.keras.layers.Input(input_shape)
    x = input_layer

    # add deconvolutions
    regularizer = tf.keras.regularizers.l2(weight_decay)
    for i in range(layer_count):
        deconv = tf.keras.layers.Conv2DTranspose(
            filters=filters[i], 
            kernel_size=kernel_sizes[i], 
            strides=strides[i], 
            padding='same',
            kernel_regularizer=regularizer)
        
        x = deconv(x)
        x = tf.keras.layers.Activation(activation=activations[i])(x)
        if batch_norm[i]:
            x = tf.keras.layers.BatchNormalization()(x)
            
    # add output convolution
    conv_output = tf.keras.layers.Conv2D(
        filters=channels_out, 
        kernel_size=kernel_size_output, 
        strides=1,
        activation=activation_output, 
        padding='same')
    x = conv_output(x)

    # return
    return tf.keras.Model(input_layer, x, name='decoder')


def _ensure_listlen(x, target_length):
    # Make list
    if not isinstance(x, list):
        x = [x]
    
    # Copy last if length is insufficient
    while len(x) < target_length:
        x.append(x[-1])
    
    return x
