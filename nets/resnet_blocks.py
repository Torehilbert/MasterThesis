import tensorflow as tf


def bottleblock_v2(inputs, filters, reduce_size=False, weight_regularizer=None, omit_initial_activation=False, use_bias=False, force_projection=False):
    x = inputs if omit_initial_activation else batchnorm_relu(inputs)
    
    x = tf.keras.layers.Conv2D(filters=filters[0], kernel_size=1, strides=2 if reduce_size else 1, padding='same', kernel_initializer='he_uniform', kernel_regularizer=weight_regularizer, use_bias=use_bias)(x)
    x = batchnorm_relu(x)
    x = tf.keras.layers.Conv2D(filters=filters[0], kernel_size=3, strides=1, padding='same', kernel_initializer='he_uniform', kernel_regularizer=weight_regularizer, use_bias=use_bias)(x)
    x = batchnorm_relu(x)
    x = tf.keras.layers.Conv2D(filters=filters[1], kernel_size=1, strides=1, padding='same', kernel_initializer='he_uniform', kernel_regularizer=weight_regularizer, use_bias=use_bias)(x)

    if reduce_size or force_projection:
        inputs = tf.keras.layers.Conv2D(filters=filters[1], kernel_size=1, strides=2 if reduce_size else 1, padding='same', kernel_initializer='he_uniform', kernel_regularizer=weight_regularizer, use_bias=use_bias)(inputs)
    
    return tf.keras.layers.Add()([inputs, x])


def block_v2(inputs, filters, reduce_size=False, weight_regularizer=None, omit_initial_activation=False, use_bias=False, force_projection=False):    
    x = inputs if omit_initial_activation else batchnorm_relu(inputs)

    x = tf.keras.layers.Conv2D(filters=filters, kernel_size=3, strides=2 if reduce_size else 1, padding='same', kernel_initializer='he_uniform', kernel_regularizer=weight_regularizer, use_bias=use_bias)(x)
    x = batchnorm_relu(x)
    x = tf.keras.layers.Conv2D(filters=filters, kernel_size=3, strides=1, padding='same', kernel_initializer='he_uniform', kernel_regularizer=weight_regularizer, use_bias=use_bias)(x)
    
    if reduce_size or force_projection:
        inputs = tf.keras.layers.Conv2D(filters=filters, kernel_size=1, strides=2 if reduce_size else 1, padding='same', kernel_initializer='he_uniform', kernel_regularizer=weight_regularizer, use_bias=use_bias)(inputs)
    
    return tf.keras.layers.Add()([inputs, x])


def resnet_head(inputs, n_classes, weight_regularizer=None):
    x = batchnorm_relu(inputs)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(n_classes, kernel_initializer='he_uniform', kernel_regularizer=weight_regularizer)(x)
    return tf.keras.layers.Softmax()(x)


def resnet_head_teacher(inputs, n_classes, weight_regularizer=None):
    x = batchnorm_relu(inputs)
    code_layer = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(n_classes, kernel_initializer='he_uniform', kernel_regularizer=weight_regularizer)(code_layer)
    return tf.keras.layers.Softmax()(x), code_layer


def resnet_stem(input_shape, filters, kernel_size, strides=1, weight_regularizer=None, use_bias=False, max_pooling=False, pool_size=(2,2)):
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same', kernel_initializer='he_uniform', kernel_regularizer=weight_regularizer, use_bias=use_bias)(inputs)
    x = batchnorm_relu(x)
    if max_pooling:
        x = tf.keras.layers.MaxPooling2D(pool_size=pool_size)(x)
    return x, inputs


def batchnorm_relu(x):
    x = tf.keras.layers.BatchNormalization()(x)
    return tf.keras.layers.ReLU()(x)

