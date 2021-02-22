import tensorflow as tf
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from layer_channel_split import ChannelSplit
from encoder import create_encoder


def create_channel_wise_encoder(
    input_shape, 
    filters, 
    kernel_sizes=3, 
    strides=2, 
    activations='relu',
    batch_norm=False,
    max_pool=False,
    weight_decay=0.0001):

    # layers
    input_layer = tf.keras.layers.Input(input_shape)
    split_layer = ChannelSplit()
    encoder, enc_shape = create_encoder(
        input_shape=(input_shape[0], input_shape[1], 1),
        filters=filters,
        kernel_sizes=kernel_sizes,
        strides=strides,
        activations=activations,
        batch_norm=batch_norm,
        max_pool=max_pool,
        weight_decay=weight_decay)
    shape_layer = tf.keras.layers.Reshape((enc_shape[0], enc_shape[1], filters[-1], 1))
    concat_layer = tf.keras.layers.Concatenate(axis=4)
    
    # flow
    splits = split_layer(input_layer)
    encoded = []
    for split in splits:
        encoded.append(shape_layer(encoder(split)))
    out = concat_layer(encoded)

    # calculate output shape
    output_shape = (enc_shape[0], enc_shape[1], enc_shape[2], input_shape[-1])

    # return
    return tf.keras.Model(input_layer, out), output_shape
