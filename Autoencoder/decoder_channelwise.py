import tensorflow as tf
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from layer_channel_split import ChannelSplit
from decoder import create_decoder


def create_channel_wise_decoder(
    input_shape,
    filters,
    kernel_sizes=3,
    strides=2,
    activations='relu',
    batch_norm=False,
    kernel_size_output=3,
    activation_output='sigmoid',
    weight_decay=0.0001):

    # layers
    input_layer = tf.keras.layers.Input(input_shape)
    split_layer = ChannelSplit(keep_dim=False)
    decoder = create_decoder(
        input_shape=(input_shape[0], input_shape[1], input_shape[2]),
        channels_out=1,
        filters=filters,
        kernel_sizes=kernel_sizes,
        strides=strides,
        batch_norm=batch_norm,
        activations=activations,
        kernel_size_output=kernel_size_output,
        activation_output=activation_output,
        weight_decay=weight_decay
    )
    concat_layer = tf.keras.layers.Concatenate(axis=3)

    # flow
    splits = split_layer(input_layer)
    decoded = []
    for split in splits:
        decoded.append(decoder(split))
    out = concat_layer(decoded)

    # return
    return tf.keras.Model(input_layer, out)
