import os
import tensorflow as tf
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from encoder import create_encoder
from decoder import create_decoder


def create_autoencoder(
        input_shape,
        encoder_filters,
        decoder_filters,
        encoder_kernel_sizes=3,
        decoder_kernel_sizes=3,
        encoder_strides=3,
        decoder_strides=3,
        encoder_activations='relu',
        decoder_activations='relu',
        decoder_output_kernel_size=3,
        decoder_output_activation='sigmoid',
        weight_decay=0.0001
        ):

    # encoding
    encoder, encoding_shape = create_encoder(
        input_shape=input_shape, 
        filters=encoder_filters, 
        kernel_sizes=encoder_kernel_sizes, 
        strides=encoder_strides, 
        activations=encoder_activations,
        weight_decay=weight_decay)

    # decoding
    decoder = create_decoder(
        input_shape=encoding_shape,
        channels_out=input_shape[2],
        filters=decoder_filters,
        kernel_sizes=decoder_kernel_sizes,
        strides=decoder_strides,
        activations=decoder_activations,
        kernel_size_output=decoder_output_kernel_size,
        activation_output=decoder_output_activation,
        weight_decay=weight_decay)

    # input
    inp = tf.keras.layers.Input(input_shape, name='input')

    # return
    return tf.keras.Model(inp, decoder(encoder(inp)))
