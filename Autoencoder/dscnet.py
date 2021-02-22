import os
import tensorflow as tf
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from encoder_channelwise import create_channel_wise_encoder
from decoder_channelwise import create_channel_wise_decoder
from layer_self_expression import SelfExpression


def create_dscnet(
    input_shape,
    encoder_filters,
    decoder_filters,
    encoder_kernel_sizes=3,
    decoder_kernel_sizes=3,
    encoder_strides=3,
    decoder_strides=3,
    encoder_maxpool=False,
    encoder_activations='relu',
    decoder_activations='relu',
    encoder_batch_norm=False,
    decoder_batch_norm=False,
    decoder_output_kernel_size=3,
    decoder_output_activation='sigmoid',
    autoencoder_weight_decay=0.0001,
    selfexpressive_weight_decay=0.0001):


# encoding
    encoder, encoding_shape = create_channel_wise_encoder(
        input_shape=input_shape, 
        filters=encoder_filters, 
        kernel_sizes=encoder_kernel_sizes, 
        strides=encoder_strides, 
        activations=encoder_activations,
        batch_norm=encoder_batch_norm,
        max_pool=encoder_maxpool,
        weight_decay=autoencoder_weight_decay)

    # self expressive layer
    self_expression = SelfExpression(encoding_shape[-1], l2=selfexpressive_weight_decay)

    # decoding
    decoder = create_channel_wise_decoder(
        input_shape=encoding_shape,
        filters=decoder_filters,
        kernel_sizes=decoder_kernel_sizes,
        strides=decoder_strides,
        activations=decoder_activations,
        batch_norm=decoder_batch_norm,
        kernel_size_output=decoder_output_kernel_size,
        activation_output=decoder_output_activation,
        weight_decay=autoencoder_weight_decay
    )

    # input
    inp = tf.keras.layers.Input(input_shape)

    # flow
    U = encoder(inp)
    Ure = self_expression(U)
    Xre = decoder(Ure)

    # return
    return tf.keras.Model(inputs=inp, outputs=[Xre, U, Ure])