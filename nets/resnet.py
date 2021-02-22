import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}
import tensorflow as tf
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from resnet_blocks import block_v2, bottleblock_v2, batchnorm_relu, resnet_head, resnet_head_teacher, resnet_stem


def ResNetV2(input_shape, 
            n_classes, 
            param_stem, # [16,3,1]
            ns, # [3,3,3]
            filters, # [16,32,64] 
            reduce_size, # [False, True, True] 
            weight_decay=(0, 0, 0, 0),
            use_bias=False, 
            stem_max_pooling=False, 
            stem_max_pool_size=(3,3)):
    
    # Creating weight regularizers
    wr_stem = _get_regularizer(l1=weight_decay[0], l2=weight_decay[1])
    wr_global = _get_regularizer(l1=weight_decay[2], l2=weight_decay[3])

    # Creating stem section
    x, inputs = resnet_stem(
        input_shape, 
        filters=param_stem[0], 
        kernel_size=param_stem[1], 
        strides=param_stem[2], 
        weight_regularizer=wr_stem, 
        use_bias=use_bias, 
        max_pooling=stem_max_pooling)

    # Creating main blocks
    for block_sequence in range(len(ns)):
        filters_curr = filters[block_sequence]
        filters_prev = filters[block_sequence - 1] if block_sequence > 0 else param_stem[0] 
        for block in range(ns[block_sequence]):
            x = block_v2(x,
                filters=filters[block_sequence],
                reduce_size=reduce_size[block_sequence] if block==0 else False,
                omit_initial_activation=True if block_sequence==0 and block==0 else False,
                force_projection=True if (filters_curr != filters_prev) and block==0 else False,
                weight_regularizer=wr_global,
                use_bias=use_bias)

    # Creating head section
    x = resnet_head(x, n_classes=n_classes, weight_regularizer=wr_global)

    # Returning model
    return tf.keras.Model(inputs, x)


def ResNetV2_Teacher(input_shape, 
                    n_classes, 
                    param_stem, # [16,3,1]
                    ns, # [3,3,3]
                    filters, # [16,32,64] 
                    reduce_size, # [False, True, True] 
                    weight_decay=(0, 0, 0, 0),
                    use_bias=False, 
                    stem_max_pooling=False, 
                    stem_max_pool_size=(3,3)):
    
    # Creating weight regularizers
    wr_stem = _get_regularizer(l1=weight_decay[0], l2=weight_decay[1])
    wr_global = _get_regularizer(l1=weight_decay[2], l2=weight_decay[3])

    # Creating stem section
    x, inputs = resnet_stem(
        input_shape, 
        filters=param_stem[0], 
        kernel_size=param_stem[1], 
        strides=param_stem[2], 
        weight_regularizer=wr_stem, 
        use_bias=use_bias, 
        max_pooling=stem_max_pooling)

    # Creating main blocks
    for block_sequence in range(len(ns)):
        filters_curr = filters[block_sequence]
        filters_prev = filters[block_sequence - 1] if block_sequence > 0 else param_stem[0] 
        for block in range(ns[block_sequence]):
            x = block_v2(x,
                filters=filters[block_sequence],
                reduce_size=reduce_size[block_sequence] if block==0 else False,
                omit_initial_activation=True if block_sequence==0 and block==0 else False,
                force_projection=True if (filters_curr != filters_prev) and block==0 else False,
                weight_regularizer=wr_global,
                use_bias=use_bias)

    # Creating head section
    x, code_layer = resnet_head_teacher(x, n_classes=n_classes, weight_regularizer=wr_global)

    # Returning model
    return tf.keras.Model(inputs, x), tf.keras.Model(inputs, code_layer)


def ResNetV2Bottleneck(input_shape, n_classes, param_stem=[16,3,1], ns=[3,3,3], filters=[64,128,256], reduce_size=[False, True, True], use_bias=False, stem_max_pooling=False, stem_max_pool_size=(3,3), weight_decay=0.0001):
    wr = tf.keras.regularizers.l2(weight_decay)

    x, inputs = resnet_stem(input_shape, filters=param_stem[0], kernel_size=param_stem[1], strides=param_stem[2], weight_regularizer=wr, use_bias=use_bias, max_pooling=stem_max_pooling)

    for block_sequence in range(len(ns)):
        filters_curr = filters[block_sequence]
        filters_bottleneck = filters_curr // 4
        filters_prev = filters[block_sequence - 1] if block_sequence > 0 else param_stem[0]
        for block in range(ns[block_sequence]):
            x = bottleblock_v2(x,
                filters=[filters_bottleneck, filters_curr],
                reduce_size=reduce_size[block_sequence] if block==0 else False,
                omit_initial_activation=True if block_sequence==0 and block==0 else False,
                force_projection=True if (filters_curr != filters_prev) and block==0 else False,
                weight_regularizer=wr,
                use_bias=use_bias)

    x = resnet_head(x, n_classes=n_classes, weight_regularizer=wr)
    return tf.keras.Model(inputs, x)


def _get_regularizer(l1, l2):
    EPS_ZERO = 1e-8
    if l1 > EPS_ZERO and l2 > EPS_ZERO:
        return tf.keras.regularizers.l1_l2(l1=l1, l2=l2)
    elif l1 < EPS_ZERO and l2 > EPS_ZERO:
        return tf.keras.regularizers.l2(l2)
    elif l1 > EPS_ZERO and l2 < EPS_ZERO:
        return tf.keras.regularizers.l1(l1)
    else:
        return None
    

def DEPRECATED_ResNetV2_50_Bottleneck(n_bands=3):
    """ ResNet-50 from "http://ethereon.github.io/netscope/#/gist/db945b393d40bfa26006" changed to ResNet-V2 with pre-activations."""

    BIAS = False
    WR = tf.keras.regularizers.l2(0.0001)

    x, inputs = resnet_stem((None, None, n_bands), filters=64, kernel_size=5, strides=1, weight_regularizer=WR, use_bias=BIAS, max_pooling=True)

    x = bottleblock_v2(x, [64,256], weight_regularizer=WR, omit_initial_activation=True, use_bias=BIAS, force_projection=True)
    x = bottleblock_v2(x, [64,256], weight_regularizer=WR, use_bias=BIAS)
    x = bottleblock_v2(x, [64,256], weight_regularizer=WR, use_bias=BIAS)

    x = bottleblock_v2(x, [128,512], reduce_size=True, weight_regularizer=WR, use_bias=BIAS)
    x = bottleblock_v2(x, [128,512], weight_regularizer=WR, use_bias=BIAS)
    x = bottleblock_v2(x, [128,512], weight_regularizer=WR, use_bias=BIAS)
    x = bottleblock_v2(x, [128,512], weight_regularizer=WR, use_bias=BIAS)

    x = bottleblock_v2(x, [256,1024], reduce_size=True, weight_regularizer=WR, use_bias=BIAS)
    x = bottleblock_v2(x, [256,1024], weight_regularizer=WR, use_bias=BIAS)
    x = bottleblock_v2(x, [256,1024], weight_regularizer=WR, use_bias=BIAS)
    x = bottleblock_v2(x, [256,1024], weight_regularizer=WR, use_bias=BIAS)
    x = bottleblock_v2(x, [256,1024], weight_regularizer=WR, use_bias=BIAS)
    x = bottleblock_v2(x, [256,1024], weight_regularizer=WR, use_bias=BIAS)

    x = bottleblock_v2(x, [512,2048], reduce_size=True, weight_regularizer=WR, use_bias=BIAS)
    x = bottleblock_v2(x, [512,2048], weight_regularizer=WR, use_bias=BIAS)
    x = bottleblock_v2(x, [512,2048], weight_regularizer=WR, use_bias=BIAS)

    x = resnet_head(x, n_classes=10, weight_regularizer=WR)
    return tf.keras.Model(inputs, x)  
