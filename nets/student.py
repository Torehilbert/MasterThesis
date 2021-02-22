import tensorflow as tf


def student_network(input_shape, n_output_codes, hidden_factor=10, l1_input_reg=0.01, l2_hidden_reg=0.0001, input_reg_type='original', return_input_regularizer=False):
    # input: 64x64x10, Output: 64x64x10
    inp = tf.keras.layers.Input(shape=input_shape)  
    
    if input_reg_type=='original':
        inp_reg_func = lambda x: student_input_regularization(x, l1_input_reg)
    elif input_reg_type=='new':
        inp_reg_func = lambda x: student_input_regularization_more_sparse(x, l1_input_reg)


    # N = n_output_codes * hidden_factor
    # input: 64x64x10, Output: 32x32xN
    x = tf.keras.layers.Conv2D(
        filters=hidden_factor*n_output_codes, 
        kernel_size=5, 
        strides=2, 
        padding='same', 
        kernel_initializer='he_uniform', 
        kernel_regularizer=inp_reg_func, 
        use_bias=True)(inp)
    
    # input: 32x32xN, Output: 32x32xN
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    # input: 32x32xN, Output: 16x16x64
    x = tf.keras.layers.Conv2D(
        filters=n_output_codes, 
        kernel_size=3, 
        strides=2, 
        padding='same', 
        kernel_initializer='he_uniform', 
        kernel_regularizer=tf.keras.regularizers.l2(l2_hidden_reg), 
        use_bias=True)(x)

    # input: 16x16x64, Output: 16x16x64
    #x = tf.keras.layers.ReLU()(x)

    # input: 16x16x64, Output: 64
    x = tf.keras.layers.GlobalAveragePooling2D()(x)  
    
    if return_input_regularizer:
        return tf.keras.Model(inp, x), inp_reg_func
    else:
        return tf.keras.Model(inp, x)


def student_input_regularization(weight_matrix, l1=0.01):
    squares = tf.multiply(weight_matrix, weight_matrix)
    sums = tf.reduce_sum(squares, axis=[0,1,3])
    roots = tf.sqrt(sums)
    finalsum = tf.reduce_sum(roots)
    return l1*finalsum


def student_input_regularization_more_sparse(weight_matrix, l1=0.01):
    squares = tf.multiply(weight_matrix, weight_matrix)
    sums = tf.reduce_sum(squares, axis=[0,1,3])
    roots = tf.sqrt(tf.sqrt(sums))
    finalsum = tf.reduce_sum(roots)
    return l1*finalsum


def student_channel_importance(weight_matrix):
    return tf.reduce_sum(tf.multiply(weight_matrix, weight_matrix), axis=[0,1,3]).numpy()

# def student_input_regularization(self, weight_matrix):
#     return self.l1*K.sum(K.sqrt(K.tf.reduce_sum(K.square(weight_matrix), axis=1)))
