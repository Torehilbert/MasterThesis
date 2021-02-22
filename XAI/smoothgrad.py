import tensorflow as tf


def smooth_grad(model, inp, noise_std, samples, squared=False, signed=True, reference_class=None):
    # ensure tf variable
    inp_tf = tf.Variable(inp)

    # get reference output neuron index
    pred = model(inp_tf)
    if(reference_class is None):
        reference_class = tf.argmax(pred[0,:])
    # get gradients wrt inputs
    inps = tf.Variable(pertubate_inputs(inp_tf, noise_std, samples))
    grads = grad_logit_wrt_input(model, inps, reference_class)
    # return sum of gradients
    if squared is False:
        if signed is True:
            return tf.reduce_mean(grads, axis=0), reference_class
        else:
            return tf.reduce_mean(tf.abs(grads), axis=0), reference_class
    else:
        if signed is True:
            return tf.reduce_mean(tf.sign(grads) * grads*grads, axis=0), reference_class
        else:
            return tf.reduce_mean(grads*grads, axis=0), reference_class


def grad_logit_wrt_input(model, inps, reference_output_index):
    with tf.GradientTape() as tape:
        tape.watch(inps)
        preds = model(inps)
        extracts = preds[:,reference_output_index]
    g = tape.gradient(extracts, inps)
    return g


def pertubate_inputs(reference_input, stddev, samples):
    return reference_input + tf.random.normal(shape=(samples, reference_input.shape[1], reference_input.shape[2], reference_input.shape[3]), stddev=stddev)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    import random

    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = np.expand_dims((x_train/255.0),axis=3) , np.expand_dims(x_test / 255.0, axis=3)

    samples = 150
    noise = 0.1
    model = tf.keras.models.load_model('model.h5')
    
    inp = tf.Variable(np.expand_dims(x_test[random.randint(0,5000)], axis=0), dtype=tf.float32) #tf.random.normal(shape=(1,28,28,1))
    g_single, c_single = smooth_grad(model, inp, 0, 1)
    g, c = smooth_grad(model, inp, noise, samples, squared=False, signed=True, reference_class=c_single)
    g2, c2 = smooth_grad(model, inp, noise, samples, squared=False, signed=False, reference_class=c_single)
    g3, c3 = smooth_grad(model, inp, noise, samples, squared=True, signed=True, reference_class=c_single)
    g4, c4 = smooth_grad(model, inp, noise, samples, squared=True, signed=False, reference_class=c_single)


    plt.figure()
    plt.subplot(2,3,1)
    plt.title('input')
    plt.imshow(inp[0,:,:,0], cmap='gray')

    plt.subplot(2,3,2)
    plt.title('signed')
    maxval = max(np.max(g[:,:,-1]), -np.min(g[:,:,-1]))
    plt.imshow(g[:,:,-1], cmap='bwr', vmin=-maxval, vmax=maxval)

    plt.subplot(2,3,3)
    plt.title('non-signed')
    maxval = max(np.max(g2[:,:,-1]), -np.min(g2[:,:,-1]))
    plt.imshow(g2[:,:,-1], cmap='bwr', vmin=-maxval, vmax=maxval)

    plt.subplot(2,3,4)
    plt.title('1 gradient (%d)' % c_single)
    maxval = max(np.max(g_single[:,:,-1]), -np.min(g_single[:,:,-1]))
    plt.imshow(g_single[:,:,-1], cmap='bwr', vmin=-maxval, vmax=maxval)

    plt.subplot(2,3,5)
    plt.title('squared-signed')
    maxval = max(np.max(g3[:,:,-1]), -np.min(g3[:,:,-1]))
    plt.imshow(g3[:,:,-1], cmap='bwr', vmin=-maxval, vmax=maxval)

    plt.subplot(2,3,6)
    plt.title('squared-non-signed')
    maxval = max(np.max(g4[:,:,-1]), -np.min(g4[:,:,-1]))
    plt.imshow(g4[:,:,-1], cmap='bwr', vmin=-maxval, vmax=maxval)

    plt.show()