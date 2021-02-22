import tensorflow as tf


def AutoEncoder(input_shape):
    inp = tf.keras.layers.Input(input_shape)
    x = tf.keras.layers.Flatten()(inp)
    x = tf.keras.layers.Dense(2, activation='relu')(x)
    x = tf.keras.layers.Dense(784, activation='sigmoid')(x)
    x = tf.keras.layers.Reshape((28,28))(x)
    return tf.keras.Model(inp, x)


def DeepAutoEncoder(input_shape, bottleneck_size=10):
    inp = tf.keras.layers.Input(input_shape)
    x = tf.keras.layers.Flatten()(inp)
    x = tf.keras.layers.Dense(500)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dense(250)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dense(64)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dense(bottleneck_size, activation='relu')(x)
    x = tf.keras.layers.Dense(64)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dense(250)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dense(784, activation='sigmoid')(x)
    x = tf.keras.layers.Reshape((28,28))(x)
    return tf.keras.Model(inp, x)   


def ConvolutionalAutoEncoder(input_shape):
    inp = tf.keras.layers.Input(input_shape)
    x = tf.keras.layers.Conv2D(32, (3,3), padding='same')(inp)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D((2,2))(x)  # 14,14
    x = tf.keras.layers.Conv2D(32, (3,3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D((2,2))(x) # 7, 7

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(10, activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-5))(x)
    x = tf.keras.layers.Dense(1568, activation='relu')(x)
    x = tf.keras.layers.Reshape((7,7,32))(x)

    x = tf.keras.layers.Conv2D(32, (3,3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.UpSampling2D((2,2))(x) # 14,14
    x = tf.keras.layers.Conv2D(32, (3,3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.UpSampling2D((2,2))(x) # 28, 28
    x = tf.keras.layers.Conv2D(1, (3,3), padding='same')(x)
    x = tf.keras.layers.Activation(activation='sigmoid')(x)
    return tf.keras.Model(inp, x)


def lr_schedule(epoch, lr_current):
    if epoch < 10:
        return 0.1
    elif epoch < 20:
        return 0.01
    else:
        return 0.001


if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import data.utils.mnist as mnist

    train, test = mnist.load()
    
    opt = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)
    loss = tf.keras.losses.BinaryCrossentropy()
    model = ConvolutionalAutoEncoder((28,28, 1))
    model.compile(optimizer=opt, loss=loss)

    lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_schedule)
    model.fit(train[0], train[0], batch_size=256, epochs=15, validation_data=(test[0], test[0]), callbacks=[lr_callback])

    import matplotlib.pyplot as plt

    pred = model(test[0][0:5,:,:,:])

    plt.figure()
    plt.subplot(2,5,1)
    plt.imshow(test[0][0,:,:,0], cmap='gray')
    plt.subplot(2,5,6)
    plt.imshow(pred[0, :,:,0], cmap='gray')

    plt.subplot(2,5,2)
    plt.imshow(test[0][1,:,:,0], cmap='gray')
    plt.subplot(2,5,7)
    plt.imshow(pred[1, :,:,0], cmap='gray')

    plt.subplot(2,5,3)
    plt.imshow(test[0][2,:,:,0], cmap='gray')
    plt.subplot(2,5,8)
    plt.imshow(pred[2, :,:,0], cmap='gray')

    plt.subplot(2,5,4)
    plt.imshow(test[0][3,:,:,0], cmap='gray')
    plt.subplot(2,5,9)
    plt.imshow(pred[3, :,:,0], cmap='gray')

    plt.subplot(2,5,5)
    plt.imshow(test[0][4,:,:,0], cmap='gray')
    plt.subplot(2,5,10)
    plt.imshow(pred[4, :,:,0], cmap='gray')
    plt.show()