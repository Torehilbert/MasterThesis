import tensorflow as tf


def load(resolution=None):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train = x_train/255.0
    x_test = x_test/255.0

    if(resolution is not None):
        x_train = tf.image.resize(images=x_train, size=resolution, method='bilinear')
        x_test = tf.image.resize(images=x_test, size=resolution, method='bilinear')
    
    return (x_train, y_train), (x_test, y_test)


if __name__ == "__main__":
    train, test = load(resolution=[64,64])