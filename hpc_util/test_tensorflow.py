import tensorflow as tf


if __name__ == "__main__":
    
    A = tf.ones((10,10))
    B = tf.ones((10,10)) * 0.5
    C = tf.reduce_sum(A*B).numpy()
    print(C)
    print(tf.__version__)