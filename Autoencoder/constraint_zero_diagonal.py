import tensorflow as tf
import numpy as np


class DiagonalZeroConstraint(tf.keras.constraints.Constraint):
    def __init__(self):
        pass


    def __call__(self, w):
        diag = tf.linalg.diag(tf.linalg.diag_part(w))
        return w - diag
