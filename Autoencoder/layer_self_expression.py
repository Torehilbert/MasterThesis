import tensorflow as tf
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from constraint_zero_diagonal import DiagonalZeroConstraint


class SelfExpression(tf.keras.layers.Layer):
    def __init__(self, num_channels, constraint_type="hardset", l1=None, l2=None, name='selfexp'):
        super(SelfExpression, self).__init__(name=name)
        self.num_channels = num_channels

        # Regularizer
        if l1 is None and l2 is None:
            reg = None
        elif l2 is None:
            reg = tf.keras.regularizers.l1(l1)
        elif l1 is None:
            reg = tf.keras.regularizers.l2(l2)
        else:
            reg = tf.keras.regularizers.l1_l2(l1=l1, l2=l2)

        # Constraint
        if(constraint_type=="hardset"):
            con = DiagonalZeroConstraint()
        else:
            con = None
        
        # Kernel
        self.kernel = self.add_weight(
            "kernel", 
            shape=(self.num_channels, self.num_channels), 
            regularizer=reg, 
            constraint=con
        )


    def call(self, U):
        out = []
        for i in range(self.num_channels):
            out.append(tf.reduce_sum(self.kernel[i] * U, axis=4, keepdims=True))
        Ure = tf.concat(out, axis=4)

        return Ure       