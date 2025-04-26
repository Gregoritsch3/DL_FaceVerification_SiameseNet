#Custom L1 Distance Layer module
import tensorflow as tf
from tensorflow.keras.layers import Layer
import numpy as np

class L2Dist(Layer):
    def __init__(self, **kwargs):
        super(L2Dist, self).__init__(**kwargs)

    def call(self, inputs):
        # Ensure inputs is a list/tuple of two tensors
        if not isinstance(inputs, (list, tuple)) or len(inputs) != 2:
            raise ValueError('L2Dist layer expects a list or tuple of two input tensors.')

        input_embedding, validation_embedding = inputs

        # Calculate the sum of squared differences
        sum_squared = tf.reduce_sum(tf.square(input_embedding - validation_embedding), axis=1, keepdims=True)

        # Calculate the Euclidean distance (square root)
        # Add epsilon to prevent taking sqrt of zero or negative numbers due to numerical instability
        distance = tf.sqrt(tf.maximum(sum_squared, tf.keras.backend.epsilon()))

        return distance

    def get_config(self):
        # Add this method if you want to save/load models containing this custom layer
        # without needing to pass it in custom_objects every time (using tf.keras.saving)
        config = super(L2Dist, self).get_config()
        return config