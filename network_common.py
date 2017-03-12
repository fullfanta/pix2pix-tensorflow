import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes

def weight_variable(shape):
    weights = tf.get_variable("weights", shape, initializer=tf.random_normal_initializer(mean = 0.0, stddev=0.02))
    #print weights.name
    return weights

def bias_variable(shape):
    biases = tf.get_variable("biases", shape, initializer=tf.constant_initializer(0.0))
    return biases

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding="SAME")

def deconv2d(x, W, output_shape):
    return tf.nn.conv2d_transpose(x, W, output_shape, strides=[1, 2, 2, 1], padding="SAME")

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")

def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak*x)

def batch_norm(x, epsilon=1e-5, momentum = 0.9, is_training = True):
    return tf.contrib.layers.batch_norm(x,
                    decay=momentum, 
                    updates_collections=None,
                    epsilon=epsilon,
                    scale=True,
                    is_training=is_training,
                    scope="batch_norm")
