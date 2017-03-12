import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from network_common import *

def get_inference(x_image, FLAGS, reuse, drop_prob = 1.0, is_train = True):

    with tf.variable_scope('discriminator', reuse) as scope:
        if reuse:
            scope.reuse_variables()

        channel_size = [64, 128, 256, 512, 512, 512, 512, 512, 512, 512]

        # convolution

        # 256x256
        with tf.variable_scope('conv1', reuse) as scope:
            W_conv = weight_variable([ 4, 4, 6, channel_size[0] ])
            b_conv = bias_variable([ channel_size[0] ])
            conv = conv2d(x_image, W_conv)
            bias = tf.nn.bias_add(conv, b_conv)
            conv1 = lrelu(bias)
        
        # 128x128
        with tf.variable_scope('conv2', reuse) as scope:
            W_conv = weight_variable([ 4, 4, channel_size[0], channel_size[1] ])
            b_conv = bias_variable([ channel_size[1] ])
            conv = conv2d(conv1, W_conv)
            bias = tf.nn.bias_add(conv, b_conv)
            bias = batch_norm(bias, is_train)
            conv2 = lrelu(bias)

        # 64x64
        with tf.variable_scope('conv3' , reuse) as scope:
            W_conv = weight_variable([4, 4, channel_size[1], channel_size[2] ])
            b_conv = bias_variable([ channel_size[2] ])
            conv = conv2d(conv2, W_conv)
            bias = tf.nn.bias_add(conv, b_conv)
            bias = batch_norm(bias, is_train)
            conv3 = lrelu(bias)

        # 32x32
        with tf.variable_scope('conv4', reuse) as scope:
            W_conv = weight_variable([ 4, 4, channel_size[2], channel_size[3] ])
            b_conv = bias_variable([ channel_size[3] ])
            conv = conv2d(conv3, W_conv)
            bias = tf.nn.bias_add(conv, b_conv)
            bias = batch_norm(bias, is_train)
            conv4 = lrelu(bias)

        # 16x16
        with tf.variable_scope('conv5', reuse) as scope:
            W_conv = weight_variable([ 4, 4, channel_size[3], channel_size[4] ])
            b_conv = bias_variable([ channel_size[4] ])
            conv = conv2d(conv4, W_conv)
            bias = tf.nn.bias_add(conv, b_conv)
            bias = batch_norm(bias, is_train)
            conv5 = lrelu(bias)

        # 8x8
        with tf.variable_scope('conv6', reuse) as scope:
            W_conv = weight_variable([ 4, 4, channel_size[4], channel_size[5] ])
            b_conv = bias_variable([ channel_size[5] ])
            conv = conv2d(conv5, W_conv)
            bias = tf.nn.bias_add(conv, b_conv)
            bias = batch_norm(bias, is_train)
            conv6 = lrelu(bias)

        # 4x4
        with tf.variable_scope('conv7', reuse) as scope:
            W_conv = weight_variable([ 4, 4, channel_size[5], channel_size[6] ])
            b_conv = bias_variable([ channel_size[6] ])
            conv = conv2d(conv6, W_conv)
            bias = tf.nn.bias_add(conv, b_conv)
            bias = batch_norm(bias, is_train)
            conv7 = lrelu(bias)

        # 2x2 -> 1x1
        with tf.variable_scope('conv8', reuse) as scope:
            W_conv = weight_variable([ 4, 4, channel_size[6], channel_size[7] ])
            b_conv = bias_variable([ channel_size[7] ])
            conv = conv2d(conv7, W_conv)
            bias = tf.nn.bias_add(conv, b_conv)
            bias = batch_norm(bias, is_train)
            conv8 = lrelu(bias)

        # 1x1 -> output
        with tf.variable_scope('output', reuse) as scope:
            W_conv = weight_variable([ 4, 4, channel_size[6], 1 ])
            b_conv = bias_variable([ 1 ])
            conv = conv2d(conv7, W_conv)
            bias = tf.nn.bias_add(conv, b_conv)
            #output = tf.nn.sigmoid(bias)
            output = tf.squeeze(bias, [1, 2, 3])

        return output



def get_softmax_loss(logits, labels, FLAGS):
    logits = tf.reshape(logits, [FLAGS.batch_size, -1])
    labels = tf.reshape(labels, [FLAGS.batch_size, -1])

    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
    labels=labels, logits=logits, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    return cross_entropy_mean
