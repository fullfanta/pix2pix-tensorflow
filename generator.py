import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from network_common import *

def get_inference(x_image, FLAGS, reuse, drop_prob = 0.5, is_train = True):

    with tf.variable_scope('generator', reuse) as scope:
        if reuse:
            scope.reuse_variables()


        channel_size = [64, 128, 256, 512, 512, 512, 512, 512, 512, 512]

        # convolution

        # 256x256
        with tf.variable_scope('conv1', reuse) as scope:
            W_conv = weight_variable([ 4, 4, 3, channel_size[0] ])
            b_conv = bias_variable([ channel_size[0] ])
            conv = conv2d(x_image, W_conv)
            conv1 = tf.nn.bias_add(conv, b_conv)

        # 128x128
        with tf.variable_scope('conv2', reuse) as scope:
            re = lrelu(conv1)
            W_conv = weight_variable([ 4, 4, channel_size[0], channel_size[1] ])
            b_conv = bias_variable([ channel_size[1] ])
            conv = conv2d(re, W_conv)
            bias = tf.nn.bias_add(conv, b_conv)
            conv2 = batch_norm(bias, is_train)

        # 64x64
        with tf.variable_scope('conv3' , reuse) as scope:
            re = lrelu(conv2)
            W_conv = weight_variable([4, 4, channel_size[1], channel_size[2] ])
            b_conv = bias_variable([ channel_size[2] ])
            conv = conv2d(re, W_conv)
            bias = tf.nn.bias_add(conv, b_conv)
            conv3 = batch_norm(bias, is_train)

        # 32x32
        with tf.variable_scope('conv4', reuse) as scope:
            re = lrelu(conv3)
            W_conv = weight_variable([ 4, 4, channel_size[2], channel_size[3] ])
            b_conv = bias_variable([ channel_size[3] ])
            conv = conv2d(re, W_conv)
            bias = tf.nn.bias_add(conv, b_conv)
            conv4 = batch_norm(bias, is_train)

        # 16x16
        with tf.variable_scope('conv5', reuse) as scope:
            re = lrelu(conv4)
            W_conv = weight_variable([ 4, 4, channel_size[3], channel_size[4] ])
            b_conv = bias_variable([ channel_size[4] ])
            conv = conv2d(re, W_conv)
            bias = tf.nn.bias_add(conv, b_conv)
            conv5 = batch_norm(bias, is_train)

        # 8x8
        with tf.variable_scope('conv6', reuse) as scope:
            re = lrelu(conv5)
            W_conv = weight_variable([ 4, 4, channel_size[4], channel_size[5] ])
            b_conv = bias_variable([ channel_size[5] ])
            conv = conv2d(re, W_conv)
            bias = tf.nn.bias_add(conv, b_conv)
            conv6 = batch_norm(bias, is_train)

        # 4x4
        with tf.variable_scope('conv7', reuse) as scope:
            re = lrelu(conv6)
            W_conv = weight_variable([ 4, 4, channel_size[5], channel_size[6] ])
            b_conv = bias_variable([ channel_size[6] ])
            conv = conv2d(re, W_conv)
            bias = tf.nn.bias_add(conv, b_conv)
            conv7 = batch_norm(bias, is_train)

        # 2x2 -> 1x1
        with tf.variable_scope('conv8', reuse) as scope:
            re = lrelu(conv7)
            W_conv = weight_variable([ 4, 4, channel_size[6], channel_size[7] ])
            b_conv = bias_variable([ channel_size[7] ])
            conv = conv2d(re, W_conv)
            conv8 = tf.nn.bias_add(conv, b_conv)


        # deconvolution
        # 1x1 -> 2x2
        with tf.variable_scope('deconv8', reuse) as scope:
            re = tf.nn.relu(conv8)
            W_conv = weight_variable([4, 4, channel_size[6], channel_size[7] ])
            b_conv = bias_variable([ channel_size[6] ])
            deconv = deconv2d(re, W_conv, conv7.get_shape())
            bias = tf.nn.bias_add(deconv, b_conv)
            bias = batch_norm(bias, is_train)
            drop = tf.nn.dropout(bias, drop_prob)
            deconv8 = tf.concat([drop, conv7], 3) 

        # 2x2 -> 4x4
        with tf.variable_scope('deconv7', reuse) as scope:
            deconv8 = tf.nn.relu(deconv8, name=scope.name)
            W_conv = weight_variable([4, 4, channel_size[5], channel_size[6] * 2])
            b_conv = bias_variable([ channel_size[5] ])
            deconv = deconv2d(deconv8, W_conv, conv6.get_shape())
            bias = tf.nn.bias_add(deconv, b_conv)
            bias = batch_norm(bias, is_train)
            drop = tf.nn.dropout(bias, drop_prob)
            deconv7 = tf.concat([drop, conv6], 3) 
            

        # 4x4 -> 8x8
        with tf.variable_scope('deconv6', reuse) as scope:
            deconv7 = tf.nn.relu(deconv7, name=scope.name)
            W_conv = weight_variable([4, 4, channel_size[4], channel_size[5] * 2 ])
            b_conv = bias_variable([ channel_size[4] ])
            deconv = deconv2d(deconv7, W_conv, conv5.get_shape())
            bias = tf.nn.bias_add(deconv, b_conv)
            bias = batch_norm(bias, is_train)
            drop = tf.nn.dropout(bias, drop_prob)
            deconv6 = tf.concat([drop, conv5], 3) 
            
        
        # 8x8 -> 16x16
        with tf.variable_scope('deconv5', reuse) as scope:
            deconv6 = tf.nn.relu(deconv6, name=scope.name)
            W_conv = weight_variable([4, 4, channel_size[3], channel_size[4] * 2 ])
            b_conv = bias_variable([ channel_size[3] ])
            deconv = deconv2d(deconv6, W_conv, conv4.get_shape())
            bias = tf.nn.bias_add(deconv, b_conv)
            bias = batch_norm(bias, is_train)
            deconv5 = tf.concat([bias, conv4], 3)
             
        
        # 16x16 -> 32x32
        with tf.variable_scope('deconv4', reuse) as scope:
            deconv5 = tf.nn.relu(deconv5, name=scope.name)
            W_conv = weight_variable([4, 4, channel_size[2], channel_size[3] * 2])
            b_conv = bias_variable([ channel_size[2] ])
            deconv = deconv2d(deconv5, W_conv, conv3.get_shape())
            bias = tf.nn.bias_add(deconv, b_conv)
            bias = batch_norm(bias, is_train)
            deconv4 = tf.concat([bias, conv3], 3)
            
        # 32x32 -> 64x64
        with tf.variable_scope('deconv3', reuse) as scope:
            deconv4 = tf.nn.relu(deconv4, name=scope.name)
            W_conv = weight_variable([4, 4, channel_size[1], channel_size[2] * 2])
            b_conv = bias_variable([channel_size[1]])
            deconv = deconv2d(deconv4, W_conv, conv2.get_shape())
            bias = tf.nn.bias_add(deconv, b_conv)
            bias = batch_norm(bias, is_train)
            deconv3 = tf.concat([bias, conv2], 3) 
        
        # 64x64 -> 128x128
        with tf.variable_scope('deconv2', reuse) as scope:
            deconv3 = tf.nn.relu(deconv3, name=scope.name)
            W_conv = weight_variable([4, 4, channel_size[0], channel_size[1] * 2])
            b_conv = bias_variable([channel_size[0]])
            deconv = deconv2d(deconv3, W_conv, conv1.get_shape())
            bias = tf.nn.bias_add(deconv, b_conv)
            bias = batch_norm(bias, is_train)
            deconv2 = tf.concat([bias, conv1], 3) 
            
        # 128x128 -> 256x256
        with tf.variable_scope('deconv1', reuse) as scope:
            deconv2 = tf.nn.relu(deconv2, name=scope.name)
            W_conv = weight_variable([4, 4, 3, channel_size[0] * 2])
            b_conv = bias_variable([3])
            deconv = deconv2d(deconv2, W_conv, x_image.get_shape())
            bias = tf.nn.bias_add(deconv, b_conv)
            output = tf.nn.tanh(bias, name=scope.name)

        return output



def get_l1_loss(preds, labels, FLAGS):
    abs_diff = tf.abs(preds - labels)
    return tf.reduce_mean(abs_diff)
    
