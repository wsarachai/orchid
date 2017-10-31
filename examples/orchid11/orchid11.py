from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import orchid11_env

FLAGS = tf.app.flags.FLAGS


def conv2d(x, W):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    """max_pool_2x2 downsamples a feature map by 2X."""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def deepnn(x_image):
    with tf.name_scope('conv1_layer'):
        with tf.name_scope('weights'):
            W_conv1 = weight_variable([5, 5, orchid11_env.IMAGE_CHANNEL, 32])
            variable_summaries(W_conv1)

        with tf.name_scope('biases'):
            b_conv1 = bias_variable([32])
            variable_summaries(b_conv1)

        with tf.name_scope('Wx_plus_b'):
            h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
            variable_summaries(h_conv1)

    with tf.name_scope('max_pool'):
        h_pool1 = max_pool_2x2(h_conv1)
        variable_summaries(h_pool1)

    with tf.name_scope('conv2_layer'):
        with tf.name_scope('weights'):
            W_conv2 = weight_variable([5, 5, 32, 64])
            variable_summaries(W_conv2)

        with tf.name_scope('biases'):
            b_conv2 = bias_variable([64])
            variable_summaries(b_conv2)

        with tf.name_scope('Wx_plus_b'):
            h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
            variable_summaries(h_conv2)

    with tf.name_scope('max_pool'):
        h_pool2 = max_pool_2x2(h_conv2)
        variable_summaries(h_pool2)

    reduce_from_maxpool = orchid11_env.IMAGE_SIZE / 4
    reduce_buff = reduce_from_maxpool * reduce_from_maxpool * 64

    with tf.name_scope('fullyc_1'):
        W_fc1 = weight_variable([reduce_buff, 1024])
        b_fc1 = bias_variable([1024])

        h_pool2_flat = tf.reshape(h_pool2, [-1, reduce_buff])
        hidden_layer = tf.add(tf.matmul(h_pool2_flat, W_fc1), b_fc1)
        hidden_layer = tf.nn.relu(hidden_layer)

    with tf.name_scope('dropout') as scope:
        keep_prob = tf.placeholder(tf.float32)
        tf.summary.scalar('dropout_keep_probability', keep_prob)
        h_fc1_drop = tf.nn.dropout(hidden_layer, keep_prob)

    with tf.name_scope('fullyc_2'):
        W_fc2 = weight_variable([1024, FLAGS.classes_num])
        b_fc2 = bias_variable([FLAGS.classes_num])

        output_layer = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    return output_layer, keep_prob
