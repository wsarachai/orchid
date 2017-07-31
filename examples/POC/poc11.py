import tensorflow as tf
import poc11_env

FLAGS = tf.app.flags.FLAGS


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


def MLP(_x):
    with tf.name_scope('hidden_1'):
        with tf.name_scope('weights'):
            W_fc1 = weight_variable([FLAGS.image_buff_size, poc11_env.HIDDEN_NEURON])
            variable_summaries(W_fc1)

        with tf.name_scope('biases'):
            b_fc1 = bias_variable([poc11_env.HIDDEN_NEURON])
            variable_summaries(b_fc1)

        with tf.name_scope('Wx_plus_b'):
            hidden_layer1 = tf.nn.relu(tf.add(tf.matmul(_x, W_fc1), b_fc1))
            variable_summaries(hidden_layer1)

    with tf.name_scope('hidden_2'):
        with tf.name_scope('weights'):
            W_fc2 = weight_variable([poc11_env.HIDDEN_NEURON, poc11_env.HIDDEN_NEURON])
            variable_summaries(W_fc2)

        with tf.name_scope('biases'):
            b_fc2 = bias_variable([poc11_env.HIDDEN_NEURON])
            variable_summaries(b_fc2)

        with tf.name_scope('Wx_plus_b'):
            hidden_layer2 = tf.nn.relu(tf.matmul(hidden_layer1, W_fc2) + b_fc2)
            variable_summaries(hidden_layer2)

    with tf.name_scope('output'):
        W_out = weight_variable([poc11_env.HIDDEN_NEURON, poc11_env.CLASS_NUM])
        b_out = bias_variable([poc11_env.CLASS_NUM])

        output_layer = tf.matmul(hidden_layer2, W_out) + b_out

    return output_layer
