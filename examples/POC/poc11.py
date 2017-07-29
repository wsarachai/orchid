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


def MLP(_x):
    W_fc1 = weight_variable([FLAGS.image_buff_size, poc11_env.HIDDEN_NEURON])
    b_fc1 = bias_variable([poc11_env.HIDDEN_NEURON])

    hidden_layer1 = tf.add(tf.matmul(_x, W_fc1), b_fc1)
    hidden_layer1 = tf.nn.relu(hidden_layer1)

    W_fc2 = weight_variable([poc11_env.HIDDEN_NEURON, poc11_env.HIDDEN_NEURON])
    b_fc2 = bias_variable([poc11_env.HIDDEN_NEURON])

    hidden_layer2 = tf.matmul(hidden_layer1, W_fc2) + b_fc2
    hidden_layer2 = tf.nn.relu(hidden_layer2)

    W_out = weight_variable([poc11_env.HIDDEN_NEURON, poc11_env.CLASS_NUM])
    b_out = bias_variable([poc11_env.CLASS_NUM])

    output_layer = tf.matmul(hidden_layer2, W_out) + b_out

    return output_layer
