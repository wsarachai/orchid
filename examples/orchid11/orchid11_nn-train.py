import os
import numpy as np
import pandas as pd
from scipy.misc import imread
import tensorflow as tf
from matplotlib import pyplot as plt

tf.logging.set_verbosity(tf.logging.INFO)

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('root_dir', '/Users/sarachaii/Desktop/trains/',
                           'Root directory.')
tf.app.flags.DEFINE_string('data_dir', '/Users/sarachaii/Desktop/trains/data/',
                           'Data directory.')
tf.app.flags.DEFINE_string('summaries_dir', '/Users/sarachaii/Desktop/trains/summaries_nn/',
                           'Summaries directory.')
tf.app.flags.DEFINE_integer('epochs', 1000,
                            'number of epochs')
tf.app.flags.DEFINE_integer('batch_size', 128,
                            'Batch siez.')
tf.app.flags.DEFINE_float('--learning_rate', 0.01,
                          'Initial learning rate')
tf.app.flags.DEFINE_float('dropout', 0.5,
                          'Dropout rate.')

IMAGE_CHANNEL = 3
IMAGE_SIZE = 32
IMAGE_BUFF_SIZE = IMAGE_SIZE*IMAGE_SIZE*IMAGE_CHANNEL
LEARNING_RATE = 0.001
NUMS_CLASSES = 11

# To stop potential randomness
rng = np.random.RandomState(128)

# check for existence
os.path.exists(FLAGS.root_dir)
os.path.exists(FLAGS.data_dir)

train = pd.read_csv(os.path.join(FLAGS.data_dir, 'train', 'train.csv'))
test = pd.read_csv(os.path.join(FLAGS.data_dir, 'test', 'test.csv'))

train.head()
test.head()

temp = []
for img_name in train.filename:
    image_path = os.path.join(FLAGS.data_dir, 'train', 'images' + str(IMAGE_SIZE), img_name)
    img = imread(image_path, flatten=False)
    img = img.astype('float32')
    #print (img.shape)
    temp.append(img)

train_x = np.stack(temp)

temp = []
for img_name in test.filename:
    image_path = os.path.join(FLAGS.data_dir, 'test', 'images' + str(IMAGE_SIZE), img_name)
    img = imread(image_path, flatten=False)
    img = img.astype('float32')
    temp.append(img)

test_x = np.stack(temp)

def dense_to_one_hot(labels_dense, num_classes=11):
    """Convert class labels from scalars to one-hot vectors"""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1

    return labels_one_hot


def preproc(unclean_batch_x):
    """Convert values to range 0-1"""
    temp_batch = unclean_batch_x / unclean_batch_x.max()

    return temp_batch


def batch_creator(batch_size, dataset_name):
    _train = eval(dataset_name + '_x')

    dataset_length = _train.shape[0]

    """Create batch with random samples and return appropriate format"""
    batch_mask = rng.choice(dataset_length, batch_size)

    batch_x = _train[[batch_mask]].reshape(-1, IMAGE_BUFF_SIZE)
    batch_x = preproc(batch_x)

    #if dataset_name == 'train':
    batch_y = eval(dataset_name).ix[batch_mask, 'label'].values
    batch_y = dense_to_one_hot(batch_y, NUMS_CLASSES)

    return batch_x, batch_y


def train_model():

    sess = tf.InteractiveSession()

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

    def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
        with tf.name_scope(layer_name):
            # This Variable will hold the state of the weights for the layer
            with tf.name_scope('weights'):
                weights = weight_variable([input_dim, output_dim])
                variable_summaries(weights)
            with tf.name_scope('biases'):
                biases = bias_variable([output_dim])
                variable_summaries(biases)
            with tf.name_scope('Wx_plus_b'):
                preactivate = tf.matmul(input_tensor, weights) + biases
                tf.summary.histogram('pre_activations', preactivate)
            activations = act(preactivate, name='activation')
            tf.summary.histogram('activations', activations)
            return activations

    # define placeholders
    with tf.name_scope('input'):
        _x = tf.placeholder(tf.float32, [None, IMAGE_BUFF_SIZE])
        _y = tf.placeholder(tf.float32, [None, NUMS_CLASSES])

    with tf.name_scope('input_reshape'):
        x_image = tf.reshape(_x, [-1, IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNEL])
        tf.summary.image('input', x_image, 10)

    hidden1 = nn_layer(_x, IMAGE_BUFF_SIZE, 500, 'layer1')

    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        tf.summary.scalar('dropout_keep_probability', keep_prob)
        dropped = tf.nn.dropout(hidden1, keep_prob)

    output_layer = nn_layer(dropped, 500, NUMS_CLASSES, 'layer2', act=tf.identity)

    with tf.name_scope('cross_entropy'):
        diff = tf.nn.softmax_cross_entropy_with_logits(labels=_y, logits=output_layer)
        with tf.name_scope('total'):
            cross_entropy = tf.reduce_mean(diff)

    tf.summary.scalar('cross_entropy', cross_entropy)

    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(cross_entropy)

    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(output_layer, 1), tf.argmax(_y, 1))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train', sess.graph)
    test_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/test')
    tf.global_variables_initializer().run()

    def feed_dict(train):
        if train: #or FLAGS.fake_data:
            batch_x, batch_y = batch_creator(FLAGS.batch_size, 'train')
            k = FLAGS.dropout
        else:
            batch_x, batch_y = batch_creator(FLAGS.batch_size, 'test')
            k = 1.0
        return {_x: batch_x, _y: batch_y, keep_prob: k}

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    for step in range(FLAGS.epochs):
        if step % 10 == 0:  # Record summaries and test-set accuracy
            summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
            test_writer.add_summary(summary, step)
            print('Accuracy at step %s: %s' % (step, acc))
        else:
            #total_batch = int(train.shape[0] / FLAGS.batch_size)
            #for i in range(total_batch):
            if step % 100 == 9:  # Record execution stats
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                summary, _ = sess.run([merged, train_step],
                                      feed_dict=feed_dict(True),
                                      options=run_options,
                                      run_metadata=run_metadata)
                train_writer.add_run_metadata(run_metadata, 'step%03d' % step)
                train_writer.add_summary(summary, step)
                print('Adding run metadata for', step)
            else:  # Record a summary
                summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
                train_writer.add_summary(summary, step)

    print ("\nTraining complete!")

    save_path = saver.save(sess, os.path.join(FLAGS.summaries_dir, "model.ckpt"))
    print("Model saved in file: %s" % save_path)

    train_writer.close()
    test_writer.close()


def main(_):
    train_model()

if __name__ == '__main__':
    tf.app.run()
