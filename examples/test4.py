import os
import numpy as np
import pandas as pd
from scipy.misc import imread
import tensorflow as tf
from matplotlib import pyplot as plt

tf.logging.set_verbosity(tf.logging.INFO)

IMAGE_CHANNEL = 3
IMAGE_SIZE = 32
IMAGE_BUFF_SIZE = IMAGE_SIZE*IMAGE_SIZE*IMAGE_CHANNEL

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('root_dir', '/Users/sarachaii/Desktop/trains/',
                           'Root directory.')
tf.app.flags.DEFINE_string('data_dir', '/Users/sarachaii/Desktop/trains/data/',
                           'Data directory.')
tf.app.flags.DEFINE_string('summaries_dir', '/Users/sarachaii/Desktop/trains/summaries' + str(IMAGE_SIZE) + '/',
                           'Summaries directory.')
tf.app.flags.DEFINE_integer('epochs', 100,
                            'number of epochs')
tf.app.flags.DEFINE_integer('batch_size', 128,
                            'Batch siez.')
tf.app.flags.DEFINE_float('dropout', 1.0,
                          'Dropout rate.')

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

def conv2d(x, W):
  """conv2d returns a 2d convolution layer with full stride."""
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
  """max_pool_2x2 downsamples a feature map by 2X."""
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


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
    batch_y = dense_to_one_hot(batch_y)

    return batch_x, batch_y


def train_model():

    sess = tf.InteractiveSession()

    # define placeholders
    with tf.name_scope('input'):
        _x = tf.placeholder(tf.float32, [None, IMAGE_BUFF_SIZE])
        _y = tf.placeholder(tf.float32, [None, 11])

    with tf.name_scope('input_reshape'):
        x_image = tf.reshape(_x, [-1, IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNEL])
        tf.summary.image('input', x_image, 10)

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
                W_conv1 = weight_variable([5, 5, IMAGE_CHANNEL, 32])
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

        reduce_from_maxpool = IMAGE_SIZE / 4
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
            W_fc2 = weight_variable([1024, 11])
            b_fc2 = bias_variable([11])

            output_layer = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

        return output_layer, keep_prob

    output_layer, keep_prob = deepnn(x_image)

    with tf.name_scope('total'):
        #output_layer = tf.nn.softmax(output_layer)
        #cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(output_layer), reduction_indices=[1]))
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output_layer, labels=_y))

    tf.summary.scalar('cross_entropy', cross_entropy)

    with tf.name_scope('train'):
        optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
        #optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cross_entropy)

    # find predictions on val set
    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            pred_temp = tf.equal(tf.argmax(output_layer, 1), tf.argmax(_y, 1))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(pred_temp, "float"))

    tf.summary.scalar('accuracy', accuracy)

    merged = tf.summary.merge_all()
    valid_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/validate')

    tf.global_variables_initializer().run()

    def feed_dict(train):
        if train:
            batch_x, batch_y = batch_creator(FLAGS.batch_size, 'train')
            k = FLAGS.dropout
        else:
            batch_x, batch_y = batch_creator(FLAGS.batch_size, 'test')
            k = 1.0
        return {_x: batch_x, _y: batch_y, keep_prob: k}

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    # Restore variables from disk.
    saver.restore(sess, os.path.join(FLAGS.summaries_dir, "model.ckpt"))
    print("Model restored.")

    summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
    valid_writer.add_summary(summary)
    print('Accuracy: %s' % acc)


    image_path = os.path.join(FLAGS.data_dir, 'test/images' + str(IMAGE_SIZE) + '/2_81.jpg')
    pd_img = imread(image_path, flatten=False)
    pd_img = pd_img.astype('float32')
    pd_img = pd_img.reshape(-1, IMAGE_BUFF_SIZE)
    pd_img = preproc(pd_img)

    perc = output_layer.eval({_x: pd_img, keep_prob: 1.0})

    perc_max = tf.nn.relu(perc)
    perc_max = perc_max.eval()

    perc_sum = tf.reduce_sum(perc_max, 1)
    perc_sum = perc_sum.eval()

    print (perc_max)
    print (perc_sum)

    perc_ans = perc_max / perc_sum * 100

    lb = 1
    for p in perc_ans[0]:
        print ("label {0}: {1:2.2f}%".format(lb, p))
        lb += 1

    #predict = tf.argmax(output_layer, 1)
    #pred = predict.eval({_x: pd_img, keep_prob: 1.0})

    #print (pred)


def main(_):
    train_model()

if __name__ == '__main__':
    tf.app.run()