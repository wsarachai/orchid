import os
import numpy as np
import pandas as pd
import tensorflow as tf
#from matplotlib import pyplot as plt

IMAGE_CHANNEL = 3
IMAGE_SIZE = 32
IMAGE_BUFF_SIZE = IMAGE_SIZE*IMAGE_SIZE*IMAGE_CHANNEL
LEARNING_RATE = 0.001

DATA_TYPE = 'ground-truth'
#DATA_TYPE = 'general'

#tf.logging.set_verbosity(tf.logging.INFO)

ROOT_DIR = '/Users/sarachaii/Desktop/trains/'
DATASET_DIR = os.path.join(ROOT_DIR, 'dataset', DATA_TYPE)
SUMMARIES_DIR = os.path.join(ROOT_DIR, 'logs', DATA_TYPE, 'summaries32')

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('root_dir', ROOT_DIR, 'Root directory.')
tf.app.flags.DEFINE_string('data_dir', DATASET_DIR, 'Data directory.')
tf.app.flags.DEFINE_string('summaries_dir', SUMMARIES_DIR, 'Summaries directory.')
tf.app.flags.DEFINE_integer('epochs', 20000, 'number of epochs')
tf.app.flags.DEFINE_integer('batch_size', 128, 'Batch size.')
tf.app.flags.DEFINE_float('dropout', 0.5, 'Dropout rate.')

# To stop potential randomness
rng = np.random.RandomState(128)

# check for existence
os.path.exists(FLAGS.root_dir)
os.path.exists(FLAGS.data_dir)

train = pd.read_csv(os.path.join(FLAGS.data_dir, 'train', 'train.csv'))
test = pd.read_csv(os.path.join(FLAGS.data_dir, 'test', 'test.csv'))

train.head()
test.head()


def decode_image(var):
    with tf.Session() as sess:
        temp = []

        graph = tf.Graph()
        with graph.as_default():
            file_name = tf.placeholder(dtype=tf.string)
            file = tf.read_file(file_name)
            image = tf.image.decode_jpeg(file)
            image = tf.cast(image, tf.float32)
            image.set_shape((IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNEL))

        with tf.Session(graph=graph) as session:
            tf.global_variables_initializer().run()
            for img_name in eval(var).filename:
                image_path = os.path.join(FLAGS.data_dir, var, 'images' + str(IMAGE_SIZE), img_name)
                img = session.run(image, feed_dict={file_name: image_path})
                temp.append(img)
            session.close()

    return np.stack(temp)


test_x = decode_image('test')
train_x = decode_image('train')

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
    _dataset = eval(dataset_name + '_x')

    dataset_length = _dataset.shape[0]

    """Create batch with random samples and return appropriate format"""
    batch_mask = rng.choice(dataset_length, batch_size)

    batch_x = _dataset[[batch_mask]].reshape(-1, IMAGE_BUFF_SIZE)
    batch_x = preproc(batch_x)

    #if dataset_name == 'train':
    batch_y = eval(dataset_name).ix[batch_mask, 'label'].values
    batch_y = dense_to_one_hot(batch_y)

    return batch_x, batch_y


def train_model():
    # define placeholders
    with tf.name_scope('input'):
        _x = tf.placeholder(tf.float32, [None, IMAGE_BUFF_SIZE])
        _y = tf.placeholder(tf.float32, [None, 11])

    with tf.name_scope('input_reshape'):
        x_image = tf.reshape(_x, [-1, IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNEL])
        tf.summary.image('input', x_image, 11)

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
        optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy)
        #optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cross_entropy)

    # find predictions on val set
    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            pred_temp = tf.equal(tf.argmax(output_layer, 1), tf.argmax(_y, 1))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(pred_temp, tf.float32))

    tf.summary.scalar('accuracy', accuracy)

    def feed_dict(train):
        if train: #or FLAGS.fake_data:
            batch_x, batch_y = batch_creator(FLAGS.batch_size, 'train')
            k = FLAGS.dropout
        else:
            batch_x, batch_y = batch_creator(FLAGS.batch_size, 'test')
            k = 1.0
        return {_x: batch_x, _y: batch_y, keep_prob: k}

    merged = tf.summary.merge_all()

    return merged, accuracy, optimizer, feed_dict


def main():
    sess = tf.InteractiveSession()

    merged, accuracy, optimizer, feed_dict = train_model()

    train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train', sess.graph)
    test_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/test')

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    tf.global_variables_initializer().run()

    for step in range(FLAGS.epochs):
        if step % 40 == 0:  # Record summaries and test-set accuracy
            summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
            test_writer.add_summary(summary, step)
            print('Accuracy at step %s: %s' % (step, acc))
        else:
            #total_batch = int(train.shape[0] / FLAGS.batch_size)
            #for i in range(total_batch):
            if step % 1000 == 9:  # Record execution stats
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                summary, _ = sess.run([merged, optimizer],
                                      feed_dict=feed_dict(True),
                                      options=run_options,
                                      run_metadata=run_metadata)
                train_writer.add_run_metadata(run_metadata, 'step%03d' % step)
                train_writer.add_summary(summary, step)
                print('Adding run metadata for', step)
            else:  # Record a summary
                summary, _ = sess.run([merged, optimizer], feed_dict=feed_dict(True))
                train_writer.add_summary(summary, step)

    print ("\nTraining complete!")

    save_path = saver.save(sess, os.path.join(FLAGS.summaries_dir, "model.ckpt"))
    print("Model saved in file: %s" % save_path)

    train_writer.close()
    test_writer.close()


if __name__ == '__main__':
    main()
