import os
import re
from datetime import datetime
import time
import numpy as np
import tensorflow as tf


IMAGE_SIZE = 224

NUM_CLASSES = 11
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 880
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 220

TOWER_NAME = 'tower'

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/Users/sarachaii/Desktop/trains/orchid11_simple_logs/',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('data_dir', '/Users/sarachaii/Desktop/trains/orchid11_data/',
                           """Directory where to write get dataset """)


def _activation_summary(x):
  # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
  # session. This helps the clarity of presentation on tensorboard.
  tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
  tf.summary.histogram(tensor_name + '/activations', x)
  tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def dense_to_one_hot(labels_dense, num_classes):
  """Convert class labels from scalars to one-hot vectors."""
  num_labels = labels_dense.shape[0]
  index_offset = np.arange(num_labels) * num_classes
  labels_one_hot = np.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot


def read_and_decode(filename_queue):
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)
  features = tf.parse_single_example(
  serialized_example,
  # Defaults are not specified since both keys are required.
  features={
    'height': tf.FixedLenFeature([], tf.int64),
    'width': tf.FixedLenFeature([], tf.int64),
    'depth': tf.FixedLenFeature([], tf.int64),
    'label': tf.FixedLenFeature([], tf.int64),
    'image_raw': tf.FixedLenFeature([], tf.string)
  })
  image = tf.decode_raw(features['image_raw'], tf.uint8)
  label = tf.cast(features['label'], tf.int32)
  height = tf.cast(features['height'], tf.int32)
  width = tf.cast(features['width'], tf.int32)
  depth = tf.cast(features['depth'], tf.int32)
  return image, label, height, width, depth


def read_orchid11(filename_queue):

  class ORCHID11Record(object):
    pass

  result = ORCHID11Record()

  result.height = IMAGE_SIZE
  result.width = IMAGE_SIZE
  result.depth = 3

  image, label, height, width, depth = read_and_decode(filename_queue=filename_queue)

  image = tf.reshape(image, [result.height, result.width, result.depth])

  # Convert from [0, 255] -> [-0.5, 0.5] floats.
  image = tf.cast(image, tf.float32)
  image = tf.cast(image, tf.float32) * (1. / 255) - 0.5

  result.uint8image = image

  # Convert label from a scalar uint8 tensor to an int32 scalar.
  result.label = tf.cast(label, tf.int32)

  return result


def _generate_image_and_label_batch(image, label, min_queue_examples, batch_size, shuffle):
  num_preprocess_threads = 16
  if shuffle:
    images, label_batch = tf.train.shuffle_batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size,
        min_after_dequeue=min_queue_examples)
  else:
    images, label_batch = tf.train.batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size)

  # Display the training images in the visualizer.
  tf.summary.image('images', images)

  return images, tf.reshape(label_batch, [batch_size])


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


def deepnn(x):
  # conv1
  with tf.variable_scope('conv1') as scope:
    # First convolutional layer - maps one grayscale image to 32 feature maps.
    W_conv1 = weight_variable([5, 5, 3, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
    _activation_summary(h_conv1)

  # Pooling layer - downsamples by 2X.
  h_pool1 = max_pool_2x2(h_conv1)

  # conv2
  with tf.variable_scope('conv2') as scope:
    # Second convolutional layer -- maps 32 feature maps to 64.
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    _activation_summary(h_conv2)

  # Second pooling layer.
  h_pool2 = max_pool_2x2(h_conv2)

  # fully3
  with tf.variable_scope('fully3') as scope:
    # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
    # is down to 7x7x64 feature maps -- maps this to 1024 features.
    W_fc1 = weight_variable([56 * 56 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 56*56*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    _activation_summary(h_fc1)

  # Dropout - controls the complexity of the model, prevents co-adaptation of
  # features.
  keep_prob = tf.placeholder(tf.float32)
  h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  # fully4
  with tf.variable_scope('fully4') as scope:
    # Map the 1024 features to 10 classes, one for each digit
    W_fc2 = weight_variable([1024, 11])
    b_fc2 = bias_variable([11])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    _activation_summary(y_conv)

  return y_conv, keep_prob


def inputs(eval_data, data_dir, batch_size):
  if not eval_data:
    filenames = [os.path.join(data_dir, 'orchid11-images-train-%d.tfrecords' % i)
                 for i in xrange(0, 44)]
    num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
  else:
    filenames = [os.path.join(data_dir, 'orchid11-images-test-%d.tfrecords' % i)
                 for i in xrange(0, 11)]
    num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

  for f in filenames:
    if not tf.gfile.Exists(f):
      raise ValueError('Failed to find file: ' + f)

  # Create a queue that produces the filenames to read.
  filename_queue = tf.train.string_input_producer(filenames)

  # Read examples from files in the filename queue.
  read_input = read_orchid11(filename_queue)
  reshaped_image = tf.cast(read_input.uint8image, tf.float32)

  height = IMAGE_SIZE
  width = IMAGE_SIZE

  # Subtract off the mean and divide by the variance of the pixels.
  float_image = tf.image.per_image_standardization(reshaped_image)

  # Set the shapes of tensors.
  float_image.set_shape([height, width, 3])

  # Ensure that the random shuffling has good mixing properties.
  min_fraction_of_examples_in_queue = 0.4
  min_queue_examples = int(num_examples_per_epoch *
                           min_fraction_of_examples_in_queue)

  # Generate a batch of images and labels by building up a queue of examples.
  return _generate_image_and_label_batch(float_image, read_input.label,
                                         min_queue_examples, batch_size,
                                         shuffle=True)


def train():
  with tf.Graph().as_default():
    global_step = tf.contrib.framework.get_or_create_global_step()

    with tf.device('/cpu:0'):
      images, labels = inputs(None, FLAGS.data_dir, 20)

    # Create the model
    x = tf.placeholder(tf.float32, [None, 224, 224, 3])

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 11])

    # Build the graph for the deep net
    y_conv, keep_prob = deepnn(x)

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


    class _LoggerHook(tf.train.SessionRunHook):
      """Logs loss and runtime."""

      def begin(self):
        self._step = -1
        self._start_time = time.time()

      def before_run(self, run_context):
        self._step += 1
        return tf.train.SessionRunArgs(loss)  # Asks for loss value.

      def after_run(self, run_context, run_values):
        if self._step % FLAGS.log_frequency == 0:
          current_time = time.time()
          duration = current_time - self._start_time
          self._start_time = current_time

          loss_value = run_values.results
          examples_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
          sec_per_batch = float(duration / FLAGS.log_frequency)

          format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f sec/batch)')
          print (format_str % (datetime.now(), self._step, loss_value, examples_per_sec, sec_per_batch))

    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(coord=coord)

      for i in range(200):
        img = sess.run(images)
        lbl = sess.run(labels)
        lbl = dense_to_one_hot(lbl, 11)
        if i % 5 == 0:
          train_accuracy = accuracy.eval(feed_dict={x: img, y_: lbl, keep_prob: 1.0})
          print('step %d, training accuracy %g' % (i, train_accuracy))
        train_step.run(feed_dict={x: img, y_: lbl, keep_prob: 0.5})

      #print('test accuracy %g' % accuracy.eval(feed_dict={
      #    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

      # stop our queue threads and properly close the session
      coord.request_stop()
      coord.join(threads)
      sess.close()


def main(argv=None):
  train()


if __name__ == '__main__':
  tf.app.run()