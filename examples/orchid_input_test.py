import os
import gzip
import pickle
import numpy as np
import tensorflow as tf


IMAGE_SIZE = 224

NUM_CLASSES = 11
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 880
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 220

DATA_DIR = '/Users/sarachaii/Desktop/trains/orchid11_data/'


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
  #image = tf.cast(image, tf.float32)
  #image = tf.cast(image, tf.float32) * (1. / 255) - 0.5

  result.uint8image = image

  # Convert label from a scalar uint8 tensor to an int32 scalar.
  result.label = tf.cast(label, tf.int32)

  return result


def _generate_image_and_label_batch(image, label, min_queue_examples, batch_size, shuffle):
  """Construct a queued batch of images and labels.

  Args:
    image: 3-D Tensor of [height, width, 3] of type.float32.
    label: 1-D Tensor of type.int32
    min_queue_examples: int32, minimum number of samples to retain
      in the queue that provides of batches of examples.
    batch_size: Number of images per batch.
    shuffle: boolean indicating whether to use a shuffling queue.

  Returns:
    images: Images. 4D tensor of [batch_size, height, width, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  # Create a queue that shuffles the examples, and then
  # read 'batch_size' images + labels from the example queue.
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


def main(argv=None):
  images, labels = inputs(1, DATA_DIR, 20)

  with tf.Session() as sess:
    # initialize the variables
    sess.run(tf.initialize_all_variables())

    # initialize the queue threads to start to shovel data
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    print "from the train set:"
    for i in range(44):
        img =  sess.run(images)
        print (img.shape)

    print "from the test set:"
    for i in range(44):
        lbl = sess.run(labels)
        lbl = dense_to_one_hot(lbl, 11)
        print (lbl.shape)
        print lbl

    # stop our queue threads and properly close the session
    coord.request_stop()
    coord.join(threads)
    sess.close()


if __name__ == '__main__':
  tf.app.run()