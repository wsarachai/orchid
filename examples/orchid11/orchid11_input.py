
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import pandas as pd
from scipy.misc import imread

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf


IMAGE_CHANNEL = 3
IMAGE_SIZE = 224
NUM_CLASSES = 11
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 880
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 220

def read_csv(data_dir):
    train = pd.read_csv(os.path.join(data_dir, 'train', 'train.csv'))
    test = pd.read_csv(os.path.join(data_dir, 'test', 'test.csv'))

    train.head()
    test.head()

    return  train, test

def read_and_decode(filename_queue):
    image_reader = tf.WholeFileReader()
    _, image_file = image_reader.read(filename_queue)
    image = tf.image.decode_jpeg(image_file)

    splitf = tf.string_split([image_file], '/')
    v = splitf.values[-1]
    splitf = tf.string_split([v], '_')
    v = splitf.values[0]
    label = tf.string_to_number(v, tf.int32)
    #label = tf.one_hot(v, NUM_CLASSES)

    return image, label, IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNEL


def read_orchid11(filename_queue):
  """Reads and parses examples from ORCHID11 data files."""

  class ORCHID11Record(object):
    pass
  result = ORCHID11Record()

  result.height = IMAGE_SIZE
  result.width = IMAGE_SIZE
  result.depth = 3

  image, label, height, width, depth = read_and_decode(filename_queue=filename_queue)

  image = tf.reshape(image, [result.height, result.width, result.depth])

  result.uint8image = image

  # Convert label from a scalar uint8 tensor to an int32 scalar.
  result.label = tf.cast(label, tf.int32)

  return result


def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size, shuffle):
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

  return images, label_batch


def distorted_inputs(data_dir, batch_size):
    train, test = read_csv(data_dir)

    for f in train.filename:
        image_path = os.path.join(data_dir, 'train', 'images' + str(IMAGE_SIZE), f)
        if not tf.gfile.Exists(image_path):
            raise ValueError('Failed to find file: ' + f)

    filenames = []
    for f in train.filename:
        filenames.append(os.path.join(data_dir, 'train', 'images224', f))

    # Create a queue that produces the filenames to read.
    filename_queue = tf.train.string_input_producer(filenames)
    #filename_queue = tf.train.string_input_producer(train.filename, num_epochs=None, shuffle=True, seed=None, shared_name=None, name=None)

    # Read examples from files in the filename queue.
    read_input = read_orchid11(filename_queue)
    reshaped_image = tf.cast(read_input.uint8image, tf.float32)

    height = IMAGE_SIZE
    width = IMAGE_SIZE

    # Image processing for training the network. Note the many random
    # distortions applied to the image.

    # Randomly crop a [height, width] section of the image.
    distorted_image = tf.random_crop(reshaped_image, [height, width, 3])

    # Randomly flip the image horizontally.
    distorted_image = tf.image.random_flip_left_right(distorted_image)

    # Because these operations are not commutative, consider randomizing
    # the order their operation.
    # NOTE: since per_image_standardization zeros the mean and makes
    # the stddev unit, this likely has no effect see tensorflow#1458.
    distorted_image = tf.image.random_brightness(distorted_image, max_delta=63)
    distorted_image = tf.image.random_contrast(distorted_image, lower=0.2, upper=1.8)

    # Subtract off the mean and divide by the variance of the pixels.
    float_image = tf.image.per_image_standardization(distorted_image)

    # Set the shapes of tensors.
    float_image.set_shape([height, width, 3])

    # Ensure that the random shuffling has good mixing properties.
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                             min_fraction_of_examples_in_queue)
    print ('Filling queue with %d CIFAR images before starting to train. '
           'This will take a few minutes.' % min_queue_examples)

    # Generate a batch of images and labels by building up a queue of examples.
    return _generate_image_and_label_batch(float_image, read_input.label,
                                           min_queue_examples, batch_size,
                                           shuffle=True)


def inputs(eval_data, data_dir, batch_size):
  train, test = read_csv(data_dir)

  if not eval_data:
    filenames = train.filename
    num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
  else:
    filenames = test.filename
    num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

  for f in filenames:
    if not tf.gfile.Exists(f):
      raise ValueError('Failed to find file: ' + f)

  # Create a queue that produces the filenames to read.
  filename_queue = tf.train.string_input_producer(filenames)
  #filename_queue = tf.train.string_input_producer(train.filename, num_epochs=None, shuffle=True, seed=None, shared_name=None, name=None)

  # Read examples from files in the filename queue.
  read_input = read_orchid11(filename_queue)
  reshaped_image = tf.cast(read_input.uint8image, tf.float32)

  height = IMAGE_SIZE
  width = IMAGE_SIZE

  # Image processing for evaluation.
  # Crop the central [height, width] of the image.
  resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image, height, width)

  # Subtract off the mean and divide by the variance of the pixels.
  float_image = tf.image.per_image_standardization(resized_image)

  # Set the shapes of tensors.
  float_image.set_shape([height, width, 3])

  # Ensure that the random shuffling has good mixing properties.
  min_fraction_of_examples_in_queue = 0.4
  min_queue_examples = int(num_examples_per_epoch *
                           min_fraction_of_examples_in_queue)

  # Generate a batch of images and labels by building up a queue of examples.
  return _generate_image_and_label_batch(float_image, read_input.label,
                                         min_queue_examples, batch_size,
                                         shuffle=False)
