#!/usr/local/bin/python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from datetime import datetime
import hashlib
import os.path
import random
import re
import sys
import tarfile

import numpy as np
from six.moves import urllib
import tensorflow as tf

from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile
from tensorflow.python.util import compat

FLAGS = None

MAX_NUM_IMAGES_PER_CLASS = 2 ** 32 - 1


def create_image_lists(image_dir, testing_percentage, validation_percentage):
  if not gfile.Exists(image_dir):
    tf.logging.error("Image directory '" + image_dir + "' not found.")
    return None

  total_sample_images = 0
  train_sample_images = 0
  test_sample_images = 0
  validation_sample_images = 0
  result = {}

  sub_dirs = [x[0] for x in gfile.Walk(image_dir)]

  # The root directory comes first, so skip it.
  is_root_dir = True
  for sub_dir in sub_dirs:
    if is_root_dir:
      is_root_dir = False
      continue

    extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
    file_list = []

    dir_name = os.path.basename(sub_dir)
    if dir_name == image_dir:
      continue

    tf.logging.info("Looking for images in '" + dir_name + "'")
    for extension in extensions:
      file_glob = os.path.join(image_dir, dir_name, '*.' + extension)
      file_list.extend(gfile.Glob(file_glob))

    if not file_list:
      tf.logging.warning('No files found')
      continue

    if len(file_list) < 20:
      tf.logging.warning(
          'WARNING: Folder has less than 20 images, which may cause issues.')
    elif len(file_list) > MAX_NUM_IMAGES_PER_CLASS:
      tf.logging.warning(
          'WARNING: Folder {} has more than {} images. Some images will '
          'never be selected.'.format(dir_name, MAX_NUM_IMAGES_PER_CLASS))

    label_name = dir_name.lower()
    training_images = []
    testing_images = []
    validation_images = []

    total_sample_images += len(file_list)

    for file_name in file_list:
      base_name = os.path.basename(file_name)
      # We want to ignore anything after '_nohash_' in the file name when
      # deciding which set to put an image in, the data set creator has a way of
      # grouping photos that are close variations of each other. For example
      # this is used in the plant disease data set to group multiple pictures of
      # the same leaf.
      hash_name = re.sub(r'_nohash_.*$', '', file_name)
      # This looks a bit magical, but we need to decide whether this file should
      # go into the training, testing, or validation sets, and we want to keep
      # existing files in the same set even if more files are subsequently
      # added.
      # To do that, we need a stable way of deciding based on just the file name
      # itself, so we do a hash of that and then use that to generate a
      # probability value that we use to assign it.
      hash_name_hashed = hashlib.sha1(compat.as_bytes(hash_name)).hexdigest()
      percentage_hash = ((int(hash_name_hashed, 16) %
                          (MAX_NUM_IMAGES_PER_CLASS + 1)) *
                         (100.0 / MAX_NUM_IMAGES_PER_CLASS))

      if percentage_hash < validation_percentage:
        validation_images.append(base_name)
      elif percentage_hash < (testing_percentage + validation_percentage):
        testing_images.append(base_name)
      else:
        training_images.append(base_name)

    _result = {
      'dir': dir_name,
      'training': training_images,
      'testing': testing_images,
      'validation': validation_images,
    }

    train_sample_images += len(training_images)
    test_sample_images += len(testing_images)
    validation_sample_images += len(validation_images)

    tf.logging.info("training size: {}, testing size: {}, validation size: {}"
                    .format(len(training_images), len(testing_images), len(validation_images)))

    result[label_name] = _result

  tf.logging.info("Total sample image: {}".format(total_sample_images))
  tf.logging.info("Training sample image: {}".format(train_sample_images))
  tf.logging.info("Testing sample image: {}".format(test_sample_images))
  tf.logging.info("Validation sample image: {}".format(validation_sample_images))
  return result


def create_image_lists_test(image_dir, testing_percentage, validation_percentage):
  if not gfile.Exists(image_dir):
    tf.logging.error("Image directory '" + image_dir + "' not found.")
    return None

  result = {}

  sub_dirs = [x[0] for x in gfile.Walk(image_dir)]

  # The root directory comes first, so skip it.
  is_root_dir = True
  for sub_dir in sub_dirs:
    if is_root_dir:
      is_root_dir = False
      continue

    extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
    file_list = []

    dir_name = os.path.basename(sub_dir)
    if dir_name == image_dir:
      continue

    tf.logging.info("Looking for images in '" + dir_name + "'")
    for extension in extensions:
      file_glob = os.path.join(image_dir, dir_name, '*.' + extension)
      file_list.extend(gfile.Glob(file_glob))

    if not file_list:
      tf.logging.warning('No files found')
      continue

    file_size = len(file_list)

    if file_size < 20:
      tf.logging.warning(
          'WARNING: Folder has less than 20 images, which may cause issues.')

    label_name = dir_name.lower()
    training_images = []
    testing_images = []
    validation_images = []

    random.shuffle(file_list)
    testing_idx = int(file_size * testing_percentage)
    validation_idx = int(file_size * validation_percentage)

    for index, file_name in enumerate(file_list):
      base_name = os.path.basename(file_name)

      if index < validation_idx:
        validation_images.append(base_name)
      elif index < (testing_idx + validation_idx):
        testing_images.append(base_name)
      else:
        training_images.append(base_name)

    _result = {
      'dir': dir_name,
      'training': training_images,
      'testing': testing_images,
      'validation': validation_images,
    }

    result[label_name] = _result
  return result


def maybe_download_and_extract(data_url, dest_directory):
  if not os.path.exists(dest_directory):
    os.makedirs(dest_directory)

  filename = data_url.split('/')[-1]
  filepath = os.path.join(dest_directory, filename)

  if not os.path.exists(filepath):
    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' %
                       (filename, float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()

    filepath, _ = urllib.request.urlretrieve(data_url, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    tf.logging.info('Successfully downloaded {0} {1} {2}'.format(filename, statinfo.st_size, 'bytes.'))
  tarfile.open(filepath, 'r:gz').extractall(dest_directory)


def create_model_graph(model_info, input_graph):
  if not os.path.exists(input_graph):
    model_dir = os.path.join(FLAGS.model_dir, 'imagenet')
    maybe_download_and_extract(model_info['data_url'], model_dir)
    model_path = os.path.join(model_dir, model_info['model_file_name'])
  else:
    model_path = input_graph

  with tf.Graph().as_default() as graph:
    with gfile.FastGFile(model_path, 'rb') as f:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(f.read())
      bottleneck_tensor, resized_input_tensor = (tf.import_graph_def(graph_def, name='',
          return_elements=[
              model_info['bottleneck_tensor_name'],
              model_info['resized_input_tensor_name'],
          ]))
  return graph, bottleneck_tensor, resized_input_tensor


def ensure_dir_exists(dir_name):
  if not os.path.exists(dir_name):
    os.makedirs(dir_name)


def prepare_file_system(summaries_dir):
  # Setup the directory we'll write summaries to for TensorBoard
  if tf.gfile.Exists(summaries_dir):
    tf.gfile.DeleteRecursively(summaries_dir)
  tf.gfile.MakeDirs(summaries_dir)
  #if FLAGS.intermediate_store_frequency > 0:
  #  ensure_dir_exists(FLAGS.intermediate_output_graphs_dir)
  if not tf.gfile.Exists(FLAGS.model_dir):
    tf.gfile.MakeDirs(FLAGS.model_dir)
  return


def add_jpeg_decoding(input_width, input_height, input_depth, input_mean, input_std):
  jpeg_data = tf.placeholder(tf.string, name='DecodeJPGInput')
  decoded_image = tf.image.decode_jpeg(jpeg_data, channels=input_depth)
  decoded_image_as_float = tf.cast(decoded_image, dtype=tf.float32)
  decoded_image_4d = tf.expand_dims(decoded_image_as_float, 0)
  resize_shape = tf.stack([input_height, input_width])
  resize_shape_as_int = tf.cast(resize_shape, dtype=tf.int32)
  resized_image = tf.image.resize_bilinear(decoded_image_4d, resize_shape_as_int)
  offset_image = tf.subtract(resized_image, input_mean)
  mul_image = tf.multiply(offset_image, 1.0 / input_std)
  return jpeg_data, mul_image


def get_image_path(image_lists, label_name, index, image_dir, category):
  if label_name not in image_lists:
    tf.logging.fatal('Label does not exist %s.', label_name)
  label_lists = image_lists[label_name]
  if category not in label_lists:
    tf.logging.fatal('Category does not exist %s.', category)
  category_list = label_lists[category]
  if not category_list:
    tf.logging.fatal('Label %s has no images in the category %s.',
                     label_name, category)
  mod_index = index % len(category_list)
  base_name = category_list[mod_index]
  sub_dir = label_lists['dir']
  full_path = os.path.join(image_dir, sub_dir, base_name)
  return full_path


def get_genus_image_path(image_lists, label_name, index, image_dir, category):
  if label_name not in image_lists:
    tf.logging.fatal('Label does not exist %s.', label_name)
  all_label_lists = image_lists[label_name]
  for li in xrange(len(all_label_lists)):
    label_lists = all_label_lists[li]
    if category not in label_lists:
      tf.logging.fatal('Category does not exist %s.', category)

    category_list = label_lists[category]
    if not category_list:
      tf.logging.fatal('Label %s has no images in the category %s.',
                       label_name, category)

  mod_index = index % len(category_list)
  base_name = category_list[mod_index]
  sub_dir = label_lists['dir']
  full_path = os.path.join(image_dir, sub_dir, base_name)
  return full_path


def get_bottleneck_path(image_lists, label_name, index, bottleneck_dir, category, archetecture):
  return get_image_path(image_lists, label_name, index, bottleneck_dir, category) + '_' + archetecture + '.txt'


def run_bottleneck_on_image(sess, image_data, image_data_tensor,
                            decoded_image_tensor, resized_input_tensor,
                            bottleneck_tensor):
  # First decode the JPEG image, resize it, and rescale the pixel values.
  resized_input_values = sess.run(decoded_image_tensor, {image_data_tensor: image_data})
  # Then run it through the recognition network.
  bottleneck_values = sess.run(bottleneck_tensor, {resized_input_tensor: resized_input_values})
  bottleneck_values = np.squeeze(bottleneck_values)
  return bottleneck_values


def create_bottleneck_file(bottleneck_path, image_lists, label_name, index,
                           image_dir, category, sess, jpeg_data_tensor,
                           decoded_image_tensor, resized_input_tensor,
                           bottleneck_tensor):
  """Create a single bottleneck file."""
  tf.logging.info('Creating bottleneck at ' + bottleneck_path)

  image_path = get_image_path(image_lists, label_name, index, image_dir, category)
  if not gfile.Exists(image_path):
    tf.logging.fatal('File does not exist %s', image_path)

  image_data = gfile.FastGFile(image_path, 'rb').read()

  try:
    bottleneck_values = run_bottleneck_on_image(
        sess, image_data, jpeg_data_tensor, decoded_image_tensor,
        resized_input_tensor, bottleneck_tensor)
  except Exception as e:
    raise RuntimeError('Error during processing file %s (%s)' % (image_path, str(e)))

  bottleneck_string = ','.join(str(x) for x in bottleneck_values)
  with open(bottleneck_path, 'w') as bottleneck_file:
    bottleneck_file.write(bottleneck_string)


def create_bottleneck_genus_file(bottleneck_path, image_lists, label_name, index,
                                 image_dir, category, sess, jpeg_data_tensor,
                                 decoded_image_tensor, resized_input_tensor,
                                 genus_tensor):
  """Create a single bottleneck file."""
  tf.logging.info('Creating genus bottleneck at ' + bottleneck_path)

  image_path = get_image_path(image_lists, label_name, index, image_dir, category)
  if not gfile.Exists(image_path):
    tf.logging.fatal('File does not exist %s', image_path)

  image_data = gfile.FastGFile(image_path, 'rb').read()

  try:
    bottleneck_values = run_bottleneck_on_image(
      sess, image_data, jpeg_data_tensor, decoded_image_tensor,
      resized_input_tensor, genus_tensor)
  except Exception as e:
    raise RuntimeError('Error during processing file %s (%s)' % (image_path, str(e)))

  bottleneck_string = ','.join(str(float(x)) for x in bottleneck_values)
  with open(bottleneck_path, 'w') as bottleneck_file:
    bottleneck_file.write(bottleneck_string)


def get_or_create_bottleneck(sess, image_lists, label_name, index, image_dir,
                             category, bottleneck_dir, jpeg_data_tensor,
                             decoded_image_tensor, resized_input_tensor,
                             bottleneck_tensor, archetecture):
  label_lists = image_lists[label_name]
  sub_dir = label_lists['dir']
  sub_dir_path = os.path.join(bottleneck_dir, sub_dir)
  ensure_dir_exists(sub_dir_path)
  bottleneck_path = get_bottleneck_path(image_lists, label_name, index, bottleneck_dir, category, archetecture)

  if not os.path.exists(bottleneck_path):
    create_bottleneck_file(bottleneck_path, image_lists, label_name, index,
                           image_dir, category, sess, jpeg_data_tensor,
                           decoded_image_tensor, resized_input_tensor,
                           bottleneck_tensor)
  with open(bottleneck_path, 'r') as bottleneck_file:
    bottleneck_string = bottleneck_file.read()
  did_hit_error = False
  try:
    bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
  except ValueError:
    tf.logging.warning('Invalid float found, recreating bottleneck')
    did_hit_error = True
  if did_hit_error:
    create_bottleneck_file(bottleneck_path, image_lists, label_name, index,
                           image_dir, category, sess, jpeg_data_tensor,
                           decoded_image_tensor, resized_input_tensor,
                           bottleneck_tensor)
    with open(bottleneck_path, 'r') as bottleneck_file:
      bottleneck_string = bottleneck_file.read()
    # Allow exceptions to propagate here, since they shouldn't happen after a
    # fresh creation
    bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
  return bottleneck_values


def cache_bottlenecks(sess, image_lists, image_dir, bottleneck_dir,
                      jpeg_data_tensor, decoded_image_tensor,
                      resized_input_tensor, bottleneck_tensor, archetecture):
  how_many_bottlenecks = 0
  ensure_dir_exists(bottleneck_dir)
  for label_name, label_lists in image_lists.items():
    for category in ['training', 'testing', 'validation']:
      category_list = label_lists[category]
      for index, unused_base_name in enumerate(category_list):
        get_or_create_bottleneck(
            sess, image_lists, label_name, index, image_dir, category,
            bottleneck_dir, jpeg_data_tensor, decoded_image_tensor,
            resized_input_tensor, bottleneck_tensor, archetecture)

        how_many_bottlenecks += 1
        if how_many_bottlenecks % 100 == 0:
          tf.logging.info(str(how_many_bottlenecks) + ' bottleneck files created.')


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


def add_genus_training_ops(class_count, final_tensor_name, bottleneck_tensor, model_info, bottleneck_tensor_size):
  with tf.name_scope('input_genus'):
    bottleneck_input = tf.placeholder_with_default(
      bottleneck_tensor,
      shape=[None, bottleneck_tensor_size],
      name='BottleneckInputPlaceholder')

    ground_truth_input = tf.placeholder(tf.float32,
                                        [None, class_count],
                                        name='GroundTruthInput')

  layer_name = 'final_training_ops_genus'
  with tf.name_scope(layer_name):
    with tf.name_scope('fullyc_1'):
      with tf.name_scope('weights'):
        initial_value = tf.truncated_normal([bottleneck_tensor_size, model_info['hidden_size']], stddev=0.001)
        fullyc_1_weights = tf.Variable(initial_value, name='weights')
        variable_summaries(fullyc_1_weights)
      with tf.name_scope('biases'):
        fullyc_1_biases = tf.Variable(tf.zeros([model_info['hidden_size']]), name='biases')
        variable_summaries(fullyc_1_biases)
      with tf.name_scope('Wx_plus_b'):
        fullyc_1_logits = tf.matmul(bottleneck_input, fullyc_1_weights) + fullyc_1_biases
        tf.summary.histogram('pre_activations', fullyc_1_logits)
        fullyc_1_hidden = tf.nn.relu(fullyc_1_logits)

    with tf.name_scope('fullyc_2'):
      with tf.name_scope('weights'):
        initial_value = tf.truncated_normal([model_info['hidden_size'], class_count], stddev=0.001)
        fullyc_2_weights = tf.Variable(initial_value, name='weights')
        variable_summaries(fullyc_2_weights)
      with tf.name_scope('biases'):
        fullyc_2_biases = tf.Variable(tf.zeros([class_count]), name='biases')
        variable_summaries(fullyc_2_biases)
      with tf.name_scope('Wx_plus_b'):
        fullyc_2_logits = tf.matmul(fullyc_1_hidden, fullyc_2_weights) + fullyc_2_biases
        tf.summary.histogram('pre_activations', fullyc_2_logits)

  final_tensor = tf.nn.softmax(fullyc_2_logits, name=final_tensor_name)
  tf.summary.histogram('activations', final_tensor)

  with tf.name_scope('cross_entropy_genus'):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=ground_truth_input, logits=fullyc_2_logits)
    with tf.name_scope('total'):
      cross_entropy_mean = tf.reduce_mean(cross_entropy)
  tf.summary.scalar('cross_entropy', cross_entropy_mean)

  with tf.name_scope('train_genus'):
    optimizer = tf.train.GradientDescentOptimizer(model_info['learning_rate'])
    train_step = optimizer.minimize(cross_entropy_mean)

  return (train_step, cross_entropy_mean, bottleneck_input, ground_truth_input, final_tensor)


def add_bulbophyllum_training_ops(class_count, final_tensor_name, bottleneck_tensor, model_info, bottleneck_tensor_size):
  with tf.name_scope('input_bulbophyllum'):
    bottleneck_input = tf.placeholder_with_default(
        bottleneck_tensor,
        shape=[None, bottleneck_tensor_size],
        name='BottleneckInputPlaceholder')

    ground_truth_input = tf.placeholder(tf.float32,
                                        [None, class_count],
                                        name='GroundTruthInput')

  layer_name = 'final_training_ops_bulbophyllum'
  with tf.name_scope(layer_name):
    with tf.name_scope('fullyc_1'):
      with tf.name_scope('weights'):
        initial_value = tf.truncated_normal([bottleneck_tensor_size, model_info['hidden_size']], stddev=0.001)
        fullyc_1_weights = tf.Variable(initial_value, name='weights')
        variable_summaries(fullyc_1_weights)
      with tf.name_scope('biases'):
        fullyc_1_biases = tf.Variable(tf.zeros([model_info['hidden_size']]), name='biases')
        variable_summaries(fullyc_1_biases)
      with tf.name_scope('Wx_plus_b'):
        fullyc_1_logits = tf.matmul(bottleneck_input, fullyc_1_weights) + fullyc_1_biases
        tf.summary.histogram('pre_activations', fullyc_1_logits)
        fullyc_1_hidden = tf.nn.relu(fullyc_1_logits)

    with tf.name_scope('fullyc_2'):
      with tf.name_scope('weights'):
        initial_value = tf.truncated_normal([model_info['hidden_size'], class_count], stddev=0.001)
        fullyc_2_weights = tf.Variable(initial_value, name='weights')
        variable_summaries(fullyc_2_weights)
      with tf.name_scope('biases'):
        fullyc_2_biases = tf.Variable(tf.zeros([class_count]), name='biases')
        variable_summaries(fullyc_2_biases)
      with tf.name_scope('Wx_plus_b'):
        fullyc_2_logits = tf.matmul(fullyc_1_hidden, fullyc_2_weights) + fullyc_2_biases
        tf.summary.histogram('pre_activations', fullyc_2_logits)

  final_tensor = tf.nn.softmax(fullyc_2_logits, name=final_tensor_name)
  tf.summary.histogram('activations', final_tensor)

  with tf.name_scope('cross_entropy_bulbophyllum'):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=ground_truth_input, logits=fullyc_2_logits)
    with tf.name_scope('total'):
      cross_entropy_mean = tf.reduce_mean(cross_entropy)
  tf.summary.scalar('cross_entropy', cross_entropy_mean)

  with tf.name_scope('train_bulbophyllum'):
    optimizer = tf.train.GradientDescentOptimizer(model_info['learning_rate'])
    train_step = optimizer.minimize(cross_entropy_mean)

  return (train_step, cross_entropy_mean, bottleneck_input, ground_truth_input, final_tensor)


def add_dendrobium_training_ops(class_count, final_tensor_name, bottleneck_tensor, model_info, bottleneck_tensor_size):
  with tf.name_scope('input_dendrobium'):
    bottleneck_input = tf.placeholder_with_default(
        bottleneck_tensor,
        shape=[None, bottleneck_tensor_size],
        name='BottleneckInputPlaceholder')

    ground_truth_input = tf.placeholder(tf.float32,
                                        [None, class_count],
                                        name='GroundTruthInput')

  layer_name = 'final_training_ops_dendrobium'
  with tf.name_scope(layer_name):
    with tf.name_scope('fullyc_1'):
      with tf.name_scope('weights'):
        initial_value = tf.truncated_normal([bottleneck_tensor_size, model_info['hidden_size']], stddev=0.001)
        fullyc_1_weights = tf.Variable(initial_value, name='weights')
        variable_summaries(fullyc_1_weights)
      with tf.name_scope('biases'):
        fullyc_1_biases = tf.Variable(tf.zeros([model_info['hidden_size']]), name='biases')
        variable_summaries(fullyc_1_biases)
      with tf.name_scope('Wx_plus_b'):
        fullyc_1_logits = tf.matmul(bottleneck_input, fullyc_1_weights) + fullyc_1_biases
        tf.summary.histogram('pre_activations', fullyc_1_logits)
        fullyc_1_hidden = tf.nn.relu(fullyc_1_logits)

    with tf.name_scope('fullyc_2'):
      with tf.name_scope('weights'):
        initial_value = tf.truncated_normal([model_info['hidden_size'], class_count], stddev=0.001)
        fullyc_2_weights = tf.Variable(initial_value, name='weights')
        variable_summaries(fullyc_2_weights)
      with tf.name_scope('biases'):
        fullyc_2_biases = tf.Variable(tf.zeros([class_count]), name='biases')
        variable_summaries(fullyc_2_biases)
      with tf.name_scope('Wx_plus_b'):
        fullyc_2_logits = tf.matmul(fullyc_1_hidden, fullyc_2_weights) + fullyc_2_biases
        tf.summary.histogram('pre_activations', fullyc_2_logits)

  final_tensor = tf.nn.softmax(fullyc_2_logits, name=final_tensor_name)
  tf.summary.histogram('activations', final_tensor)

  with tf.name_scope('cross_entropy_dendrobium'):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=ground_truth_input, logits=fullyc_2_logits)
    with tf.name_scope('total'):
      cross_entropy_mean = tf.reduce_mean(cross_entropy)
  tf.summary.scalar('cross_entropy', cross_entropy_mean)

  with tf.name_scope('train_dendrobium'):
    optimizer = tf.train.GradientDescentOptimizer(model_info['learning_rate'])
    train_step = optimizer.minimize(cross_entropy_mean)

  return (train_step, cross_entropy_mean, bottleneck_input, ground_truth_input, final_tensor)


def add_paphiopedilum_training_ops(class_count, final_tensor_name, bottleneck_tensor, model_info, bottleneck_tensor_size):
  with tf.name_scope('input_paphiopedilum'):
    bottleneck_input = tf.placeholder_with_default(
        bottleneck_tensor,
        shape=[None, bottleneck_tensor_size],
        name='BottleneckInputPlaceholder')

    ground_truth_input = tf.placeholder(tf.float32,
                                        [None, class_count],
                                        name='GroundTruthInput')

  layer_name = 'final_training_ops_paphiopedilum'
  with tf.name_scope(layer_name):
    with tf.name_scope('fullyc_1'):
      with tf.name_scope('weights'):
        initial_value = tf.truncated_normal([bottleneck_tensor_size, model_info['hidden_size']], stddev=0.001)
        fullyc_1_weights = tf.Variable(initial_value, name='weights')
        variable_summaries(fullyc_1_weights)
      with tf.name_scope('biases'):
        fullyc_1_biases = tf.Variable(tf.zeros([model_info['hidden_size']]), name='biases')
        variable_summaries(fullyc_1_biases)
      with tf.name_scope('Wx_plus_b'):
        fullyc_1_logits = tf.matmul(bottleneck_input, fullyc_1_weights) + fullyc_1_biases
        tf.summary.histogram('pre_activations', fullyc_1_logits)
        fullyc_1_hidden = tf.nn.relu(fullyc_1_logits)

    with tf.name_scope('fullyc_2'):
      with tf.name_scope('weights'):
        initial_value = tf.truncated_normal([model_info['hidden_size'], class_count], stddev=0.001)
        fullyc_2_weights = tf.Variable(initial_value, name='weights')
        variable_summaries(fullyc_2_weights)
      with tf.name_scope('biases'):
        fullyc_2_biases = tf.Variable(tf.zeros([class_count]), name='biases')
        variable_summaries(fullyc_2_biases)
      with tf.name_scope('Wx_plus_b'):
        fullyc_2_logits = tf.matmul(fullyc_1_hidden, fullyc_2_weights) + fullyc_2_biases
        tf.summary.histogram('pre_activations', fullyc_2_logits)

  final_tensor = tf.nn.softmax(fullyc_2_logits, name=final_tensor_name)
  tf.summary.histogram('activations', final_tensor)

  with tf.name_scope('cross_entropy_paphiopedilum'):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=ground_truth_input, logits=fullyc_2_logits)
    with tf.name_scope('total'):
      cross_entropy_mean = tf.reduce_mean(cross_entropy)
  tf.summary.scalar('cross_entropy', cross_entropy_mean)

  with tf.name_scope('train_paphiopedilum'):
    optimizer = tf.train.GradientDescentOptimizer(model_info['learning_rate'])
    train_step = optimizer.minimize(cross_entropy_mean)

  return (train_step, cross_entropy_mean, bottleneck_input, ground_truth_input, final_tensor)


def add_final_training_ops(class_count, final_tensor_name, bottleneck_tensor, model_info, bottleneck_tensor_size):
  with tf.name_scope('input_final'):
    bottleneck_input = tf.placeholder_with_default(
      bottleneck_tensor,
      shape=[None, bottleneck_tensor_size],
      name='BottleneckInputPlaceholder')

    ground_truth_input = tf.placeholder(tf.float32,
                                        [None, class_count],
                                        name='GroundTruthInput')

  layer_name = 'final_training_ops'
  with tf.name_scope(layer_name):
    with tf.name_scope('fullyc_1'):
      with tf.name_scope('weights'):
        initial_value = tf.truncated_normal([bottleneck_tensor_size, model_info['hidden_size']], stddev=0.001)
        fullyc_1_weights = tf.Variable(initial_value, name='weights')
        variable_summaries(fullyc_1_weights)
      with tf.name_scope('biases'):
        fullyc_1_biases = tf.Variable(tf.zeros([model_info['hidden_size']]), name='biases')
        variable_summaries(fullyc_1_biases)
      with tf.name_scope('Wx_plus_b'):
        fullyc_1_logits = tf.matmul(bottleneck_input, fullyc_1_weights) + fullyc_1_biases
        tf.summary.histogram('pre_activations', fullyc_1_logits)
        fullyc_1_hidden = tf.nn.relu(fullyc_1_logits)

    with tf.name_scope('fullyc_2'):
      with tf.name_scope('weights'):
        initial_value = tf.truncated_normal([model_info['hidden_size'], class_count], stddev=0.001)
        #initial_value = tf.truncated_normal([bottleneck_tensor_size, class_count], stddev=0.001)
        fullyc_2_weights = tf.Variable(initial_value, name='weights')
        variable_summaries(fullyc_2_weights)
      with tf.name_scope('biases'):
        fullyc_2_biases = tf.Variable(tf.zeros([class_count]), name='biases')
        variable_summaries(fullyc_2_biases)
      with tf.name_scope('Wx_plus_b'):
        fullyc_2_logits = tf.matmul(fullyc_1_hidden, fullyc_2_weights) + fullyc_2_biases
        #fullyc_2_logits = tf.matmul(bottleneck_input, fullyc_2_weights) + fullyc_2_biases
        tf.summary.histogram('pre_activations', fullyc_2_logits)

  final_tensor = tf.nn.softmax(fullyc_2_logits, name=final_tensor_name) # The output predition
  tf.summary.histogram('activations', final_tensor)

  with tf.name_scope('cross_entropy'):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=ground_truth_input, logits=fullyc_2_logits)
    with tf.name_scope('total'):
      cross_entropy_mean = tf.reduce_mean(cross_entropy)
  tf.summary.scalar('cross_entropy', cross_entropy_mean)

  with tf.name_scope('train_final'):
    optimizer = tf.train.GradientDescentOptimizer(model_info['learning_rate'])
    train_step = optimizer.minimize(cross_entropy_mean)

  return (train_step, cross_entropy_mean, bottleneck_input, ground_truth_input, final_tensor)


def add_evaluation_step(name_scope, result_tensor, ground_truth_tensor):
  with tf.name_scope(name_scope):
    with tf.name_scope('correct_prediction'):
      prediction = tf.argmax(result_tensor, 1)
      correct_prediction = tf.equal(prediction, tf.argmax(ground_truth_tensor, 1))
    with tf.name_scope('accuracy'):
      evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  tf.summary.scalar(name_scope, evaluation_step)
  return evaluation_step, prediction


def getKeys(image_lists):
  lst = list(image_lists.keys())
  lst.sort()
  return lst


def get_random_cached_bottlenecks(sess, image_lists, how_many, category,
                                  bottleneck_dir, image_dir, jpeg_data_tensor,
                                  decoded_image_tensor, resized_input_tensor,
                                  bottleneck_tensor, archetecture):
  class_count = len(getKeys(image_lists))
  bottlenecks = []
  ground_truths = []
  filenames = []

  if how_many >= 0:
    # Retrieve a random sample of bottlenecks.
    for unused_i in range(how_many):
      label_index = random.randrange(class_count)
      label_name = list(getKeys(image_lists))[label_index]
      image_index = random.randrange(MAX_NUM_IMAGES_PER_CLASS + 1)
      image_name = get_image_path(image_lists, label_name, image_index, image_dir, category)
      bottleneck = get_or_create_bottleneck(
          sess, image_lists, label_name, image_index, image_dir, category,
          bottleneck_dir, jpeg_data_tensor, decoded_image_tensor,
          resized_input_tensor, bottleneck_tensor, archetecture)
      ground_truth = np.zeros(class_count, dtype=np.float32)
      ground_truth[label_index] = 1.0
      bottlenecks.append(bottleneck)
      ground_truths.append(ground_truth)
      filenames.append(image_name)
  else:
    # Retrieve all bottlenecks.
    for label_index, label_name in enumerate(getKeys(image_lists)):
      for image_index, image_name in enumerate(
          image_lists[label_name][category]):
        image_name = get_image_path(image_lists, label_name, image_index, image_dir, category)
        bottleneck = get_or_create_bottleneck(
            sess, image_lists, label_name, image_index, image_dir, category,
            bottleneck_dir, jpeg_data_tensor, decoded_image_tensor,
            resized_input_tensor, bottleneck_tensor, archetecture)
        ground_truth = np.zeros(class_count, dtype=np.float32)
        ground_truth[label_index] = 1.0
        bottlenecks.append(bottleneck)
        ground_truths.append(ground_truth)
        filenames.append(image_name)
  return bottlenecks, ground_truths, filenames


def save_graph_to_file(sess, graph, graph_file_name, final_tensor_names):
  output_graph_def = graph_util.convert_variables_to_constants(sess, graph.as_graph_def(), final_tensor_names)
  with gfile.FastGFile(graph_file_name, 'wb') as f:
    f.write(output_graph_def.SerializeToString())
  return


def train(role, model_info, image_lists, final_results, add_final_training_ops, image_dir, bottleneck_dir, summaries_dir, input_graph):

  output_graph = os.path.join(FLAGS.model_dir, '{0}_output_graph.pb'.format(role))
  output_label = os.path.join(FLAGS.model_dir, '{0}_output_label.txt'.format(role))

  # Set up the pre-trained graph.
  graph, bottleneck_tensor, resized_image_tensor = (create_model_graph(model_info, input_graph))

  architecture = 'train'

  with tf.Session(graph=graph) as sess:
    # Set up the image decoding sub-graph.
    jpeg_data_tensor, decoded_image_tensor = add_jpeg_decoding(
      model_info['input_width'], model_info['input_height'],
      model_info['input_depth'], model_info['input_mean'],
      model_info['input_std'])

    cache_bottlenecks(sess, image_lists, image_dir,
                      bottleneck_dir, jpeg_data_tensor,
                      decoded_image_tensor, resized_image_tensor,
                      bottleneck_tensor, architecture)

    (train_step, cross_entropy, bottleneck_input, ground_truth_input, final_tensor) = add_final_training_ops(
      len(getKeys(image_lists)),
      final_results[0],
      bottleneck_tensor,
      model_info[role],
      model_info['bottleneck_tensor_size'])

    evaluation_step, prediction = add_evaluation_step('{0}_accuracy'.format(role), final_tensor, ground_truth_input)

    # Merge all the summaries and write them out to the summaries_dir
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(summaries_dir + '/{0}_train'.format(role), sess.graph)
    validation_writer = tf.summary.FileWriter(summaries_dir + '/{0}_validation'.format(role))

    # Set up all our weights to their initial default values.
    init = tf.global_variables_initializer()
    sess.run(init)

    for i in range(model_info[role]['how_many_training_steps']):
      (train_bottlenecks, train_ground_truth, _) = get_random_cached_bottlenecks(
        sess, image_lists, FLAGS.train_batch_size, 'training',
        bottleneck_dir, image_dir, jpeg_data_tensor,
        decoded_image_tensor, resized_image_tensor,
        bottleneck_tensor, architecture)

      # Feed the bottlenecks and ground truth into the graph, and run a training
      # step. Capture training summaries for TensorBoard with the `merged` op.
      train_summary, _ = sess.run(
        [merged, train_step],
        feed_dict={bottleneck_input: train_bottlenecks,
                   ground_truth_input: train_ground_truth})
      train_writer.add_summary(train_summary, i)

      # Every so often, print out how well the graph is training.
      is_last_step = (i + 1 == model_info[role]['how_many_training_steps'])
      if (i % FLAGS.eval_step_interval) == 0 or is_last_step:
        train_accuracy, cross_entropy_value = sess.run(
            [evaluation_step, cross_entropy],
            feed_dict={bottleneck_input: train_bottlenecks,
                       ground_truth_input: train_ground_truth})
        tf.logging.info('%s: Step %d: Train accuracy = %.1f%%' % (datetime.now(), i, train_accuracy * 100))
        tf.logging.info('%s: Step %d: Cross entropy = %f' % (datetime.now(), i, cross_entropy_value))

        validation_bottlenecks, validation_ground_truth, _ = (
            get_random_cached_bottlenecks(
                #sess, image_lists, FLAGS.validation_batch_size, 'validation',
                sess, image_lists, FLAGS.validation_batch_size, 'testing',
                bottleneck_dir, image_dir, jpeg_data_tensor,
                decoded_image_tensor, resized_image_tensor, bottleneck_tensor, architecture))

        # Run a validation step and capture training summaries for TensorBoard
        # with the `merged` op.
        validation_summary, validation_accuracy = sess.run(
          [merged, evaluation_step],
          feed_dict={bottleneck_input: validation_bottlenecks,
                     ground_truth_input: validation_ground_truth})
        validation_writer.add_summary(validation_summary, i)
        tf.logging.info('%s: Step %d: Validation accuracy = %.1f%% (N=%d)' %
                        (datetime.now(), i, validation_accuracy * 100,
                         len(validation_bottlenecks)))

      # Store intermediate results
      intermediate_frequency = FLAGS.intermediate_store_frequency

      if (intermediate_frequency > 0 and (i % intermediate_frequency == 0) and i > 0):
        intermediate_file_name = (FLAGS.intermediate_output_graphs_dir + 'intermediate_{0}_'.format(role) + str(i) + '.pb')
        tf.logging.info('Save intermediate result to : ' + intermediate_file_name)
        save_graph_to_file(sess, graph, intermediate_file_name, final_results)

      if ((cross_entropy_value * 10000) <= model_info[role]['convergence']):
        break

    # We've completed all our training, so run a final test evaluation on
    # some new images we haven't used before.
    test_bottlenecks, test_ground_truth, test_filenames = (
        get_random_cached_bottlenecks(
            sess, image_lists, FLAGS.test_batch_size, 'testing',
            bottleneck_dir, image_dir, jpeg_data_tensor,
            decoded_image_tensor, resized_image_tensor, bottleneck_tensor, architecture))
    test_accuracy, predictions = sess.run(
        [evaluation_step, prediction],
        feed_dict={bottleneck_input: test_bottlenecks, ground_truth_input: test_ground_truth})
    tf.logging.info('Final test accuracy = %.1f%% (N=%d)' % (test_accuracy * 100, len(test_bottlenecks)))

    if FLAGS.print_misclassified_test_images:
      tf.logging.info('=== MISCLASSIFIED TEST IMAGES ===')
      for i, test_filename in enumerate(test_filenames):
        if predictions[i] != test_ground_truth[i].argmax():
          tf.logging.info('%70s  %s' % (test_filename, list(getKeys(image_lists))[predictions[i]]))

    with gfile.FastGFile(output_label, 'w') as f:
      f.write('\n'.join(getKeys(image_lists)) + '\n')

    save_graph_to_file(sess,
                       graph,
                       output_graph,
                       final_results)


def load_labels(label_file):
  label = []
  proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
  for l in proto_as_ascii_lines:
    label.append(l.rstrip())
  return label


def load_final_graph(model_path, final_output_tensors):
  with tf.Graph().as_default() as graph:
    with gfile.FastGFile(model_path, 'rb') as f:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(f.read())
      resized_input_tensor,\
      final_tensor,\
      bulbophyllum_tensor, \
      dendrobium_tensor, \
      paphiopedilum_tensor = (tf.import_graph_def(
        graph_def,
        name='',
        return_elements=final_output_tensors))
  return graph, resized_input_tensor,\
         final_tensor,\
         bulbophyllum_tensor, \
         dendrobium_tensor, \
         paphiopedilum_tensor


def load_test_graph(model_path, final_output_tensors):
  with tf.Graph().as_default() as graph:
    with gfile.FastGFile(model_path, 'rb') as f:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(f.read())
      resized_input_tensor,\
      final_tensor = (tf.import_graph_def(
        graph_def,
        name='',
        return_elements=final_output_tensors))
  return graph, resized_input_tensor, final_tensor


def extend_arr(list, n):
  return (list + [''] * (n - len(list)))[:n]


def delFile(filename):
    try:
      os.remove(filename)
    except IOError as e:
        print('Error: %s' % e.strerror)


def resetBottleneck(architecture, bottleneck_dir):
  sub_dirs = [x[0] for x in gfile.Walk(bottleneck_dir)]

  is_root_dir = True
  for sub_dir in sub_dirs:
    if is_root_dir:
      is_root_dir = False
      continue

    file_list = []
    dir_name = os.path.basename(sub_dir)
    file_glob = os.path.join(bottleneck_dir, dir_name, '*.txt')
    file_list.extend(gfile.Glob(file_glob))

    l = len(architecture) + len('.txt')

    for f in file_list:
      if f[-l:] == '{0}.txt'.format(architecture):
        tf.logging.info('Remove file: {0}'.format(f))
        delFile(f)


def main(_):
  # Needed to make sure the logging output is visible.
  # See https://github.com/tensorflow/tensorflow/issues/3047
  tf.logging.set_verbosity(tf.logging.INFO)

  final_result_genus = 'final_result_genus'
  final_result_bulbophyllum = 'final_result_bulbophyllum'
  final_result_dendrobium = 'final_result_dendrobium'
  final_result_paphiopedilum = 'final_result_paphiopedilum'
  final_result = 'final_orchid_result'

  # Gather information about the model architecture we'll be using.
  model_info = {
    # pylint: disable=line-too-long
    'data_url': 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz',
    # pylint: enable=line-too-long
    'bottleneck_tensor_name': 'pool_3/_reshape:0',
    'bottleneck_tensor_size': 2048,
    'input_width': 299,
    'input_height': 299,
    'input_depth': 3,
    'resized_input_tensor_name': 'Mul:0',
    'model_file_name': 'classify_image_graph_def.pb',
    'input_mean': 128,
    'input_std': 128,
    'genus': {
      'how_many_training_steps': 20000,
      'learning_rate': 0.1,
      'hidden_size': 128,
      'convergence': 1
    },
    'bobuphyllum': {
      'how_many_training_steps': 20000,
      'learning_rate': 0.1,
      'hidden_size': 512,
      'convergence': 1
    },
    'dendrobium': {
      'how_many_training_steps': 20000,
      'learning_rate': 0.1,
      'hidden_size': 512,
      'convergence': 1
    },
    'paphiopedelum': {
      'how_many_training_steps': 20000,
      'learning_rate': 0.1,
      'hidden_size': 512,
      'convergence': 1
    },
    'final': {
      'how_many_training_steps': 15000,
      'learning_rate': 0.01,
      'hidden_size': 256,
      'convergence': 9
    }
  }

  if FLAGS.running_method == 'all_train' or FLAGS.running_method == 'train_genus':
    image_dir = '/Volumes/Data/_dataset/orchid-genus/flower_photos/'
    bottleneck_dir = '/Volumes/Data/_dataset/orchid-genus/bottleneck'
    summaries_dir = os.path.join(FLAGS.summaries_dir, 'genus')

    prepare_file_system(summaries_dir)

    # Look at the folder structure, and create lists of all the images.
    image_lists  = create_image_lists(image_dir, FLAGS.testing_percentage, FLAGS.validation_percentage)

    # genus train ##########################################################################
    train('genus',
          model_info,
          image_lists,
          [final_result_genus],
          add_genus_training_ops,
          image_dir,
          bottleneck_dir,
          summaries_dir,
          'none')


  if FLAGS.running_method == 'all_train' or FLAGS.running_method == 'train_bulbophyllum':
    image_dir = '/Volumes/Data/_dataset/orchid-bulbophyllum_001/flower_photos/'
    bottleneck_dir = '/Volumes/Data/_dataset/orchid-bulbophyllum_001/bottleneck'
    summaries_dir = os.path.join(FLAGS.summaries_dir, 'bulbophyllum')

    prepare_file_system(summaries_dir)

    # Look at the folder structure, and create lists of all the images.
    image_lists = create_image_lists(image_dir, FLAGS.testing_percentage, FLAGS.validation_percentage)

    # bulbophyllum train ####################################################################
    input_graph = os.path.join(FLAGS.model_dir, '{0}_output_graph.pb'.format('genus'))

    train('bulbophyllum',
          model_info,
          image_lists,
          [final_result_bulbophyllum, final_result_genus],
          add_bulbophyllum_training_ops,
          image_dir,
          bottleneck_dir,
          summaries_dir,
          input_graph)

  if FLAGS.running_method == 'all_train' or FLAGS.running_method == 'train_dendrobium':
    image_dir = '/Volumes/Data/_dataset/orchid-dendrobium_001/flower_photos/'
    bottleneck_dir = '/Volumes/Data/_dataset/orchid-dendrobium_001/bottleneck'
    summaries_dir = os.path.join(FLAGS.summaries_dir, 'dendrobium')

    prepare_file_system(summaries_dir)

    # Look at the folder structure, and create lists of all the images.
    image_lists  = create_image_lists(image_dir, FLAGS.testing_percentage, FLAGS.validation_percentage)

    # dendrobium train ######################################################################
    input_graph = os.path.join(FLAGS.model_dir, '{0}_output_graph.pb'.format('bulbophyllum'))

    train('dendrobium',
          model_info,
          image_lists,
          [final_result_dendrobium, final_result_bulbophyllum, final_result_genus],
          add_dendrobium_training_ops,
          image_dir,
          bottleneck_dir,
          summaries_dir,
          input_graph)

  if FLAGS.running_method == 'all_train' or FLAGS.running_method == 'train_paphiopedilum':
    image_dir = '/Volumes/Data/_dataset/orchid-paphiopedilum_001/flower_photos/'
    bottleneck_dir = '/Volumes/Data/_dataset/orchid-paphiopedilum_001/bottleneck'
    summaries_dir = os.path.join(FLAGS.summaries_dir, 'paphiopedilum')

    prepare_file_system(summaries_dir)

    # Look at the folder structure, and create lists of all the images.
    image_lists = create_image_lists(image_dir, FLAGS.testing_percentage, FLAGS.validation_percentage)

    # paphiopedilum train ####################################################################
    input_graph = os.path.join(FLAGS.model_dir, '{0}_output_graph.pb'.format('dendrobium'))

    train('paphiopedilum',
          model_info,
          image_lists,
          [final_result_paphiopedilum, final_result_dendrobium, final_result_bulbophyllum, final_result_genus],
          add_paphiopedilum_training_ops,
          image_dir,
          bottleneck_dir,
          summaries_dir,
          input_graph)

  if FLAGS.running_method == 'all_train' or FLAGS.running_method == 'train_final_all':
    #image_dir = '/Volumes/Data/_dataset/orchid-final/flower_photos/'
    #bottleneck_dir = '/Volumes/Data/_dataset/orchid-final/bottleneck'
    image_dir = '/Volumes/Data/_dataset/17Flowers/images/'
    bottleneck_dir = '/Volumes/Data/_dataset/17Flowers/bottleneck'
    summaries_dir = os.path.join(FLAGS.summaries_dir, '17f-final-all')

    prepare_file_system(summaries_dir)

    # Look at the folder structure, and create lists of all the images.
    image_lists = create_image_lists(image_dir, FLAGS.testing_percentage, FLAGS.validation_percentage)

    # paphiopedilum train ####################################################################
    input_graph = 'imagenet'

    train('final',
          model_info,
          image_lists,
          [final_result],
          add_final_training_ops,
          image_dir,
          bottleneck_dir,
          summaries_dir,
          input_graph)

  if FLAGS.running_method == 'all_train' or FLAGS.running_method == 'train_final':
    image_dir = '/Volumes/Data/_dataset/orchid-final/flower_photos/'
    bottleneck_dir = '/Volumes/Data/_dataset/orchid-final/bottleneck'
    summaries_dir = os.path.join(FLAGS.summaries_dir, 'final')

    prepare_file_system(summaries_dir)

    # Look at the folder structure, and create lists of all the images.
    image_lists = create_image_lists(image_dir, FLAGS.testing_percentage, FLAGS.validation_percentage)

    bottleneck_tensor_genus = final_result_genus + ':0'
    bottleneck_tensor_bulbophyllum = final_result_bulbophyllum + ':0'
    bottleneck_tensor_dendrobium = final_result_dendrobium + ':0'
    bottleneck_tensor_paphiopedilum = final_result_paphiopedilum + ':0'

    input_graph = os.path.join(FLAGS.model_dir, '{0}_output_graph.pb'.format('paphiopedilum'))

    graph, \
    resized_input_tensor, \
    genus_tensor, \
    bulbophyllum_tensor, \
    dendrobium_tensor, \
    paphiopedilum_tensor = load_final_graph(input_graph,
                                      [model_info['resized_input_tensor_name'],
                                      bottleneck_tensor_genus,
                                      bottleneck_tensor_bulbophyllum,
                                      bottleneck_tensor_dendrobium,
                                      bottleneck_tensor_paphiopedilum])

    with tf.Session(graph=graph) as sess:
      bottleneck_tensor = tf.concat([genus_tensor, bulbophyllum_tensor, dendrobium_tensor, paphiopedilum_tensor], axis=1)
      bottleneck_tensor_relu = tf.nn.relu(bottleneck_tensor)

      train_step, \
      cross_entropy, \
      bottleneck_input, \
      ground_truth_input, \
      final_tensor = add_final_training_ops(len(getKeys(image_lists)),
                                            final_result,
                                            bottleneck_tensor_relu,
                                            model_info['final'],
                                            bottleneck_tensor_size=25)

      jpeg_data_tensor, decoded_image_tensor = add_jpeg_decoding(
        model_info['input_width'], model_info['input_height'],
        model_info['input_depth'], model_info['input_mean'],
        model_info['input_std'])

      archetecture = 'final_input'
      if FLAGS.reset_bottleneck:
        resetBottleneck(archetecture, bottleneck_dir)

      evaluation_step, prediction = add_evaluation_step('final_accuracy', final_tensor, ground_truth_input)

      merged = tf.summary.merge_all()
      train_writer = tf.summary.FileWriter(summaries_dir + '/final_train', sess.graph)
      validation_writer = tf.summary.FileWriter(summaries_dir + '/final_validation')
      test_writer = tf.summary.FileWriter(summaries_dir + '/final_test')

      init = tf.global_variables_initializer()
      sess.run(init)

      # Iterate and train.
      for step in range(FLAGS.how_many_training_steps):

        train_bottlenecks, train_ground_truth, train_filenames = (get_random_cached_bottlenecks(
            sess, image_lists, FLAGS.train_batch_size, 'training',
            bottleneck_dir, image_dir, jpeg_data_tensor,
            decoded_image_tensor, resized_input_tensor, bottleneck_tensor, archetecture))

        train_summary, _ = sess.run(
          [merged, train_step],
          feed_dict={bottleneck_input: train_bottlenecks,
                     ground_truth_input: train_ground_truth})
        train_writer.add_summary(train_summary, step)

        is_last_step = (step + 1 == FLAGS.how_many_training_steps)
        if (step % FLAGS.eval_step_interval) == 0 or is_last_step:
          train_accuracy, cross_entropy_value = sess.run(
              [evaluation_step, cross_entropy],
              feed_dict={bottleneck_input: train_bottlenecks,
                         ground_truth_input: train_ground_truth})
          tf.logging.info('%s: Step %d: Train accuracy = %.1f%%' % (datetime.now(), step, train_accuracy * 100))
          tf.logging.info('%s: Step %d: Cross entropy = %f' % (datetime.now(), step, cross_entropy_value))

          validation_bottlenecks, validation_ground_truth, _ = (
              get_random_cached_bottlenecks(
                  sess, image_lists, FLAGS.validation_batch_size, 'validation',
                  bottleneck_dir, image_dir, jpeg_data_tensor,
                  decoded_image_tensor, resized_input_tensor, bottleneck_tensor, archetecture))

          validation_summary, validation_accuracy = sess.run(
            [merged, evaluation_step],
            feed_dict={bottleneck_input: validation_bottlenecks,
                       ground_truth_input: validation_ground_truth})
          validation_writer.add_summary(validation_summary, step)

          tf.logging.info('%s: Step %d: Validation accuracy = %.1f%% (N=%d)' %
                          (datetime.now(), step, validation_accuracy * 100,
                           len(validation_bottlenecks)))

          test_bottlenecks, test_ground_truth, test_filenames = (get_random_cached_bottlenecks(
              sess, image_lists, FLAGS.test_batch_size, 'testing',
              bottleneck_dir, image_dir, jpeg_data_tensor,
              decoded_image_tensor, resized_input_tensor, bottleneck_tensor, archetecture))

          test_summary, test_accuracy, predictions = sess.run(
            [merged, evaluation_step, prediction],
            feed_dict={bottleneck_input: test_bottlenecks, ground_truth_input: test_ground_truth})
          test_writer.add_summary(test_summary, step)
          tf.logging.info('Final test accuracy = %.1f%% (N=%d)' % (test_accuracy * 100, len(test_bottlenecks)))

        if ((cross_entropy_value * 10000) <= model_info['final']['convergence']):
          break

      if FLAGS.print_misclassified_test_images:
        tf.logging.info('=== MISCLASSIFIED TEST IMAGES ===')
        for i, test_filename in enumerate(test_filenames):
          if predictions[i] != test_ground_truth[i].argmax():
            tf.logging.info('%70s  %s' % (test_filename, list(getKeys(image_lists))[predictions[i]]))

      output_label = os.path.join(FLAGS.model_dir, 'final_output_label.txt')

      with gfile.FastGFile(output_label, 'w') as f:
        f.write('\n'.join(getKeys(image_lists)) + '\n')

      save_graph_to_file(sess,
                         graph,
                         os.path.join(FLAGS.model_dir, 'final_output_graph.pb'),
                         [final_result])

  if FLAGS.running_method == 'accuracy':
    image_dir = '/Volumes/Data/_dataset/orchid-final/flower_photos/'
    #image_dir = '/Volumes/Data/_dataset/orchid-final/open_flower_photos'
    model_dir = '/Volumes/Data/_dataset/models'
    labels = load_labels("{0}/{1}_output_label.txt".format(model_dir, 'final'))
    input_graph = os.path.join(model_dir, '{0}_output_graph.pb'.format('final'))
    #bottleneck_dir = '/Volumes/Data/_dataset/orchid-final/bottleneck'
    bottleneck_dir = '/Volumes/Data/_dataset/orchid-final/bottleneck_graph'
    #bottleneck_dir = '/Volumes/Data/_dataset/orchid_open-closed/bottleneck/raw/closed'

    # Look at the folder structure, and create lists of all the images.
    image_lists  = create_image_lists(image_dir, FLAGS.testing_percentage, FLAGS.validation_percentage)
    #image_lists = create_image_lists_test(image_dir, 1, 0)

    final_result_sensor = final_result + ':0'
    #final_result_sensor = 'final_training_ops/fullyc_2/Wx_plus_b/add:0'

    graph, \
    resized_input_tensor, \
    final_result_tensor = load_test_graph(os.path.join(model_dir, input_graph),
                                          [model_info['resized_input_tensor_name'],
                                           final_result_sensor])

    with tf.Session(graph=graph) as sess:
      results_input = tf.placeholder(tf.float32,
                                     [None, len(labels)],
                                     name='ResultsInput')

      ground_truth_input = tf.placeholder(tf.float32,
                                          [None, len(labels)],
                                          name='GroundTruthInput')

      prediction = tf.argmax(results_input, 1)
      correct_prediction = tf.equal(prediction, tf.argmax(ground_truth_input, 1))
      evaluation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

      jpeg_data_tensor, decoded_image_tensor = add_jpeg_decoding(
        model_info['input_width'], model_info['input_height'],
        model_info['input_depth'], model_info['input_mean'],
        model_info['input_std'])

      archetecture = 'graph'
      #archetecture = 'test'
      #if FLAGS.reset_bottleneck:
      #  resetBottleneck(archetecture)
      #resetBottleneck(archetecture, bottleneck_dir)

      #results, test_ground_truth, test_filenames = (get_random_cached_bottlenecks(
      #    sess, image_lists, FLAGS.test_batch_size, 'testing',
      #    bottleneck_dir, image_dir, jpeg_data_tensor,
      #    decoded_image_tensor, resized_input_tensor, final_result_tensor, archetecture))

      results, test_ground_truth, test_filenames = (get_random_cached_bottlenecks(
        sess, image_lists, -1, 'testing',
        bottleneck_dir, image_dir, jpeg_data_tensor,
        decoded_image_tensor, resized_input_tensor, final_result_tensor, archetecture))

      test_accuracy, predictions = sess.run([evaluation, prediction],
        feed_dict={results_input: results, ground_truth_input: test_ground_truth})

      tf.logging.info('Final test accuracy = %.1f%% (N=%d)' % (test_accuracy * 100, len(results)))

      tf.logging.info('=== MISCLASSIFIED TEST IMAGES ===')
      misc = 0
      for i, test_filename in enumerate(test_filenames):
        if predictions[i] != test_ground_truth[i].argmax():
          misc += 1
          tf.logging.info('{0:70s}:  {1}'.format(os.path.basename(test_filename), list(getKeys(image_lists))[predictions[i]]))
      tf.logging.info('Misclassified number: {0}/{1} images'.format(misc, len(results)))

  if FLAGS.running_method == 'predict':
    model_dir = '/Volumes/Data/_dataset/models'
    labels = load_labels("{0}/{1}_output_label.txt".format(model_dir, 'final'))
    input_graph = os.path.join(model_dir, '{0}_output_graph.pb'.format('final'))

    final_result_sensor = final_result + ':0'

    graph, \
    resized_input_tensor, \
    final_result_tensor = load_test_graph(os.path.join(model_dir, input_graph),
                                          [model_info['resized_input_tensor_name'],
                                           final_result_sensor])

    with tf.Session(graph=graph) as sess:
      # Set up the image decoding sub-graph.
      jpeg_data_tensor, decoded_image_tensor = add_jpeg_decoding(
        model_info['input_width'], model_info['input_height'],
        model_info['input_depth'], model_info['input_mean'],
        model_info['input_std'])

      #image_data = gfile.FastGFile(FLAGS.filename, 'rb').read()
      image_data = gfile.FastGFile('/Volumes/Data/_dataset/orchids/ThaiNativeOrchids/raw data/unknow/orchid27.jpg', 'rb').read()
      #image_data = gfile.FastGFile('/Volumes/Data/_dataset/orchid-final/flower_photos/dendrobium_chrysotoxum Lindl_เอื้องคำ/dendrobium_chrysotoxum lindl_เอื้องคำ_042.jpg').read()

      try:
        results = run_bottleneck_on_image(
          sess, image_data, jpeg_data_tensor, decoded_image_tensor,
          resized_input_tensor, final_result_tensor)
      except Exception as e:
        raise RuntimeError('Error during processing file %s (%s)' % (FLAGS.filename, str(e)))

      top_k = results.argsort()[::-1]
      for i in top_k:
        print("{0} {1:0.4f}".format(labels[i], results[i]))

if __name__ == '__main__':
  print ("Tensorflow version: {0}".format(tf.VERSION))
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--summaries_dir',
      type=str,
      default='/Volumes/Data/_dataset/retrain_logs',
      help='Where to save summary logs for TensorBoard.'
  )
  parser.add_argument(
      '--intermediate_store_frequency',
      type=int,
      default=0,
      help="""\
         How many steps to store intermediate graph. If "0" then will not
         store.\
      """
  )
  parser.add_argument(
      '--model_dir',
      type=str,
      default='/Volumes/Data/_dataset/models',
      help="""\
      Path to classify_image_graph_def.pb,
      imagenet_synset_to_human_label_map.txt, and
      imagenet_2012_challenge_label_map_proto.pbtxt.\
      """
  )
  parser.add_argument(
      '--testing_percentage',
      type=int,
      default=20,
      help='What percentage of images to use as a test set.'
  )
  parser.add_argument(
      '--validation_percentage',
      type=int,
      default=0,
      help='What percentage of images to use as a validation set.'
  )
  parser.add_argument(
      '--reset_bottleneck',
      default=False,
      help='Reset bottlenect files.',
      action='store_true'
  )
  parser.add_argument(
      '--how_many_training_steps',
      type=int,
      default=40000,
      help='How many training steps to run before ending.'
  )
  parser.add_argument(
      '--learning_rate',
      type=float,
      default=0.01,
      help='How large a learning rate to use when training.'
  )
  parser.add_argument(
      '--eval_step_interval',
      type=int,
      default=10,
      help='How often to evaluate the training results.'
  )
  parser.add_argument(
      '--train_batch_size',
      type=int,
      default=50,
      help='How many images to train on at a time.'
  )
  parser.add_argument(
      '--test_batch_size',
      type=int,
      default=-1,
      help="""\
      How many images to test on. This test set is only used once, to evaluate
      the final accuracy of the model after training completes.
      A value of -1 causes the entire test set to be used, which leads to more
      stable results across runs.\
      """
  )
  parser.add_argument(
      '--validation_batch_size',
      type=int,
      default=50,
      help="""\
      How many images to use in an evaluation batch. This validation set is
      used much more often than the test set, and is an early indicator of how
      accurate the model is during training.
      A value of -1 causes the entire validation set to be used, which leads to
      more stable results across training iterations, but may be slower on large
      training sets.\
      """
  )
  parser.add_argument(
      '--print_misclassified_test_images',
      default=False,
      help="""\
      Whether to print out a list of all misclassified test images.\
      """,
      action='store_true'
  )
  parser.add_argument(
      '--intermediate_output_graphs_dir',
      type=str,
      default='/Volumes/Data/_dataset/orchid_final/intermediate_graph/',
      help='Where to save the intermediate graphs.'
  )
  parser.add_argument(
      '--input_graph',
      type=str,
      default='/Volumes/Data/_dataset/orchid_final/models/final_output_graph-002.pb',
      help='Where to save the trained graph.'
  )
  parser.add_argument(
      '--filename',
      type=str,
      #default='/Volumes/Data/_dataset/orchid_final/flower_photos/bulbophyllum_dayanum Rchb_สิงโตขยุกขยุย/bulbophyllum_dayanum rchb_สิงโตขยุกขยุย_010.jpg',
      #default='/Volumes/Data/_dataset/orchid_final/flower_photos/paphiopedilum_intanon-villosum_อินทนนท์/paphiopedilum_intanon-villosum_อินทนนท์_007.jpg',
      #default='/Volumes/Data/_dataset/orchid_final/flower_photos/paphiopedilum_callosum_คางกบ/paphiopedilum_callosum_คางกบ_045.jpg',
      #default='/Volumes/Data/_dataset/orchid_final/flower_photos/dendrobium_lindleyi Steud_เอื้องผึ้ง/dendrobium_lindleyi steud_เอื้องผึ้ง_004.jpg',
      default='/Volumes/Data/_dataset/orchid_final/flower_photos/dendrobium_chrysotoxum Lindl_เอื้องคำ/dendrobium_chrysotoxum lindl_เอื้องคำ_004.jpg',
      help='The image file to predict.'
  )
  parser.add_argument(
      '--running_method',
      type=str,
      #default='all_train',
      #default='train_final',
      #default='accuracy',
      #default='predict',
      help="""\
      The training method 'add' to train all model otherwise \
      'genus' for genus training
      'dendrobium' for dentrobium training
      'paphiopedilum' for paphiopedilum training
      'bulbophyllum' for bulbophyllum training
      """,
  )
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
