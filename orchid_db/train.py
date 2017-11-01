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

# These are all parameters that are tied to the particular model architecture
# we're using for Inception v3. These include things like tensor names and their
# sizes. If you want to adapt this script to work with another model, you will
# need to update these to reflect the values in the network you're using.
MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1  # ~134M


def create_image_lists(image_dir, testing_percentage, validation_percentage):
  all_genus = ['bulbophyllum', 'dendrobium', 'paphiopedilum']
  all_genus_size = [len(all_genus[0]), len(all_genus[1]), len(all_genus[2])]

  if not gfile.Exists(image_dir):
    tf.logging.error("Image directory '" + image_dir + "' not found.")
    return None

  all_result = {}

  image_lists = {
    all_genus[0]: {},
    all_genus[1]: {},
    all_genus[2]: {}
  }

  genus_result = {}


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

    result = None
    for i in xrange(len(all_genus)):
      if dir_name[:all_genus_size[i]] == all_genus[i]:
        result = image_lists[all_genus[i]]
        break

    is_genus = False
    if not result:
      if dir_name[:len('genus')] == 'genus':
        result = genus_result
        is_genus = True

    if not is_genus:
      all_result[label_name] = _result

    result[label_name] = _result
  return all_result, image_lists, genus_result


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


def create_model_graph(model_info):
  if not os.path.exists(FLAGS.output_graph):
    model_dir = os.path.join(FLAGS.model_dir, 'imagenet')
    maybe_download_and_extract(model_info['data_url'], model_dir)
    model_path = os.path.join(model_dir, model_info['model_file_name'])
  else:
    model_path = FLAGS.output_graph

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


def prepare_file_system():
  # Setup the directory we'll write summaries to for TensorBoard
  if tf.gfile.Exists(FLAGS.summaries_dir):
    tf.gfile.DeleteRecursively(FLAGS.summaries_dir)
  tf.gfile.MakeDirs(FLAGS.summaries_dir)
  if FLAGS.intermediate_store_frequency > 0:
    ensure_dir_exists(FLAGS.intermediate_output_graphs_dir)
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


def add_genus_training_ops(class_count, final_tensor_name, bottleneck_tensor, bottleneck_tensor_size):
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
        initial_value = tf.truncated_normal([bottleneck_tensor_size, 1024], stddev=0.001)
        fullyc_1_weights = tf.Variable(initial_value, name='weights')
        variable_summaries(fullyc_1_weights)
      with tf.name_scope('biases'):
        fullyc_1_biases = tf.Variable(tf.zeros([1024]), name='biases')
        variable_summaries(fullyc_1_biases)
      with tf.name_scope('Wx_plus_b'):
        fullyc_1_logits = tf.matmul(bottleneck_input, fullyc_1_weights) + fullyc_1_biases
        tf.summary.histogram('pre_activations', fullyc_1_logits)
        fullyc_1_hidden = tf.nn.relu(fullyc_1_logits)

    with tf.name_scope('fullyc_2'):
      with tf.name_scope('weights'):
        initial_value = tf.truncated_normal([1024, class_count], stddev=0.001)
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
    optimizer = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)
    train_step = optimizer.minimize(cross_entropy_mean)

  return (train_step, cross_entropy_mean, bottleneck_input, ground_truth_input, final_tensor)


def add_bulbophyllum_training_ops(class_count, final_tensor_name, bottleneck_tensor, bottleneck_tensor_size):
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
        initial_value = tf.truncated_normal([bottleneck_tensor_size, 1024], stddev=0.001)
        fullyc_1_weights = tf.Variable(initial_value, name='weights')
        variable_summaries(fullyc_1_weights)
      with tf.name_scope('biases'):
        fullyc_1_biases = tf.Variable(tf.zeros([1024]), name='biases')
        variable_summaries(fullyc_1_biases)
      with tf.name_scope('Wx_plus_b'):
        fullyc_1_logits = tf.matmul(bottleneck_input, fullyc_1_weights) + fullyc_1_biases
        tf.summary.histogram('pre_activations', fullyc_1_logits)
        fullyc_1_hidden = tf.nn.relu(fullyc_1_logits)

    with tf.name_scope('fullyc_2'):
      with tf.name_scope('weights'):
        initial_value = tf.truncated_normal([1024, class_count], stddev=0.001)
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
    optimizer = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)
    train_step = optimizer.minimize(cross_entropy_mean)

  return (train_step, cross_entropy_mean, bottleneck_input, ground_truth_input, final_tensor)


def add_dendrobium_training_ops(class_count, final_tensor_name, bottleneck_tensor, bottleneck_tensor_size):
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
        initial_value = tf.truncated_normal([bottleneck_tensor_size, 1024], stddev=0.001)
        fullyc_1_weights = tf.Variable(initial_value, name='weights')
        variable_summaries(fullyc_1_weights)
      with tf.name_scope('biases'):
        fullyc_1_biases = tf.Variable(tf.zeros([1024]), name='biases')
        variable_summaries(fullyc_1_biases)
      with tf.name_scope('Wx_plus_b'):
        fullyc_1_logits = tf.matmul(bottleneck_input, fullyc_1_weights) + fullyc_1_biases
        tf.summary.histogram('pre_activations', fullyc_1_logits)
        fullyc_1_hidden = tf.nn.relu(fullyc_1_logits)

    with tf.name_scope('fullyc_2'):
      with tf.name_scope('weights'):
        initial_value = tf.truncated_normal([1024, class_count], stddev=0.001)
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
    optimizer = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)
    train_step = optimizer.minimize(cross_entropy_mean)

  return (train_step, cross_entropy_mean, bottleneck_input, ground_truth_input, final_tensor)


def add_paphiopedilum_training_ops(class_count, final_tensor_name, bottleneck_tensor, bottleneck_tensor_size):
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
        initial_value = tf.truncated_normal([bottleneck_tensor_size, 1024], stddev=0.001)
        fullyc_1_weights = tf.Variable(initial_value, name='weights')
        variable_summaries(fullyc_1_weights)
      with tf.name_scope('biases'):
        fullyc_1_biases = tf.Variable(tf.zeros([1024]), name='biases')
        variable_summaries(fullyc_1_biases)
      with tf.name_scope('Wx_plus_b'):
        fullyc_1_logits = tf.matmul(bottleneck_input, fullyc_1_weights) + fullyc_1_biases
        tf.summary.histogram('pre_activations', fullyc_1_logits)
        fullyc_1_hidden = tf.nn.relu(fullyc_1_logits)

    with tf.name_scope('fullyc_2'):
      with tf.name_scope('weights'):
        initial_value = tf.truncated_normal([1024, class_count], stddev=0.001)
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
    optimizer = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)
    train_step = optimizer.minimize(cross_entropy_mean)

  return (train_step, cross_entropy_mean, bottleneck_input, ground_truth_input, final_tensor)


def add_final_training_svm(class_count, final_tensor_name, bottleneck_tensor, bottleneck_tensor_size):
  with tf.name_scope('input_final'):
    bottleneck_input = tf.placeholder_with_default(
      bottleneck_tensor,
      shape=[None, bottleneck_tensor_size],
      name='BottleneckInputPlaceholder')

    ground_truth_input = tf.placeholder(tf.float32,
                                        [None, class_count],
                                        name='GroundTruthInput')

  layer_name = 'final_training_SVM_ops'
  with tf.name_scope(layer_name):
    with tf.name_scope('weights'):
      initial_value = tf.truncated_normal([bottleneck_tensor_size, class_count], stddev=0.001)
      W = tf.Variable(initial_value, name='weights')
      variable_summaries(W)
    with tf.name_scope('biases'):
      b = tf.Variable(tf.zeros([class_count]), name='biases')
      variable_summaries(b)
    with tf.name_scope('Wx_plus_b'):
      final_tensor = tf.add(tf.matmul(bottleneck_input, W), b, name=final_tensor_name)
      tf.summary.histogram('pre_activations', final_tensor)

    with tf.name_scope('cross_entropy_final'):
      cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=ground_truth_input, logits=final_tensor)
      with tf.name_scope('total'):
        cross_entropy_mean = tf.reduce_mean(cross_entropy)
    tf.summary.scalar('cross_entropy', cross_entropy_mean)

  with tf.name_scope('train_final'):
    regularization_loss = 0.5 * tf.reduce_sum(tf.square(W))
    hinge_loss = tf.reduce_sum(tf.maximum(tf.zeros([FLAGS.train_batch_size, 1]), 1 - ground_truth_input * final_tensor));
    svm_loss = regularization_loss + FLAGS.svmC * hinge_loss;
    train_step = tf.train.GradientDescentOptimizer(FLAGS.learning_rate).minimize(svm_loss)

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


def get_random_cached_bottlenecks(sess, image_lists, how_many, category,
                                  bottleneck_dir, image_dir, jpeg_data_tensor,
                                  decoded_image_tensor, resized_input_tensor,
                                  bottleneck_tensor, archetecture):
  class_count = len(image_lists.keys())
  bottlenecks = []
  ground_truths = []
  filenames = []

  if how_many >= 0:
    # Retrieve a random sample of bottlenecks.
    for unused_i in range(how_many):
      label_index = random.randrange(class_count)
      label_name = list(image_lists.keys())[label_index]
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
    for label_index, label_name in enumerate(image_lists.keys()):
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


def train(role, model_info, image_lists, final_results, add_final_training_ops):
  # Set up the pre-trained graph.
  graph, bottleneck_tensor, resized_image_tensor = (create_model_graph(model_info))

  architecture = 'train'

  with tf.Session(graph=graph) as sess:
    # Set up the image decoding sub-graph.
    jpeg_data_tensor, decoded_image_tensor = add_jpeg_decoding(
      model_info['input_width'], model_info['input_height'],
      model_info['input_depth'], model_info['input_mean'],
      model_info['input_std'])

    cache_bottlenecks(sess, image_lists, FLAGS.image_dir,
                      FLAGS.bottleneck_dir, jpeg_data_tensor,
                      decoded_image_tensor, resized_image_tensor,
                      bottleneck_tensor, architecture)

    (train_step, cross_entropy, bottleneck_input, ground_truth_input, final_tensor) = add_final_training_ops(
      len(image_lists.keys()),
      final_results[0],
      bottleneck_tensor,
      model_info['bottleneck_tensor_size'])

    evaluation_step, prediction = add_evaluation_step('{0}_accuracy'.format(role), final_tensor, ground_truth_input)

    # Merge all the summaries and write them out to the summaries_dir
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/{0}_train'.format(role), sess.graph)
    validation_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/{0}_validation'.format(role))

    # Set up all our weights to their initial default values.
    init = tf.global_variables_initializer()
    sess.run(init)

    for i in range(FLAGS.how_many_training_steps):
      (train_bottlenecks, train_ground_truth, _) = get_random_cached_bottlenecks(
        sess, image_lists, FLAGS.train_batch_size, 'training',
        FLAGS.bottleneck_dir, FLAGS.image_dir, jpeg_data_tensor,
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
      is_last_step = (i + 1 == FLAGS.how_many_training_steps)
      if (i % FLAGS.eval_step_interval) == 0 or is_last_step:
        train_accuracy, cross_entropy_value = sess.run(
            [evaluation_step, cross_entropy],
            feed_dict={bottleneck_input: train_bottlenecks,
                       ground_truth_input: train_ground_truth})
        tf.logging.info('%s: Step %d: Train accuracy = %.1f%%' % (datetime.now(), i, train_accuracy * 100))
        tf.logging.info('%s: Step %d: Cross entropy = %f' % (datetime.now(), i, cross_entropy_value))

        validation_bottlenecks, validation_ground_truth, _ = (
            get_random_cached_bottlenecks(
                sess, image_lists, FLAGS.validation_batch_size, 'validation',
                FLAGS.bottleneck_dir, FLAGS.image_dir, jpeg_data_tensor,
                decoded_image_tensor, resized_image_tensor, bottleneck_tensor))

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

    # We've completed all our training, so run a final test evaluation on
    # some new images we haven't used before.
    test_bottlenecks, test_ground_truth, test_filenames = (
        get_random_cached_bottlenecks(
            sess, image_lists, FLAGS.test_batch_size, 'testing',
            FLAGS.bottleneck_dir, FLAGS.image_dir, jpeg_data_tensor,
            decoded_image_tensor, resized_image_tensor, bottleneck_tensor))
    test_accuracy, predictions = sess.run(
        [evaluation_step, prediction],
        feed_dict={bottleneck_input: test_bottlenecks, ground_truth_input: test_ground_truth})
    tf.logging.info('Final test accuracy = %.1f%% (N=%d)' % (test_accuracy * 100, len(test_bottlenecks)))

    if FLAGS.print_misclassified_test_images:
      tf.logging.info('=== MISCLASSIFIED TEST IMAGES ===')
      for i, test_filename in enumerate(test_filenames):
        if predictions[i] != test_ground_truth[i].argmax():
          tf.logging.info('%70s  %s' % (test_filename, list(image_lists.keys())[predictions[i]]))

    with gfile.FastGFile('/Volumes/Data/_Corpus-data/orchid_final/models/{0}_labels.txt'.format(role), 'w') as f:
      f.write('\n'.join(image_lists.keys()) + '\n')

    save_graph_to_file(sess,
                       graph,
                       os.path.join(FLAGS.output_graph, '{0}_output_graph.pb'.format(role)),
                       final_results)


def load_labels(label_file):
  label = []
  proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
  for l in proto_as_ascii_lines:
    label.append(l.rstrip())
  return label


def load_graph(model_path,
               input_name,
               bottleneck_tensor_genus,
               bottleneck_tensor_bulbophyllum,
               bottleneck_tensor_dendrobium,
               bottleneck_tensor_paphiopedilum):
  with tf.Graph().as_default() as graph:
    with gfile.FastGFile(model_path, 'rb') as f:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(f.read())
      resized_input_tensor,\
      genus_tensor,\
      bulbophyllum_tensor, \
      dendrobium_tensor, \
      paphiopedilum_tensor = (tf.import_graph_def(
        graph_def,
        name='',
        return_elements=[
          input_name,
          bottleneck_tensor_genus,
          bottleneck_tensor_bulbophyllum,
          bottleneck_tensor_dendrobium,
          bottleneck_tensor_paphiopedilum
        ]))
  return graph, resized_input_tensor,\
         genus_tensor,\
         bulbophyllum_tensor, \
         dendrobium_tensor, \
         paphiopedilum_tensor


def extend_arr(list, n):
  return (list + [''] * (n - len(list)))[:n]


def main(_):
  # Needed to make sure the logging output is visible.
  # See https://github.com/tensorflow/tensorflow/issues/3047
  tf.logging.set_verbosity(tf.logging.INFO)

  final_result_genus = 'final_result_genus'
  final_result_bulbophyllum = 'final_result_bulbophyllum'
  final_result_dendrobium = 'final_result_dendrobium'
  final_result_paphiopedilum = 'final_result_paphiopedilum'

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
  }

  # Prepare necessary directories  that can be used during training
  prepare_file_system()

  # Look at the folder structure, and create lists of all the images.
  all_image_lists, specie_image_list, genus_image_list,  = create_image_lists(FLAGS.image_dir, FLAGS.testing_percentage, FLAGS.validation_percentage)


  if FLAGS.running_method == 'all_train' or FLAGS.running_method == 'train_genus':
    # genus train ##########################################################################
    train('genus',
          model_info,
          genus_image_list,
          [final_result_genus],
          add_genus_training_ops)


  if FLAGS.running_method == 'all_train' or FLAGS.running_method == 'train_bulbophyllum':
    # bulbophyllum train ####################################################################
    train('bulbophyllum',
          model_info,
          specie_image_list['bulbophyllum'],
          [final_result_bulbophyllum, final_result_genus],
          add_bulbophyllum_training_ops)

  if FLAGS.running_method == 'all_train' or FLAGS.running_method == 'train_dendrobium':
    # dendrobium train ######################################################################
    train('dendrobium',
          model_info,
          specie_image_list['dendrobium'],
          [final_result_dendrobium, final_result_bulbophyllum, final_result_genus],
          add_dendrobium_training_ops)

  if FLAGS.running_method == 'all_train' or FLAGS.running_method == 'train_paphiopedilum':
    # paphiopedilum train ####################################################################
    train('paphiopedilum',
          model_info,
          specie_image_list['paphiopedilum'],
          [final_result_paphiopedilum, final_result_dendrobium, final_result_bulbophyllum, final_result_genus],
          add_paphiopedilum_training_ops)

  if FLAGS.running_method == 'all_train' or FLAGS.running_method == 'train_final':
    workspace = '/Volumes/Data/_Corpus-data/orchid_final'
    genus_labels = load_labels("{0}/models/{1}_labels.txt".format(workspace, 'genus'))
    bulbophyllum_labels = load_labels("{0}/models/{1}_labels.txt".format(workspace, 'bulbophyllum'))
    dendrobium_labels = load_labels("{0}/models/{1}_labels.txt".format(workspace, 'dendrobium'))
    paphiopedilum_labels = load_labels("{0}/models/{1}_labels.txt".format(workspace, 'paphiopedilum'))

    bottleneck_tensor_genus = final_result_genus + ':0'
    bottleneck_tensor_bulbophyllum = final_result_bulbophyllum + ':0'
    bottleneck_tensor_dendrobium = final_result_dendrobium + ':0'
    bottleneck_tensor_paphiopedilum = final_result_paphiopedilum + ':0'

    graph, \
    resized_input_tensor, \
    genus_tensor, \
    bulbophyllum_tensor, \
    dendrobium_tensor, \
    paphiopedilum_tensor = load_graph(os.path.join(FLAGS.output_graph_path, 'paphiopedilum_output_graph.pb'),
                                      model_info['resized_input_tensor_name'],
                                      bottleneck_tensor_genus,
                                      bottleneck_tensor_bulbophyllum,
                                      bottleneck_tensor_dendrobium,
                                      bottleneck_tensor_paphiopedilum)

    with tf.Session(graph=graph) as sess:
      jpeg_data_tensor, decoded_image_tensor = add_jpeg_decoding(
        model_info['input_width'], model_info['input_height'],
        model_info['input_depth'], model_info['input_mean'],
        model_info['input_std'])

      genus_tensor = tf.squeeze(genus_tensor)
      bulbophyllum_tensor = tf.squeeze(bulbophyllum_tensor)
      dendrobium_tensor = tf.squeeze(dendrobium_tensor)
      paphiopedilum_tensor = tf.squeeze(paphiopedilum_tensor)

      bottleneck_tensor = tf.concat([genus_tensor, bulbophyllum_tensor, dendrobium_tensor, paphiopedilum_tensor], axis=0)

      train_bottlenecks, train_ground_truth, train_filenames = (get_random_cached_bottlenecks(
          sess, all_image_lists, FLAGS.train_batch_size, 'training',
          FLAGS.bottleneck_dir, FLAGS.image_dir, jpeg_data_tensor,
          decoded_image_tensor, resized_input_tensor, bottleneck_tensor, 'final_input'))

      train_step, \
      cross_entropy, \
      bottleneck_input, \
      ground_truth_input, \
      final_tensor = add_final_training_svm(len(all_image_lists.keys()),
                                            'final_result',
                                            bottleneck_tensor,
                                            bottleneck_tensor_size=25)

      predicted_class = tf.sign(final_tensor);
      evaluation_step, prediction = add_evaluation_step('final_accuracy', predicted_class, ground_truth_input)

      with tf.Session() as sess:
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/final_train', sess.graph)
        validation_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/final_validation')

        init = tf.global_variables_initializer()
        sess.run(init)

        # Iterate and train.
        for step in range(FLAGS.how_many_training_steps):
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
                    sess, all_image_lists, FLAGS.validation_batch_size, 'validation',
                    FLAGS.bottleneck_dir, FLAGS.image_dir, jpeg_data_tensor,
                    decoded_image_tensor, resized_input_tensor, bottleneck_tensor, 'final_input'))

            validation_summary, validation_accuracy = sess.run(
              [merged, evaluation_step],
              feed_dict={bottleneck_input: validation_bottlenecks,
                         ground_truth_input: validation_ground_truth})
            validation_writer.add_summary(validation_summary, step)

            tf.logging.info('%s: Step %d: Validation accuracy = %.1f%% (N=%d)' %
                            (datetime.now(), step, validation_accuracy * 100,
                             len(validation_bottlenecks)))

            test_bottlenecks, test_ground_truth, test_filenames = (
              get_random_cached_bottlenecks(
                sess, all_image_lists, FLAGS.test_batch_size, 'testing',
                FLAGS.bottleneck_dir, FLAGS.image_dir, jpeg_data_tensor,
                decoded_image_tensor, resized_input_tensor, bottleneck_tensor))

            test_accuracy, predictions = sess.run(
              [evaluation_step, prediction],
              feed_dict={bottleneck_input: test_bottlenecks, ground_truth_input: test_ground_truth})
            tf.logging.info('Final test accuracy = %.1f%% (N=%d)' % (test_accuracy * 100, len(test_bottlenecks)))

            if FLAGS.print_misclassified_test_images:
              tf.logging.info('=== MISCLASSIFIED TEST IMAGES ===')
              for i, test_filename in enumerate(test_filenames):
                if predictions[i] != test_ground_truth[i].argmax():
                  tf.logging.info('%70s  %s' % (test_filename, list(all_image_lists.keys())[predictions[i]]))

            with gfile.FastGFile('/Volumes/Data/_Corpus-data/orchid_final/models/final_labels.txt', 'w') as f:
              f.write('\n'.join(all_image_lists.keys()) + '\n')

            save_graph_to_file(sess,
                               graph,
                               os.path.join(FLAGS.output_graph, 'final_output_graph.pb'),
                               'final_result')

  if FLAGS.running_method == 'test_all':
    workspace = '/Volumes/Data/_Corpus-data/orchid_final'
    genus_labels = load_labels("{0}/models/{1}_labels.txt".format(workspace, 'genus'))
    bulbophyllum_labels = load_labels("{0}/models/{1}_labels.txt".format(workspace, 'bulbophyllum'))
    dendrobium_labels = load_labels("{0}/models/{1}_labels.txt".format(workspace, 'dendrobium'))
    paphiopedilum_labels = load_labels("{0}/models/{1}_labels.txt".format(workspace, 'paphiopedilum'))

    bottleneck_tensor_genus = final_result_genus + ':0'
    bottleneck_tensor_bulbophyllum = final_result_bulbophyllum + ':0'
    bottleneck_tensor_dendrobium = final_result_dendrobium + ':0'
    bottleneck_tensor_paphiopedilum = final_result_paphiopedilum + ':0'

    graph, \
    resized_input_tensor, \
    genus_tensor, \
    bulbophyllum_tensor, \
    dendrobium_tensor, \
    paphiopedilum_tensor = load_graph(os.path.join(FLAGS.output_graph_path, 'paphiopedilum_output_graph.pb'),
                                      model_info['resized_input_tensor_name'],
                                      bottleneck_tensor_genus,
                                      bottleneck_tensor_bulbophyllum,
                                      bottleneck_tensor_dendrobium,
                                      bottleneck_tensor_paphiopedilum)

    with tf.Session(graph=graph) as sess:
      # Set up the image decoding sub-graph.
      jpeg_data_tensor, decoded_image_tensor = add_jpeg_decoding(
        model_info['input_width'], model_info['input_height'],
        model_info['input_depth'], model_info['input_mean'],
        model_info['input_std'])

      genus_finals, test_ground_truth, test_filenames = (get_random_cached_bottlenecks(
          sess, all_image_lists, FLAGS.test_batch_size, 'testing',
          FLAGS.bottleneck_dir, FLAGS.image_dir, jpeg_data_tensor,
          decoded_image_tensor, resized_input_tensor, genus_tensor, 'genus'))


      lable_max = [len(paphiopedilum_labels), len(bulbophyllum_labels), len(dendrobium_labels)]
      max = np.amax(lable_max, axis=0)

      paphiopedilum_labels = extend_arr(paphiopedilum_labels, max)
      bulbophyllum_labels = extend_arr(bulbophyllum_labels, max)
      dendrobium_labels = extend_arr(dendrobium_labels, max)

      genus_final_input = tf.placeholder(tf.float32, shape=(None, len(genus_labels)))

      c1 = tf.equal(tf.argmax(genus_final_input, 0), 0)
      c2 = tf.equal(tf.argmax(genus_final_input, 0), 1)
      c3 = tf.equal(tf.argmax(genus_final_input, 0), 2)
      a1 = lambda: (paphiopedilum_tensor, paphiopedilum_labels)
      a2 = lambda: (bulbophyllum_tensor, bulbophyllum_labels)
      a3 = lambda: (dendrobium_tensor, dendrobium_labels)
      final_sensor, labels = tf.case([(c1, a1), (c2, a2), (c3, a3)], default=a1)

      #results = sess.run(final_sensor, {resized_input_tensor: resized_input_values})

      print (genus_finals)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--summaries_dir',
      type=str,
      default='/Volumes/Data/_Corpus-data/orchid_final/retrain_logs',
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
      default='/Volumes/Data/_Corpus-data/orchid_final/models',
      help="""\
      Path to classify_image_graph_def.pb,
      imagenet_synset_to_human_label_map.txt, and
      imagenet_2012_challenge_label_map_proto.pbtxt.\
      """
  )
  parser.add_argument(
      '--image_dir',
      type=str,
      default='/Volumes/Data/_Corpus-data/orchid_final/flower_photos',
      help='Path to folders of labeled images.'
  )
  parser.add_argument(
      '--testing_percentage',
      type=int,
      default=10,
      help='What percentage of images to use as a test set.'
  )
  parser.add_argument(
      '--validation_percentage',
      type=int,
      default=10,
      help='What percentage of images to use as a validation set.'
  )
  parser.add_argument(
      '--bottleneck_dir',
      type=str,
      default='/Volumes/Data/_Corpus-data/orchid_final/bottleneck',
      help='Path to cache bottleneck layer values as files.'
  )
  parser.add_argument(
      '--how_many_training_steps',
      type=int,
      default=20000,
      help='How many training steps to run before ending.'
  )
  parser.add_argument(
      '--learning_rate',
      type=float,
      default=0.001,
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
      default=True,
      help="""\
      Whether to print out a list of all misclassified test images.\
      """,
      action='store_true'
  )
  parser.add_argument(
      '--intermediate_output_graphs_dir',
      type=str,
      default='/Volumes/Data/_Corpus-data/orchid_final/intermediate_graph/',
      help='Where to save the intermediate graphs.'
  )
  parser.add_argument(
      '--output_graph_path',
      type=str,
      default='/Volumes/Data/_Corpus-data/orchid_final/models',
      help='Where to save the trained graph.'
  )
  parser.add_argument(
      '--svmC',
      type=int,
      default=1,
      help='The C parameter of the SVM cost function.'
  )
  parser.add_argument(
      '--running_method',
      type=str,
      #default='all_train',
      #default='test_all',
      default='train_final',
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
