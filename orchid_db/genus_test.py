# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random

import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile


def create_image_lists(image_dir, testing_percentage, validation_percentage):
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


def ensure_dir_exists(dir_name):
  if not os.path.exists(dir_name):
    os.makedirs(dir_name)


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
      if unused_i == 4:
        unused_i = unused_i
      label_index = random.randrange(class_count)
      label_name = list(image_lists.keys())[label_index]
      image_index = random.randrange(256 + 1)
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


def load_labels(label_file):
  label = []
  proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
  for l in proto_as_ascii_lines:
    label.append(l.rstrip())
  return label


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


running_method = "accuracy"

if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)

  if running_method == 'accuracy':
    final_result_sensor = 'final_result_genus:0'
    model_dir = '/Volumes/Data/_Corpus-data/models'
    labels = load_labels("{0}/genus_output_label.txt".format(model_dir))
    input_graph = os.path.join(model_dir, '{0}_output_graph.pb'.format('genus'))
    resized_input_tensor_name = 'Mul:0'
    image_dir = '/Volumes/Data/_Corpus-data/orchid-genus/flower_photos/'
    bottleneck_dir = '/Volumes/Data/_Corpus-data/orchid-genus/bottleneck'

    image_lists = create_image_lists(image_dir, testing_percentage=1, validation_percentage=0)

    graph, \
    resized_input_tensor, \
    final_result_tensor = load_test_graph(os.path.join(model_dir, input_graph),
                                          [resized_input_tensor_name, final_result_sensor])

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

      jpeg_data_tensor, decoded_image_tensor = add_jpeg_decoding(299, 299, 3, 128, 128)

      results, test_ground_truth, test_filenames = (get_random_cached_bottlenecks(
          sess, image_lists, -1, 'testing', bottleneck_dir, image_dir, jpeg_data_tensor,
          decoded_image_tensor, resized_input_tensor, final_result_tensor, 'test'))

      test_accuracy, predictions = sess.run([evaluation, prediction],
        feed_dict={results_input: results, ground_truth_input: test_ground_truth})

      tf.logging.info('Final test accuracy = %.1f%% (N=%d)' % (test_accuracy * 100, len(results)))
      tf.logging.info('=== MISCLASSIFIED TEST IMAGES ===')
      misc = 0
      for i, test_filename in enumerate(test_filenames):
        if predictions[i] != test_ground_truth[i].argmax():
          misc += 1
          tf.logging.info('{0:70s}  {0}'.format(test_filename, list(image_lists.keys())[predictions[i]]))
      tf.logging.info('Misclassified number: {0}/{1} images'.format(misc, len(results)))

