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

import argparse
import sys

import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile

def load_graph(model_path, input_name, output_name):
  with tf.Graph().as_default() as graph:
    with gfile.FastGFile(model_path, 'rb') as f:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(f.read())
      final_tensor, resized_input_tensor = (tf.import_graph_def(
        graph_def,
        name='',
        return_elements=[
          output_name,
          input_name,
        ]))
  return graph, final_tensor, resized_input_tensor

def read_tensor_from_image_file(file_name, input_height=299, input_width=299, input_mean=0, input_std=255):
  input_name = "file_reader"
  file_reader = tf.read_file(file_name, input_name)
  if file_name.endswith(".png"):
    image_reader = tf.image.decode_png(file_reader, channels = 3, name='png_reader')
  elif file_name.endswith(".gif"):
    image_reader = tf.squeeze(tf.image.decode_gif(file_reader, name='gif_reader'))
  elif file_name.endswith(".bmp"):
    image_reader = tf.image.decode_bmp(file_reader, name='bmp_reader')
  else:
    image_reader = tf.image.decode_jpeg(file_reader, channels = 3, name='jpeg_reader')
  float_caster = tf.cast(image_reader, tf.float32)
  decoded_image_4d = tf.expand_dims(float_caster, 0);
  resize_shape = tf.stack([input_height, input_width])
  resize_shape_as_int = tf.cast(resize_shape, dtype=tf.int32)
  resized_image = tf.image.resize_bilinear(decoded_image_4d, resize_shape_as_int)
  offset_image = tf.subtract(resized_image, input_mean)
  normalized = tf.multiply(offset_image, 1.0 / input_std)
  sess = tf.Session()
  result = sess.run(normalized)

  return result

def load_labels(label_file):
  label = []
  proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
  for l in proto_as_ascii_lines:
    label.append(l.rstrip())
  return label

if __name__ == "__main__":
  #file_name = "/Volumes/Data/_Corpus-data/orchid-genus/flower_photos/bulbophyllum/bulbophyllum_001.jpg"
  #file_name = "/Volumes/Data/_Corpus-data/orchid-genus/flower_photos/bulbophyllum/bulbophyllum_002.jpg"
  #file_name = "/Volumes/Data/_Corpus-data/orchid-genus/flower_photos/dendrobium/dendrobium_017.jpg"
  #file_name = "/Volumes/Data/_Corpus-data/orchid-genus/flower_photos/paphiopedilum/paphiopedilum_015.jpg"
  file_name = "/Volumes/Data/_Corpus-data/orchid-bulbophyllum_001/flower_photos/lasiochilum Par. & Rchb.f./lasiochilum-par-rchb-f-_013.jpg"

  model_file = "/Volumes/Data/_Corpus-data/orchid-genus/output_graph.pb"
  label_file = "/Volumes/Data/_Corpus-data/orchid-genus/output_labels.txt"
  input_height = 299
  input_width = 299
  input_mean = 128
  input_std = 128
  input_layer = "Mul:0"
  output_layer = "final_result_genus:0"

  parser = argparse.ArgumentParser()
  parser.add_argument("--image", help="image to be processed")
  parser.add_argument("--graph", help="graph/model to be executed")
  parser.add_argument("--labels", help="name of file containing labels")
  parser.add_argument("--input_height", type=int, help="input height")
  parser.add_argument("--input_width", type=int, help="input width")
  parser.add_argument("--input_mean", type=int, help="input mean")
  parser.add_argument("--input_std", type=int, help="input std")
  parser.add_argument("--input_layer", help="name of input layer")
  parser.add_argument("--output_layer", help="name of output layer")
  args = parser.parse_args()

  if args.graph:
    model_file = args.graph
  if args.image:
    file_name = args.image
  if args.labels:
    label_file = args.labels
  if args.input_height:
    input_height = args.input_height
  if args.input_width:
    input_width = args.input_width
  if args.input_mean:
    input_mean = args.input_mean
  if args.input_std:
    input_std = args.input_std
  if args.input_layer:
    input_layer = args.input_layer
  if args.output_layer:
    output_layer = args.output_layer

  graph, final_tensor, resized_input_tensor = load_graph(model_file, input_layer, output_layer)

  #op = graph.get_operations()
  #for m in op:
  #   print (m.values())

  resized_input_values = read_tensor_from_image_file(file_name)

  with tf.Session(graph=graph) as sess:
    results = sess.run(final_tensor, {resized_input_tensor: resized_input_values})
    results = np.squeeze(results)

  top_k = results.argsort()[::-1]
  labels = load_labels(label_file)
  for i in top_k:
    print("{0} {1:.2f}".format(labels[i], results[i]))
