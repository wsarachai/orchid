from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile

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

def read_tensor_from_image_file(file_name, input_height=299, input_width=299, input_mean=0, input_std=255):
  input_name = "file_reader"
  output_name = "normalized"
  file_reader = tf.read_file(file_name, input_name)
  if file_name.endswith(".png"):
    image_reader = tf.image.decode_png(file_reader, channels=3, name='png_reader')
  elif file_name.endswith(".gif"):
    image_reader = tf.squeeze(tf.image.decode_gif(file_reader, name='gif_reader'))
  elif file_name.endswith(".bmp"):
    image_reader = tf.image.decode_bmp(file_reader, name='bmp_reader')
  else:
    image_reader = tf.image.decode_jpeg(file_reader, channels=3, name='jpeg_reader')
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


def extend_arr(list, n):
  return (list + [''] * (n - len(list)))[:n]


if __name__ == "__main__":
  #file_name = "/Volumes/Data/_Corpus-data/orchid-bulbophyllum_001/flower_photos/dayanum Rchb.f./dayanum-rchb-f-_002.jpg"
  #file_name = "/Volumes/Data/_Corpus-data/orchid-bulbophyllum_001/flower_photos/auricomum Lindl./auricomum-lindl-_038.jpg"
  file_name = "/Volumes/Data/_Corpus-data/orchid-bulbophyllum_001/flower_photos/lasiochilum Par. & Rchb.f./lasiochilum-par-rchb-f-_013.jpg"

  #model_file = "/Volumes/Data/_Corpus-data/orchid-paphiopedilum_001/output_graph.pb"
  model_file = "/Volumes/Data/_Corpus-data/orchid_final/output_graph.pb"
  input_height = 299
  input_width = 299
  input_mean = 128
  input_std = 128
  input_layer = "Mul:0"

  genus_label_file = "/Volumes/Data/_Corpus-data/orchid-genus/output_labels.txt"
  bulbophyllum_label_file = "/Volumes/Data/_Corpus-data/orchid-bulbophyllum_001/output_labels.txt"
  dendrobium_label_file = "/Volumes/Data/_Corpus-data/orchid-dendrobium_001/output_labels.txt"
  paphiopedilum_label_file = "/Volumes/Data/_Corpus-data/orchid-paphiopedilum_001/output_labels.txt"

  bottleneck_tensor_genus_size = 3
  bottleneck_tensor_species_size = 11
  bottleneck_tensor_species_size = 21
  bottleneck_tensor_species_size = 5

  bottleneck_tensor_genus = 'final_result_genus:0'
  bottleneck_tensor_paphiopedilum = 'final_result_paphiopedilum:0'
  bottleneck_tensor_bulbophyllum = 'final_result_bulbophyllum:0'
  bottleneck_tensor_dendrobium = 'final_result_dendrobium:0'

  graph,\
  resized_input_tensor,\
  genus_tensor,\
  bulbophyllum_tensor, \
  dendrobium_tensor, \
  paphiopedilum_tensor = load_graph(model_file,
                                    input_layer,
                                    bottleneck_tensor_genus,
                                    bottleneck_tensor_bulbophyllum,
                                    bottleneck_tensor_dendrobium,
                                    bottleneck_tensor_paphiopedilum)

  resized_input_values = read_tensor_from_image_file(file_name)

  with tf.Session(graph=graph) as sess:
    results = sess.run(genus_tensor, {resized_input_tensor: resized_input_values})
    results = np.squeeze(results)

    genus_labels = load_labels(genus_label_file)
    paphiopedilum_labels = load_labels(paphiopedilum_label_file)
    bulbophyllum_labels = load_labels(bulbophyllum_label_file)
    dendrobium_labels = load_labels(dendrobium_label_file)

    lable_max = [len(paphiopedilum_labels), len(bulbophyllum_labels), len(dendrobium_labels)]
    max = np.amax(lable_max, axis=0)

    paphiopedilum_labels = extend_arr(paphiopedilum_labels, max)
    bulbophyllum_labels = extend_arr(bulbophyllum_labels, max)
    dendrobium_labels = extend_arr(dendrobium_labels, max)

    c1 = tf.equal(tf.argmax(results, 0), 0)
    c2 = tf.equal(tf.argmax(results, 0), 1)
    c3 = tf.equal(tf.argmax(results, 0), 2)
    a1 = lambda: (paphiopedilum_tensor, paphiopedilum_labels)
    a2 = lambda: (bulbophyllum_tensor, bulbophyllum_labels)
    a3 = lambda: (dendrobium_tensor, dendrobium_labels)
    final_sensor, labels = tf.case([(c1, a1), (c2, a2), (c3, a3)], default=a1)

    results = sess.run(final_sensor, {resized_input_tensor: resized_input_values})
    results = np.squeeze(results)

    top_k = results.argsort()[::-1]
    for i in top_k:
      print("{0}: {1:.2f}".format(sess.run(labels[i]), results[i]))
