from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from datasets import convert_genus_flowers

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'dataset_name',
    None,
    'The name of the dataset to convert, one of "genus", "bulbophyllum", "dendrobium", "paphiopedilum".')

tf.app.flags.DEFINE_string(
    'dataset_dir',
    None,
    'The directory where the output TFRecords and temporary files are saved.')


def main(_):
  if not FLAGS.dataset_name:
    raise ValueError('You must supply the dataset name with --dataset_name')
  if not FLAGS.dataset_dir:
    raise ValueError('You must supply the dataset directory with --dataset_dir')

  if FLAGS.dataset_name == 'genus':
    convert_genus_flowers.run(FLAGS.dataset_dir)
  else:
    raise ValueError(
        'dataset_name [%s] was not recognized.' % FLAGS.dataset_name)

if __name__ == '__main__':
  tf.app.run()
