#!/usr/local/bin/python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import re
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile


def load_data_lines(filename):
  label = []
  data_as_ascii_lines = gfile.GFile(filename).readlines()
  for l in data_as_ascii_lines:
    did_hit_error = False
    try:
      l_values = [float(x) for x in l.split(',')]
    except ValueError:
      did_hit_error = True

    if not did_hit_error:
      label.append(l_values)
  return label


def scale_one(v):
  _min = np.min(v)
  _min = np.abs(_min)
  return np.add(v, _min)


def normalize_one(v):
  norm = np.linalg.norm(v, ord=1)
  if norm == 0:
    norm = np.finfo(v.dtype).eps
  return (v/norm)


def scale(min, max, v):
  _min = np.min(v)
  _min = np.abs(_min)
  return np.add(min, _min), np.add(max, _min), np.add(v, _min)


def normalize(min, max, v):
  norm = np.linalg.norm(v, ord=1)
  if norm == 0:
    norm = np.finfo(v.dtype).eps
  return (min/norm), (max/norm), (v/norm)


def remove_non_ascii(text):
  return re.sub(r'[^\x00-\x7f]', r' ', text)


def load_labels(label_file):
  label = []
  proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
  for l in proto_as_ascii_lines:
    label.append(l.rstrip())
  return label