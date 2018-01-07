#!/usr/local/bin/python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
import numpy as np
from sklearn import svm

from tensorflow.python.platform import gfile

import utils

def load_data():
  data_all = []
  d_closed_all = utils.load_data_lines('closed-all.txt')
  d_closed = utils.load_data_lines('closed.txt')
  d_open = utils.load_data_lines('open.txt')

  data_all.extend(d_closed_all)
  data_all.extend(d_closed)
  data_all.extend(d_open)

  random.shuffle(data_all)
  return data_all


def main():
  data_all = load_data()

  data_size = len(data_all)
  train_size = int(data_size * 0.8)

  train_data = []
  test_data = []
  train_labels = []
  test_labels = []
  for i,bottleneck_string in enumerate(data_all):
    did_hit_error = False
    try:
      bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
    except ValueError:
      did_hit_error = True

    if not did_hit_error:
      if i < train_size:
        train_data.append(bottleneck_values[:-1])
        train_labels.append(bottleneck_values[-1:][0])
      else:
        test_data.append(bottleneck_values[:-1])
        test_labels.append(bottleneck_values[-1:][0])

  clf = svm.SVC()
  for _ in xrange(10):
    clf.fit(train_data, train_labels)

  predit = clf.predict(test_data)
  res = predit == test_labels
  print ("Accuracy: {0:.2f}%".format(np.sum(res)/len(res)*100))


if __name__ == '__main__':
  main()
  print ("Done.")
  exit()