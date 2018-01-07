#!/usr/local/bin/python
# -*- coding: utf-8 -*-

import os
import random
import numpy as np
import utils
import constants

from sklearn import svm
from sklearn import metrics


def load_data(info, output_data):
  d_closed_all = utils.load_data_lines(os.path.join(output_data, 'closed-all.txt'))
  d_closed = utils.load_data_lines(os.path.join(output_data, 'closed.txt'))
  d_open = utils.load_data_lines(os.path.join(output_data, 'open.txt'))

  return d_closed_all, d_closed, d_open


def main():
  info = constants.info
  output_data = os.path.join(constants.output_dir, info["data_type"], "dataset")

  d_closed_all, d_closed, d_outliers = load_data(info, output_data)

  data_all = []
  data_all.extend(d_closed_all)
  data_all.extend(d_closed)

  random.shuffle(data_all)
  random.shuffle(d_outliers)

  data_size = len(data_all)
  train_size = int(data_size * 0.8)

  train_data = []
  test_data = []
  outliers_data = []
  train_labels = []
  test_labels = []
  outliers_labels = []

  for i, bottleneck_values in enumerate(data_all):
    if i < train_size:
      train_data.append(bottleneck_values[:-1])
      train_labels.append(bottleneck_values[-1:][0])
    else:
      test_data.append(bottleneck_values[:-1])
      test_labels.append(bottleneck_values[-1:][0])

  for i, bottleneck_values in enumerate(d_outliers):
    outliers_data.append(bottleneck_values[:-1])
    outliers_labels.append(bottleneck_values[-1:][0])

  model = svm.OneClassSVM(nu=0.55, kernel='rbf', gamma=0.00005)
  for _ in xrange(500):
    model.fit(train_data)

  preds = model.predict(train_data)
  targs = np.ones(len(train_data))

  print("Training...")
  print("accuracy: ", metrics.accuracy_score(targs, preds))
  #print("precision: ", metrics.precision_score(targs, preds))
  #print("recall: ", metrics.recall_score(targs, preds))
  #print("f1: ", metrics.f1_score(targs, preds))

  preds = model.predict(test_data)
  targs = np.ones(len(test_data))

  print("Testing...")
  print("accuracy: ", metrics.accuracy_score(targs, preds))
  #print("precision: ", metrics.precision_score(targs, preds))
  #print("recall: ", metrics.recall_score(targs, preds))
  #print("f1: ", metrics.f1_score(targs, preds))

  preds = model.predict(outliers_data)
  targs = np.zeros(len(outliers_data)) - 1

  print("Open...")
  print("accuracy: ", metrics.accuracy_score(targs, preds))
  #print("precision: ", metrics.precision_score(targs, preds))
  #print("recall: ", metrics.recall_score(targs, preds))
  #print("f1: ", metrics.f1_score(targs, preds))


if __name__ == '__main__':
  main()
  print ("Done.")
  exit()