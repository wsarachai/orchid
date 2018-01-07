#!/usr/local/bin/python
# -*- coding: utf-8 -*-

import os
import random
import numpy as np
import utils
import constants
import cPickle

from tensorflow.python.platform import gfile

from sklearn import svm
from sklearn import metrics


def get_file_list(image_data):
  if not gfile.Exists(image_data):
    exit(1)

  file_glob = os.path.join(image_data, '*.txt')
  return gfile.Glob(file_glob)


def main():
  info = constants.info
  input_data = os.path.join(constants.image_dir, info["data_type"], "closed")

  tmp_dirs = [x[0] for x in gfile.Walk(input_data)]
  sub_dirs = []

  is_root_dir = True
  for sub_dir in tmp_dirs:
    if is_root_dir:
      is_root_dir = False
      continue
    sub_dirs.append(sub_dir)

  configs = []
  models = []

  sub_dirs = np.array(sub_dirs)
  for i, _train_dir in enumerate(sub_dirs):
    bc = sub_dirs != _train_dir
    _test_dirs = sub_dirs[bc]

    closed_files = get_file_list(os.path.join(input_data, _train_dir))

    file_size = len(closed_files)
    train_size = int(file_size * 0.8)

    training_set = []
    testing_set = []
    open_set = []

    random.shuffle(closed_files)
    for index, filename in enumerate(closed_files):
      if index < train_size:
        training_set.append(utils.load_data_lines(filename)[0])
      else:
        testing_set.append(utils.load_data_lines(filename)[0])

    for _test_dir in _test_dirs:
      closed_files = get_file_list(os.path.join(input_data, _test_dir))
      for _, filename in enumerate(closed_files):
        open_set.append(utils.load_data_lines(filename)[0])

    _data_set = {
      'train': training_set,
      'test': testing_set,
      'open': open_set
    }

    configs.append(_data_set)
    models.append(svm.OneClassSVM(nu=0.16, kernel='rbf', gamma=0.00005))

  for i, config in enumerate(configs):
    model = models[i]
    for _ in xrange(10000):
      model.fit(config['train'])

    train_preds = model.predict(config['train'])
    train_targs = np.ones(len(config['train']))
    test_preds = model.predict(config['test'])
    test_targs = np.ones(len(config['test']))
    open_preds = model.predict(config['open'])
    open_targs = np.zeros(len(config['open'])) - 1

    _data_set['train_accuracy'] = metrics.accuracy_score(train_targs, train_preds)
    _data_set['test_accuracy'] = metrics.accuracy_score(test_targs, test_preds)
    _data_set['open_accuracy'] = metrics.accuracy_score(open_targs, open_preds)

    print("\nTraining: {0}".format(i+1))
    print("Training accuracy: ", _data_set['train_accuracy'])
    print("Testing accuracy: ", _data_set['test_accuracy'])
    print("Open accuracy: ", _data_set['open_accuracy'])

  with open('/tmp/oneclass.pkl', 'wb') as fid:
    cPickle.dump(models, fid)


def test_open():
  info = constants.info
  open_data = os.path.join(constants.image_dir, info["data_type"], "open")
  closed_data = os.path.join(constants.image_dir, info["data_type"], "closed")

  open_set = []
  closed_set = []
  open_dirs = []
  closed_dirs = []
  tmp_open_dirs = [x[0] for x in gfile.Walk(open_data)]
  tmp_closed_dirs = [x[0] for x in gfile.Walk(closed_data)]

  is_root_dir = True
  for sub_dir in tmp_open_dirs:
    if is_root_dir:
      is_root_dir = False
      continue
    open_dirs.append(sub_dir)

  is_root_dir = True
  for sub_dir in tmp_closed_dirs:
    if is_root_dir:
      is_root_dir = False
      continue
    closed_dirs.append(sub_dir)

  for _open_dir in open_dirs:
    open_files = get_file_list(os.path.join(open_data, _open_dir))
    for _, filename in enumerate(open_files):
      open_set.append(utils.load_data_lines(filename)[0])

  for _closed_dir in closed_dirs:
    loaded_sets = []
    closed_files = get_file_list(os.path.join(open_data, _closed_dir))
    for _, filename in enumerate(closed_files):
      loaded_sets.append(utils.load_data_lines(filename)[0])
    closed_set.append(loaded_sets)

  with open('/Volumes/Data/_dataset/models/oneclass/oneclass.pkl', 'rb') as fid:
    models = cPickle.load(fid)

  for model in models:
    preds = model.predict(open_set)
    targs = np.zeros(len(open_set)) - 1
    print ("Open accuracy : ", metrics.accuracy_score(targs, preds))

  for i, model in enumerate(models):
    accs = []
    for closed in closed_set:
      preds = model.predict(closed)
      targs = np.ones(len(closed))
      accs.append(metrics.accuracy_score(targs, preds))
    imax = np.argmax(accs)
    print ("Closed accuracy model({0})-data({1}): {2}".format(i, imax, accs[imax]) )

if __name__ == '__main__':
  #main()
  test_open()
  print ("Done.")
  exit()