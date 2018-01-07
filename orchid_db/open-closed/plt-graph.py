#!/usr/local/bin/python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import utils
import constants

from tensorflow.python.platform import gfile


def get_data_dimension(data, precision):
  mode_std = round(np.std(data), precision)
  mode_mean = round(np.mean(data), precision)
  mode_max = round(np.max(data), precision)
  mode_min = round(np.min(data), precision)
  diff = mode_max - mode_min
  data.sort()
  data.reverse()
  r1 = round(data[0] - data[1], precision)
  r2 = round(data[0] - data[2], precision)
  return mode_std, mode_mean, mode_max, mode_min, diff, r1, r2


def get_bottleneck_string(mode_std, mode_mean, mode_max, mode_min, diff, r1, r2):
  return '{},{},{},{},{},{}'.format(mode_mean, mode_std, r1, r2, mode_max, diff)


def format_graph_x(mode_std, mode_mean, mode_max, mode_min, diff, r1, r2):
  return [mode_mean, mode_std, r1, r2, mode_max, diff]


def get_graph_param(info):
  if info["data_type"] == "softmax":
    smin = -0.5
    smax = 1.5
    step = 0.2
  else:
    smin = -55
    smax = 55
    step = 5
  return smin, smax, step

def generate_all(info):
  output_data = os.path.join(constants.output_dir, info["data_type"], "dataset")
  image_data = os.path.join(constants.image_dir, info["data_type"],  "closed")

  if not gfile.Exists(output_data):
    os.makedirs(output_data)

  if not gfile.Exists(image_data):
    return None

  line = []
  sub_dirs = [x[0] for x in gfile.Walk(image_data)]

  is_root_dir = True
  for sub_dir in sub_dirs:
    if is_root_dir:
      is_root_dir = False
      continue

    file_list = []
    dir_name = os.path.basename(sub_dir)
    file_glob = os.path.join(image_data, dir_name, '*.txt')
    file_list.extend(gfile.Glob(file_glob))

    for f in file_list:
      with open(f, 'r') as bottleneck_file:
        bottleneck_string = bottleneck_file.read()

        did_hit_error = False
        try:
          bottleneck_values = [round(float(x), 3) for x in bottleneck_string.split(',')]
        except ValueError:
          did_hit_error = True

      filename = os.path.basename(f)

      if info["moment"] == 1:
        mode_std, mode_mean, mode_max, mode_min, diff, r1, r2 = get_data_dimension(bottleneck_values, 3)
        bottleneck_string = get_bottleneck_string(mode_std, mode_mean, mode_max, mode_min, diff, r1, r2)

      line.append(bottleneck_string)
      print (filename)

  if info["moment"] == 1:
    output_filename = 'm1-closed-all.txt'
  else:
    output_filename = 'closed-all.txt'

  with gfile.FastGFile(os.path.join(output_data, output_filename), 'w') as f:
    f.write('\n'.join(line) + '\n')


def generate_all_mean(info):
  output_data = os.path.join(constants.output_dir, info["data_type"], "dataset")
  output_img = os.path.join(constants.output_dir, info["data_type"], "graphs", "closed")
  image_data = os.path.join(constants.image_dir, info["data_type"],  "closed")

  if not gfile.Exists(output_data):
    os.makedirs(output_data)

  if not gfile.Exists(output_img):
    os.makedirs(output_img)

  if not gfile.Exists(image_data):
    return None

  all_data = []
  label_names = []

  sub_dirs = [x[0] for x in gfile.Walk(image_data)]

  is_root_dir = True
  for sub_dir in sub_dirs:
    if is_root_dir:
      is_root_dir = False
      continue

    file_list = []
    dir_name = os.path.basename(sub_dir)
    label_names.append(dir_name)
    file_glob = os.path.join(image_data, dir_name, '*.txt')
    file_list.extend(gfile.Glob(file_glob))

    data = np.zeros((info["n_classes"], len(file_list)))
    fid = 0
    for f in file_list:
      with open(f, 'r') as bottleneck_file:
        bottleneck_string = bottleneck_file.read()

        did_hit_error = False
        try:
          if info["data_type"] == "softmax":
            bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
          else:
            bottleneck_values = [round(float(x), 3) for x in bottleneck_string.split(',')]
        except ValueError:
          did_hit_error = True

        if not did_hit_error:
          for i, v in enumerate(bottleneck_values):
            data[i][fid] = v

        fid += 1

    all_data.append(data)

  line = []

  for i, vv in enumerate(all_data):
    mode = []
    mode_c = []
    mean = []
    min = []
    max = []
    for _, v in enumerate(vv):
      min.append(np.min(v))
      max.append(np.max(v))
      mean.append(np.mean(v))
      mv, mc = stats.mode(v)
      mode.append(mv[0])
      mode_c.append(mc[0])

    _data_to_eva = mode

    if info["moment"] == 1:
      label_name = 'm1_label {}'.format(np.argmax(max) + 1)
    else:
      label_name = 'label {}'.format(np.argmax(max) + 1)

    if info["moment"] == 1:
      mode_std, mode_mean, mode_max, mode_min, diff, r1, r2 = get_data_dimension(_data_to_eva, 3)

    if info["moment"] == 1:
      _data = format_graph_x(mode_std, mode_mean, mode_max, mode_min, diff, r1, r2)
    else:
      _data = _data_to_eva


    if info["print_graph"]:
      if info["moment"] == 1:
        labels = np.arange(1, 7, 1)
      else:
        labels = np.arange(1, info["n_classes"] + 1, 1)

      smin, smax, step = get_graph_param(info)

      #fig, axs = plt.subplots(2, 1, sharex=True)
      fig, axs = plt.subplots(1, 1, sharex=True)
      #fig.subplots_adjust(hspace=1)
      fig.suptitle(label_name, fontsize=20)

      if info["moment"] == 1:
        axs.plot(labels, _data, 'o--')
      else:
        axs.plot(labels, min, 'r--', labels, max, 'r--', labels, _data, 'o-')
      axs.set_yticks(np.arange(smin, smax, step))
      axs.set_ylim(smin, smax)
      #axs.set_title('mean={}, r1={}, mode={}, r2={}, mode={}, diff={}'.format(mode_mean, r1, mode_std, r2, mode_max, diff))

      #axs[1].plot(labels, mode_c, '-o')
      #axs[1].set_yticks(np.arange(0, 3, 1))
      #axs[1].set_ylim(0, 5)
      #axs[1].set_title('mode num')

      #plt.show()
      fig.savefig('{}/{}.jpg'.format(output_img, label_name))

    if info["moment"] == 1:
      line.append(get_bottleneck_string(mode_std, mode_mean, mode_max, mode_min, diff, r1, r2))
    else:
      line.append(','.join(str(x) for x in _data))

    print(label_name)

  if info["moment"] == 1:
    output_filename = 'm1-closed.txt'
  else:
    output_filename = 'closed.txt'

  with gfile.FastGFile(os.path.join(output_data, output_filename), 'w') as f:
    f.write('\n'.join(line) + '\n')


def open_labels(info):
  output_data = os.path.join(constants.output_dir, info["data_type"], "dataset")
  output_img = os.path.join(constants.output_dir, info["data_type"], "graphs", "open")
  image_data = os.path.join(constants.image_dir, info["data_type"],  "open")

  if not gfile.Exists(output_data):
    os.makedirs(output_data)

  if not gfile.Exists(output_img):
    os.makedirs(output_img)

  file_list = []
  sub_dirs = [x[0] for x in gfile.Walk(image_data)]

  is_root_dir = True
  for sub_dir in sub_dirs:
    if is_root_dir:
      is_root_dir = False
      continue

    dir_name = os.path.basename(sub_dir)
    file_glob = os.path.join(image_data, dir_name, '*.txt')
    file_list.extend(gfile.Glob(file_glob))

  line = []
  c = 0
  graph_limit = 30

  for f in file_list:
    with open(f, 'r') as bottleneck_file:
      bottleneck_string = bottleneck_file.read()

      did_hit_error = False
      try:
        if info["data_type"] == "softmax":
          bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
        else:
          bottleneck_values = [round(float(x), 3) for x in bottleneck_string.split(',')]
      except ValueError:
        did_hit_error = True

      if info["moment"] == 1:
        mode_std, mode_mean, mode_max, mode_min, diff, r1, r2 = get_data_dimension(bottleneck_values, 3)

      filename = os.path.basename(f)

      if info["moment"] == 1:
        label_name = 'm1_{}'.format(filename)
      else:
        label_name = filename

      if info["moment"] == 1:
        _data = format_graph_x(mode_std, mode_mean, mode_max, mode_min, diff, r1, r2)
      else:
        _data = bottleneck_values

      if info["print_graph"] and c < graph_limit:

        if info["moment"] == 1:
          labels = np.arange(1, 7, 1)
        else:
          labels = np.arange(1, info["n_classes"] + 1, 1)

        smin, smax, step = get_graph_param(info)

        fig, axs = plt.subplots(1, 1, sharex=True)
        #fig.suptitle(filename, fontsize=20)
        fig.suptitle('', fontsize=20)

        axs.plot(labels, _data, '--o')
        axs.set_yticks(np.arange(smin, smax, step))
        axs.set_ylim(smin, smax)
        #axs.set_title('mean={}, r1={}, mode={}, r2={}, mode={}, diff={}'.format(mode_mean, r1, mode_std, r2, mode_max, diff))
        fig.savefig('{}/{}.jpg'.format(output_img, label_name))

        c+=1

      if info["moment"] == 1:
        line.append(get_bottleneck_string(mode_std, mode_mean, mode_max, mode_min, diff, r1, r2))
      else:
        line.append(','.join(str(x) for x in _data))

      print (filename)

  if info["moment"] == 1:
    output_filename = 'm1-open.txt'
  else:
    output_filename = 'open.txt'

  with gfile.FastGFile(os.path.join(output_data, output_filename), 'w') as f:
    f.write('\n'.join(line) + '\n')

def main():
  info = constants.info
  generate_all(info)
  generate_all_mean(info)
  open_labels(info)


if __name__ == '__main__':
  main()
  print ("All done.")
  exit()

