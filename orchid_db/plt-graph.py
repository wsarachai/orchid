#!/usr/local/bin/python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import numpy as np
import tensorflow as tf
from scipy import stats
import matplotlib.pyplot as plt

from tensorflow.python.platform import gfile

image_dir = '/Volumes/Data/_Corpus-data/orchid-final/bottleneck_graph'

print_graph = False


def remove_non_ascii(text):
  return re.sub(r'[^\x00-\x7f]', r' ', text)


def load_labels(label_file):
  label = []
  proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
  for l in proto_as_ascii_lines:
    label.append(l.rstrip())
  return label


def get_data_dimension(data, precision):
  mode_std = round(np.std(data), precision)
  mode_mean = round(np.mean(data), precision)
  mode_max = round(np.max(data), precision)
  mode_min = round(np.min(data), precision)
  diff = mode_max - mode_min
  data.sort()
  data.reverse()
  r1 = data[0] - data[1]
  r2 = data[0] - data[2]
  return mode_std, mode_mean, mode_max, mode_min, diff, r1, r2


def generate_all():
  if not gfile.Exists(image_dir):
    return None

  line = []
  sub_dirs = [x[0] for x in gfile.Walk(image_dir)]

  is_root_dir = True
  for sub_dir in sub_dirs:
    if is_root_dir:
      is_root_dir = False
      continue

    data = []
    file_list = []
    dir_name = os.path.basename(sub_dir)
    file_glob = os.path.join(image_dir, dir_name, '*.txt')
    file_list.extend(gfile.Glob(file_glob))

    for f in file_list:
      with open(f, 'r') as bottleneck_file:
        bottleneck_string = bottleneck_file.read()

        did_hit_error = False
        try:
          bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
        except ValueError:
          did_hit_error = True

        if not did_hit_error:
          for i, v in enumerate(bottleneck_values):
            data.append(v)

      mode_std, mode_mean, mode_max, mode_min, diff, r1, r2 = get_data_dimension(data, 3)

      filename = os.path.basename(f)

      line.append('{},{},{},{},{},{},{}'.format(mode_max, mode_std, mode_mean, diff, r1, r2, 1))

      print (filename)

  with gfile.FastGFile('/tmp/closed-all.txt', 'w') as f:
    f.write('\n'.join(line) + '\n')


def generate_all_mean():
  model_dir = '/Volumes/Data/_Corpus-data/models'
  ord_labels = load_labels("{0}/{1}_output_label.txt".format(model_dir, 'final'))

  if not gfile.Exists(image_dir):
    return None

  all_data = []
  label_names = []

  sub_dirs = [x[0] for x in gfile.Walk(image_dir)]

  is_root_dir = True
  for sub_dir in sub_dirs:
    if is_root_dir:
      is_root_dir = False
      continue

    file_list = []
    dir_name = os.path.basename(sub_dir)
    label_names.append(dir_name)
    file_glob = os.path.join(image_dir, dir_name, '*.txt')
    file_list.extend(gfile.Glob(file_glob))

    data = np.zeros((24, len(file_list)))
    fid = 0
    for f in file_list:
      with open(f, 'r') as bottleneck_file:
        bottleneck_string = bottleneck_file.read()

        did_hit_error = False
        try:
          bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
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
      min.append(round(np.min(v), 3))
      max.append(round(np.max(v), 3))
      mean.append(round(np.mean(v), 3))
      vmode = []
      for vm in v:
        vmode.append(round(vm))
      mv, mc = stats.mode(vmode)
      mode.append(round(mv[0]))
      mode_c.append(mc[0])

    mode_std, mode_mean, mode_max, mode_min, diff, r1, r2 = get_data_dimension(mode, 2)

    label_name = ""
    unicode_name = ""
    for _, name in enumerate(ord_labels):
      name1 = remove_non_ascii(name).lower()
      name2 = remove_non_ascii(label_names[i]).lower()
      if (name1 == name2):
        label_name = 'label {}'.format(_ + 1)
        unicode_name = name
        break

    if print_graph:
      labels = np.arange(1, 25, 1)

      smin = -40
      smax = 40

      #fig, axs = plt.subplots(2, 1, sharex=True)
      fig, axs = plt.subplots(1, 1, sharex=True)
      #fig.subplots_adjust(hspace=1)
      fig.suptitle(label_name, fontsize=20)

      axs.plot(labels, min, 'r--', labels, max, 'r--', labels, mode, '-o')
      axs.set_yticks(np.arange(smin, smax, 5))
      axs.set_ylim(smin, smax)
      axs.set_title('max={}, std={}, mean={}, diff={}, rank={},{}'.format(mode_max, mode_std, mode_mean, diff, r1, r2))

      #axs[1].plot(labels, mode_c, '-o')
      #axs[1].set_yticks(np.arange(0, 3, 1))
      #axs[1].set_ylim(0, 5)
      #axs[1].set_title('mode num')

      #plt.show()
      fig.savefig('/tmp/{}.jpg'.format(unicode_name))

    line.append('{},{},{},{},{},{},{}'.format(mode_max, mode_std, mode_mean, diff, r1, r2, 1))
    print(label_name)

  with gfile.FastGFile('/tmp/closed.txt', 'w') as f:
    f.write('\n'.join(line) + '\n')

  print("Done.")


def open_labels():
  image_dir = '/Volumes/Data/_Corpus-data/17Flowers/bottleneck'

  file_list = []
  file_glob = os.path.join(image_dir, 'images', '*.txt')
  file_list.extend(gfile.Glob(file_glob))

  line = []

  for f in file_list:
    data = []
    with open(f, 'r') as bottleneck_file:
      bottleneck_string = bottleneck_file.read()

      did_hit_error = False
      try:
        bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
      except ValueError:
        did_hit_error = True

      if not did_hit_error:
        for i, v in enumerate(bottleneck_values):
          data.append(v)

      mode_std, mode_mean, mode_max, mode_min, diff, r1, r2 = get_data_dimension(data, 3)

      filename = os.path.basename(f)

      if print_graph:
        labels = np.arange(1, 25, 1)

        fig, axs = plt.subplots(1, 1, sharex=True)
        #fig.suptitle(filename, fontsize=20)
        fig.suptitle('', fontsize=20)

        smin = -40
        smax = 40
        axs.plot(labels, data, '-o')
        axs.set_yticks(np.arange(smin, smax, 5))
        axs.set_ylim(smin, smax)
        axs.set_title('max={}, std={}, mean={}, diff={}, rank={},{}'.format(mode_max, mode_std, mode_mean, diff, r1, r2))
        fig.savefig('/tmp/zopen/{}.jpg'.format(filename))

      line.append('{},{},{},{},{},{},{}'.format(mode_max, mode_std, mode_mean, diff, r1, r2, -1))

      print (filename)

  with gfile.FastGFile('/tmp/open.txt', 'w') as f:
    f.write('\n'.join(line) + '\n')

def main():
  generate_all()
  generate_all_mean()
  open_labels()


if __name__ == '__main__':
  main()

