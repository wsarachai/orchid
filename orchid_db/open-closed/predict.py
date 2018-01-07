#!/usr/local/bin/python
# -*- coding: utf-8 -*-

import os
import numpy as np
import utils
import constants


def KLDivergence(p, q):
  _p = utils.scale_one(p) + 0.001
  _q = utils.scale_one(q) + 0.001
  div = np.divide(_q, _p)
  log = np.log(div)
  mul = np.multiply(_p, log)
  sum = np.sum(mul)
  Dlk = np.multiply(sum, -1)
  return Dlk


def load_data(info, output_data):
  if (info["moment"] == 1):
    d_closed_all = utils.load_data_lines(os.path.join(output_data, 'm1-closed-all.txt'))
    d_closed = utils.load_data_lines(os.path.join(output_data, 'm1-closed.txt'))
    d_open = utils.load_data_lines(os.path.join(output_data, 'm1-open.txt'))
  else:
    d_closed_all = utils.load_data_lines(os.path.join(output_data, 'closed-all.txt'))
    d_closed = utils.load_data_lines(os.path.join(output_data, 'closed.txt'))
    d_open = utils.load_data_lines(os.path.join(output_data, 'open.txt'))

  return d_closed_all, d_closed, d_open


def main():
  info = constants.info
  output_data = os.path.join(constants.output_dir, info["data_type"], "dataset")
  d_closed_all, d_closed, d_open = load_data(info, output_data)

  c1 = 0.0
  for i, all in enumerate(d_closed_all):
    result = []
    for closed in d_closed:
      result.append(round(KLDivergence(closed, all), 4))
    result.sort()
    print ("{}".format(result[0]))
    if (result[0] < info["threshold"]):
      c1 += 1

  c2 = 0.0
  for i, all in enumerate(d_open):
    result = []
    for closed in d_closed:
      result.append(round(KLDivergence(closed, all), 4))
    result.sort()
    print ("{}".format(result[0]))
    if (result[0] >= info["threshold"]):
      c2 += 1

  print ("Closed dataset {0}/{1} accuracy is: {2:.2f}".format(c1, len(d_closed_all), c1/len(d_closed_all)*100.0))
  print ("Open dataset {0}/{1} accuracy is: {2:.2f}".format(c2, len(d_open), c2/len(d_open)*100.0))


if __name__ == '__main__':
  main()
  exit()