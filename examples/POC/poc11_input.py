import os
import cv2
import math
import numpy as np
from scipy.special import cbrt
import sfta
import tensorflow as tf
import pandas as pd
import poc11_env


# To stop potential randomness
rng = np.random.RandomState(128)

FLAGS = tf.app.flags.FLAGS

train = pd.read_csv(os.path.join(FLAGS.dataset_dir, 'train', 'train.csv'))
test = pd.read_csv(os.path.join(FLAGS.dataset_dir, 'test', 'test.csv'))

train.head()
test.head()


def preproc(unclean_batch_x):
    """Convert values to range 0-1"""
    temp_batch = unclean_batch_x / unclean_batch_x.max()

    return temp_batch


def dense_to_one_hot(labels_dense, num_classes=poc11_env.CLASS_NUM):
    """Convert class labels from scalars to one-hot vectors"""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1

    return labels_one_hot


def batch_creator(dataset_name):
    _dataset = eval(dataset_name + '_x')

    dataset_length = _dataset.shape[0]

    """Create batch with random samples and return appropriate format"""
    batch_mask = rng.choice(dataset_length, FLAGS.batch_size)

    batch_x = _dataset[[batch_mask]].reshape(-1, FLAGS.image_buff_size)
    batch_x = preproc(batch_x)

    #if dataset_name == 'train':
    batch_y = eval(dataset_name).ix[batch_mask, 'label'].values
    batch_y = dense_to_one_hot(batch_y)

    return batch_x, batch_y


def feed_dict(train, _x, _y):
    if train: #or FLAGS.fake_data:
        batch_x, batch_y = batch_creator('train')
    else:
        batch_x, batch_y = batch_creator('test')
    return {_x: batch_x, _y: batch_y}


def get_vactors(image_file):
    img = cv2.imread(image_file)
    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _hue, _sat, _val = hsv_image[:,:,0], hsv_image[:,:,1], hsv_image[:,:,2]
    hue = np.ndarray.flatten(_hue).astype(float)
    sat = np.ndarray.flatten(_sat).astype(float)
    val = np.ndarray.flatten(_val).astype(float)
    avgs = []
    sd = []
    sk = []

    # average
    arr = hue / hue.shape[0]
    avgs.append(arr.sum(axis=0))

    arr = sat / sat.shape[0]
    avgs.append(arr.sum(axis=0))

    arr = val / val.shape[0]
    avgs.append(arr.sum(axis=0))

    # standard deviation
    arr = np.power(hue - avgs[0], 2)
    sd.append(math.sqrt(arr.sum(axis=0) / hue.shape[0]))

    arr = np.power(sat - avgs[1], 2)
    sd.append(math.sqrt(arr.sum(axis=0) / sat.shape[0]))

    arr = np.power(val - avgs[2], 2)
    sd.append(math.sqrt(arr.sum(axis=0) / val.shape[0]))

    # skewness
    arr = np.power(hue - avgs[0], 3)
    sk.append(cbrt(arr.sum(axis=0) / hue.shape[0]))

    arr = np.power(sat - avgs[1], 3)
    sk.append(cbrt(arr.sum(axis=0) / sat.shape[0]))

    arr = np.power(val - avgs[2], 3)
    sk.append(cbrt(arr.sum(axis=0) / val.shape[0]))

    params = np.concatenate((avgs, sd, sk), 0)

    chans = (_hue, _sat, _val)
    bins = (16, 4, 4)
    ranges = ([0, 15], [0, 3], [0, 3])
    hist_values = []

    for (chan, bin, range) in zip(chans, bins, ranges):
        hist = cv2.calcHist([chan], [0], None, [bin], range)
        hist_values.append(np.ndarray.flatten(hist))

    params = np.concatenate((np.concatenate(hist_values, 0), params), 0)

    SFTA = sfta.SegmentationFractalTextureAnalysis(8)
    fv = SFTA.feature_vector(gray)

    print ("Loading " + image_file)
    return np.concatenate((fv, params), 0)


def decode_image(var):
    temp = []
    for image_file in eval(var).filename:
        image_path = os.path.join(FLAGS.dataset_dir, var, 'images' + str(FLAGS.image_size), image_file)
        temp.append(get_vactors(image_path))

    return np.stack(temp)


if not poc11_env.ON_TEST:
    test_x = decode_image('test')
    train_x = decode_image('train')
