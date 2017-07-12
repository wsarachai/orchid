from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import glob
import pickle


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('image_dir', '/Users/sarachaii/Desktop/trains/resized/',
                           'Directory to source of images')
tf.app.flags.DEFINE_string('data_dir', '/Users/sarachaii/Desktop/trains/orchid11_data/',
                            'Directory to download data files and write the converted result')
tf.app.flags.DEFINE_string('file_ext', '*.jpg',
                            'The extension of image files')


def decode_image(image_file_names, resize_func=None):
    images = []

    graph = tf.Graph()
    with graph.as_default():
        file_name = tf.placeholder(dtype=tf.string)
        file = tf.read_file(file_name)
        image = tf.image.decode_jpeg(file)
        if resize_func != None:
            image = resize_func(image)

    with tf.Session(graph=graph) as session:
        tf.global_variables_initializer().run()
        for i in range(len(image_file_names)):
            images.append(session.run(image, feed_dict={file_name: image_file_names[i]}))
            if (i + 1) % 10 == 0:
                print('Images processed: ', i + 1)

        session.close()

    return np.array(images)

def create_batch(data, label, batch_size):
    i = 0
    while (i * batch_size * 2) < len(data):
        bst = i * batch_size * 2
        ben = (i+1) * batch_size * 2
        with open(label+ '_' + str(i) +'.pickle', 'wb') as handle:
            content = data[bst:ben]
            pickle.dump(content, handle)
            print('Saved',label,'part #' + str(i), 'with', len(content),'entries.')
        i += 1


#create_batch(train_images, os.path.join(DATA_DIR, 'train_images'), 10)
#create_batch(test_images, os.path.join(DATA_DIR, 'test_images'), 10)

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to(images, labels, name):
  if images.shape[0] != labels.shape[0]:
    raise ValueError('Images size %d does not match label size %d.' %
                     (images.shape[0], labels.shape[0]))

  # for i in range(5):
  #    plt.imshow(processed_test_images[i])
  #    plt.show()

  rows = images.shape[1]
  cols = images.shape[2]
  depth = images.shape[3]

  filename = os.path.join(FLAGS.data_dir, name + '.tfrecords')

  print('Writing', filename)
  writer = tf.python_io.TFRecordWriter(filename)

  for index in range(0, images.shape[0]):
    image_raw = images[index].tostring()

    res_labels = tf.one_hot([0, 3], 4)
    with tf.Session() as sess:
        sess.run(res_labels)

    example = tf.train.Example(features=tf.train.Features(feature={
        'height': _int64_feature(rows),
        'width': _int64_feature(cols),
        'depth': _int64_feature(depth),
        'label': _int64_feature(int(labels[index])),
        'image_raw': _bytes_feature(image_raw)}))
    writer.write(example.SerializeToString())
  writer.close()

def main(unused_argv):
  dirs = [FLAGS.image_dir + i for i in os.listdir(FLAGS.image_dir)]

  ilabel = np.uint8(0)

  train_label = np.array([])
  test_label = np.array([])
  train_image_file_names = np.array([])
  test_image_file_names = np.array([])

  for d in dirs:
    files = glob.glob(os.path.join(d, FLAGS.file_ext))
    if len(files) != 100:
      raise ValueError('A number of dataset is invalid.')

    for lbl in files[0:80]:
      train_image_file_names = np.append(train_image_file_names, lbl)

    for lbl in files[-20:]:
      test_image_file_names = np.append(test_image_file_names, lbl)

    for i in range(80):
      train_label = np.append(train_label, ilabel)

    for i in range(20):
      test_label = np.append(test_label, ilabel)

    ilabel += 1

  #resize_func = lambda image: tf.image.resize_image_with_crop_or_pad(image, HEIGHT, WIDTH)

  processed_train_images = decode_image(train_image_file_names, resize_func=None)
  processed_test_images = decode_image(test_image_file_names, resize_func=None)

  idx = 0
  st = 0
  en = 20
  for i in range(0, 44):
    print (len(processed_train_images[st:en]))
    convert_to(processed_train_images[st:en], train_label[st:en], 'oh-orchid11-images-train-%d' % i)
    i += 1
    st += 20
    en += 20

  idx = 0
  st = 0
  en = 20
  for i in range(0, 11):
    print(len(processed_test_images[st:en]))
    convert_to(processed_test_images[st:en], test_label[st:en], 'oh-orchid11-images-test-%d' % i)
    i += 1
    st += 20
    en += 20


if __name__ == '__main__':
  tf.app.run()
