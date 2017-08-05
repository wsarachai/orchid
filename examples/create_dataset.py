from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
import skimage.io as io
from matplotlib import pyplot as plt
import glob
import pickle

DATA_TYPE = 'general'
DATA_SIZE = 224

FLAGS = tf.app.flags.FLAGS

ROOT_DIR = '/Volumes/Data/_Corpus-data/Orchids/orchid11/trains/'
DATASET_DIR = os.path.join(ROOT_DIR, 'dataset', DATA_TYPE)
IMAGE_DIR = os.path.join(DATASET_DIR, 'test/images' + str(DATA_SIZE))

tf.app.flags.DEFINE_string('src_dir', IMAGE_DIR, 'Directory to source of images')
tf.app.flags.DEFINE_string('data_dir', DATASET_DIR, 'Directory to dest of images')
tf.app.flags.DEFINE_string('file_ext', '*.jpg', 'The extension of image files')


tfrecords_filename = DATASET_DIR + str(DATA_SIZE) + '\orchid11.tfrecords'


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
        raise ValueError('Images size %d does not match label size %d.' %(images.shape[0], labels.shape[0]))

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
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(rows),
            'width': _int64_feature(cols),
            'depth': _int64_feature(depth),
            'label': _int64_feature(int(labels[index])),
            'image_raw': _bytes_feature(image_raw)}))
        writer.write(example.SerializeToString())
    writer.close()

def main(unused_argv):
    dirs = [FLAGS.src_dir + i for i in os.listdir(FLAGS.src_dir)]

    writer = tf.python_io.TFRecordWriter(tfrecords_filename)

    for d in dirs:
        img = np.array(io.imread(d))
        print(d)

if __name__ == '__main__':
  tf.app.run()
