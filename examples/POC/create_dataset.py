import os
import pandas as pd
import tensorflow as tf
import poc11_env
import numpy as np
import skimage.io as io

poc11_env.ON_TEST = True
import poc11_input

FLAGS = tf.app.flags.FLAGS

train = pd.read_csv(os.path.join(FLAGS.dataset_dir, 'train', 'train.csv'))
test = pd.read_csv(os.path.join(FLAGS.dataset_dir, 'test', 'test.csv'))

train.head()
test.head()


def _float32_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def dense_to_one_hot(labels_dense, num_classes=poc11_env.CLASS_NUM):
    """Convert class labels from scalars to one-hot vectors"""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1

    return labels_one_hot


var = 'train'
tfrecords_filename = poc11_env.DATASET_DIR + '/orchid11_' + var + '.tfrecords'


def write_record():
    writer = tf.python_io.TFRecordWriter(tfrecords_filename)

    labels = eval(var)['label'].values
    labels = dense_to_one_hot(labels)

    for row in eval(var).values:
        n = row[0]
        img_file = row[1]
        l = labels[n]

        image_path = os.path.join(FLAGS.dataset_dir, var, 'images' + str(FLAGS.image_size), img_file)
        v = poc11_input.get_vactors(image_path)

        example = tf.train.Example(features=tf.train.Features(feature={
            'label': _float32_feature(l),
            'vector': _float32_feature(v)}))

        writer.write(example.SerializeToString())

    writer.close()


def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
        serialized_example,
        features={
            'label': tf.FixedLenFeature([11], tf.float32),
            'vector': tf.FixedLenFeature([81], tf.float32),
        })

    lbl = tf.cast(features['label'], tf.float32)
    vec = tf.cast(features['vector'], tf.float32)


    return vec, lbl


def read_record():
    filename_queue = tf.train.string_input_producer([tfrecords_filename])

    vec, lbl = read_and_decode(filename_queue)

    #tf.global_variables_initializer().run()

    with tf.Session()  as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for i in xrange(880):
            v, l = sess.run([vec, lbl])

            print l


def main():
    write_record()
    read_record()


if __name__ == '__main__':
    main()
