import os
import tensorflow as tf
import pandas as pd
import numpy as np

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


def dense_to_one_hot(labels_dense, num_classes=11):
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


def feed_dict(train, _x, _y, keep_prob):
    if train: #or FLAGS.fake_data:
        batch_x, batch_y = batch_creator('train')
        k = FLAGS.dropout
    else:
        batch_x, batch_y = batch_creator('test')
        k = 1.0
    return {_x: batch_x, _y: batch_y, keep_prob: k}


def decode_image(var):
    with tf.Session() as sess:
        temp = []

        graph = tf.Graph()
        with graph.as_default():
            file_name = tf.placeholder(dtype=tf.string)
            file = tf.read_file(file_name)
            image = tf.image.decode_jpeg(file)
            image = tf.cast(image, tf.float32)
            image.set_shape((FLAGS.image_size, FLAGS.image_size, FLAGS.image_channels))

        with tf.Session(graph=graph) as session:
            tf.global_variables_initializer().run()
            for img_name in eval(var).filename:
                image_path = os.path.join(FLAGS.dataset_dir, var, 'images' + str(FLAGS.image_size), img_name)
                img = session.run(image, feed_dict={file_name: image_path})
                temp.append(img)
            session.close()

    return np.stack(temp)


test_x = decode_image('test')
train_x = decode_image('train')
