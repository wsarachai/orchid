import os
import tensorflow as tf

IMAGE_SIZE = 224
IMAGE_CHANNEL = 3
IMAGE_BUFF_SIZE = IMAGE_SIZE*IMAGE_SIZE*IMAGE_CHANNEL
CLASSES_NUM = 11

#DATA_TYPE = 'ground-truth'
DATA_TYPE = 'general'

if IMAGE_SIZE == 32:
    BATCH_SIZE = CLASSES_NUM * 16
    LEARNING_RATE = 0.001
    DROPOUT = 0.5
    EPOCHS = 20000
elif IMAGE_SIZE == 224:
    BATCH_SIZE = CLASSES_NUM * 3
    LEARNING_RATE = 0.001
    DROPOUT = 0.5
    EPOCHS = 15000

#ROOT_DIR = '/home/keng/Desktop/trains/'
ROOT_DIR = '/Users/sarachaii/Desktop/trains/'
DATASET_DIR = os.path.join(ROOT_DIR, 'dataset', DATA_TYPE)
SUMMARIES = 'summaries' + str(IMAGE_SIZE)
SUMMARIES_DIR = os.path.join(ROOT_DIR, 'logs', DATA_TYPE, SUMMARIES)

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('root_dir', ROOT_DIR, 'Root directory.')
tf.app.flags.DEFINE_string('summaries_dir', SUMMARIES_DIR, 'Summaries directory.')
tf.app.flags.DEFINE_string('dataset_dir', DATASET_DIR, 'Dataset directory')
tf.app.flags.DEFINE_string('orchid_summaries_dir', SUMMARIES_DIR, 'Summaries directory')
tf.app.flags.DEFINE_integer('image_channels', IMAGE_CHANNEL, 'Image channels')
tf.app.flags.DEFINE_integer('image_size', IMAGE_SIZE, 'Image size')
tf.app.flags.DEFINE_integer('image_buff_size', IMAGE_BUFF_SIZE, 'Image buffer size')
tf.app.flags.DEFINE_integer('batch_size', BATCH_SIZE, 'Training batch size')
tf.app.flags.DEFINE_integer('classes_num', CLASSES_NUM, 'Number of classes')
tf.app.flags.DEFINE_integer('epochs', EPOCHS, 'Number of epochs')
tf.app.flags.DEFINE_float('dropout', DROPOUT, 'Dropout rate.')
tf.app.flags.DEFINE_float('learning_rate', LEARNING_RATE, 'Learning rate.')

# check for existence
os.path.exists(FLAGS.root_dir)
os.path.exists(FLAGS.dataset_dir)
