import os
import tensorflow as tf

IMAGE_SIZE = 224
CLASS_NUM = 11
IMAGE_BUFF_SIZE = 81
HIDDEN_NEURON = 512

DATA_TYPE = 'ground-truth'
#DATA_TYPE = 'general'

if IMAGE_SIZE == 32:
    BATCH_SIZE = 128
    LEARNING_RATE = 0.001
    EPOCHS = 100000
elif IMAGE_SIZE == 224:
    BATCH_SIZE = 400
    LEARNING_RATE = 0.001
    EPOCHS = 50000

ROOT_DIR = '/Users/sarachaii/Desktop/trains/'
DATASET_DIR = os.path.join(ROOT_DIR, 'dataset', DATA_TYPE)
SUMMARIES = 'summaries' + str(IMAGE_SIZE)
SUMMARIES_DIR = os.path.join(ROOT_DIR, 'poc11-logs', DATA_TYPE, SUMMARIES)

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('root_dir', ROOT_DIR, 'Root directory.')
tf.app.flags.DEFINE_string('summaries_dir', SUMMARIES_DIR, 'Summaries directory.')
tf.app.flags.DEFINE_string('dataset_dir', DATASET_DIR, 'Dataset directory')
tf.app.flags.DEFINE_string('orchid_summaries_dir', SUMMARIES_DIR, 'Summaries directory')
tf.app.flags.DEFINE_integer('image_buff_size', IMAGE_BUFF_SIZE, 'Image buffer size')
tf.app.flags.DEFINE_integer('image_size', IMAGE_SIZE, 'Image size')
tf.app.flags.DEFINE_integer('batch_size', BATCH_SIZE, 'Training batch size')
tf.app.flags.DEFINE_integer('epochs', EPOCHS, 'Number of epochs')
tf.app.flags.DEFINE_float('learning_rate', LEARNING_RATE, 'Learning rate.')

# check for existence
os.path.exists(FLAGS.root_dir)
os.path.exists(FLAGS.dataset_dir)
