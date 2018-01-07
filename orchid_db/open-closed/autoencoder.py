from __future__ import division, print_function, absolute_import

import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
import constants
from scipy.stats import entropy

from tensorflow.python.platform import gfile


def create_image_lists_test(image_dir, testing_percentage, validation_percentage):
  if not gfile.Exists(image_dir):
    tf.logging.error("Image directory '" + image_dir + "' not found.")
    return None

  result = {}

  sub_dirs = [x[0] for x in gfile.Walk(image_dir)]

  # The root directory comes first, so skip it.
  is_root_dir = True
  for sub_dir in sub_dirs:
    if is_root_dir:
      is_root_dir = False
      continue

    extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
    file_list = []

    dir_name = os.path.basename(sub_dir)
    if dir_name == image_dir:
      continue

    tf.logging.info("Looking for images in '" + dir_name + "'")
    for extension in extensions:
      file_glob = os.path.join(image_dir, dir_name, '*.' + extension)
      file_list.extend(gfile.Glob(file_glob))

    if not file_list:
      tf.logging.warning('No files found')
      continue

    file_size = len(file_list)

    if file_size < 20:
      tf.logging.warning(
          'WARNING: Folder has less than 20 images, which may cause issues.')

    label_name = dir_name.lower()
    training_images = []
    testing_images = []
    validation_images = []

    random.shuffle(file_list)
    testing_idx = int(file_size * testing_percentage)
    validation_idx = int(file_size * validation_percentage)

    for index, file_name in enumerate(file_list):
      base_name = os.path.basename(file_name)

      if index < validation_idx:
        validation_images.append(base_name)
      elif index < (testing_idx + validation_idx):
        testing_images.append(base_name)
      else:
        training_images.append(base_name)

    _result = {
      'dir': dir_name,
      'training': training_images,
      'testing': testing_images,
      'validation': validation_images,
    }

    result[label_name] = _result
  return result

def normalize(v):
  norm = np.linalg.norm(v, ord=1)
  if norm == 0:
    norm = np.finfo(v.dtype).eps
  return v / norm


def get_next_batch(file_list, batch_size):
  batch_x = []
  if batch_size > 0:
    for unused_i in range(batch_size):
      image_index = random.randrange(2 ** 32 - 1)
      mod_index = image_index % len(file_list)

      with open(file_list[mod_index], 'r') as bottleneck_file:
        bottleneck_string = bottleneck_file.read()

      try:
        bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
      except ValueError:
        tf.logging.warning('Invalid float found, recreating bottleneck')

      min = np.abs(np.min(bottleneck_values))
      bottleneck = np.add(bottleneck_values, [min])
      bottleneck = normalize(bottleneck)

      batch_x.append(bottleneck)
  else:
    for f in file_list:
      with open(f, 'r') as bottleneck_file:
        bottleneck_string = bottleneck_file.read()

      try:
        bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
      except ValueError:
        tf.logging.warning('Invalid float found, recreating bottleneck')

      min = np.abs(np.min(bottleneck_values))
      bottleneck = np.add(bottleneck_values, [min])
      bottleneck = normalize(bottleneck)

      batch_x.append(bottleneck)

  return batch_x


def get_file_list(image_data):
  if not gfile.Exists(image_data):
    exit(1)

  sub_dirs = [x[0] for x in gfile.Walk(image_data)]

  file_list = []

  is_root_dir = True
  for sub_dir in sub_dirs:
    if is_root_dir:
      is_root_dir = False
      continue

    dir_name = os.path.basename(sub_dir)
    file_glob = os.path.join(image_data, dir_name, '*.txt')
    file_list.extend(gfile.Glob(file_glob))
  return file_list


# Building the encoder
def encoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']), biases['encoder_b1']))
    # Encoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']), biases['encoder_b2']))
    return layer_2
    #return layer_1


# Building the decoder
def decoder(x):
    # Decoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']), biases['decoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']), biases['decoder_b2']))
    #layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h2']), biases['decoder_b2']))
    return layer_2

b_train = False

# Training Parameters
learning_rate = 0.01
num_steps = 30000
batch_size = 512

display_step = 50
examples_to_show = 10

# Network Parameters
num_hidden_1 = 128 # 1st layer num features
num_hidden_2 = 64 # 2nd layer num features (the latent dim)
num_input = 27 # data input

threshold = 0.053

# tf Graph input (only pictures)
X = tf.placeholder("float", [None, num_input])

weights = {
    'encoder_h1': tf.Variable(tf.random_normal([num_input, num_hidden_1])),
    'encoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_hidden_2])),
    'decoder_h1': tf.Variable(tf.random_normal([num_hidden_2, num_hidden_1])),
    'decoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_input])),
}
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([num_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([num_hidden_2])),
    'decoder_b1': tf.Variable(tf.random_normal([num_hidden_1])),
    'decoder_b2': tf.Variable(tf.random_normal([num_input])),
}

info = constants.info

# Construct model
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

# Prediction
y_pred = decoder_op
# Targets (Labels) are the input data.
y_true = X

# Define loss and optimizer, minimize the squared error
loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2))

optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)
#optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy_mean)


if b_train:
  # Initialize the variables (i.e. assign their default value)
  init = tf.global_variables_initializer()

saver = tf.train.Saver()

file_list_closed = get_file_list(os.path.join(constants.image_dir, info["data_type"], "closed"))
file_list_open = get_file_list(os.path.join(constants.image_dir, info["data_type"], "open"))

with tf.Session() as sess:
  if b_train:
    # Run the initializer
    sess.run(init)

    # Training
    for i in range(1, num_steps+1):
      batch_x = get_next_batch(file_list_closed, batch_size)
      # Run optimization op (backprop) and cost op (to get loss value)
      _, l = sess.run([optimizer, loss], feed_dict={X: batch_x})
      # Display logs per step
      if i % display_step == 0 or i == 1:
          print('Step %i: Minibatch Loss: %f' % (i, l))

      if l < 0.00005:
        break

    save_path = saver.save(sess, "/tmp/model.ckpt")
    print("Model saved in file: %s" % save_path)
  else:
    saver.restore(sess, "/Volumes/Data/_dataset/models/autoencoder/ex03/model.ckpt")
    print("Model restored.")

  closed_x = get_next_batch(file_list_closed, -1)
  closed_g = sess.run(decoder_op, feed_dict={X: closed_x})

  closed_correct = 0
  open_correct = 0
  closed_mean = 0
  open_mean = 0

  for i, x in enumerate(closed_x):
    l_ = entropy(x, closed_g[i])
    closed_mean += l_
    if l_ < threshold:
      closed_correct += 1

  open_x = get_next_batch(file_list_open, -1)
  open_g = sess.run(decoder_op, feed_dict={X: open_x})

  for i, x in enumerate(open_x):
    l_ = entropy(x, open_g[i])
    open_mean += l_
    if l_ >= threshold:
      open_correct += 1

print ("closed accuracy={0:.2f}".format(closed_correct/len(closed_x)))
print ("mean={0:.5f}".format(closed_mean/len(closed_x)))
print ("")
print ("open accuracy={0:.2f}".format(open_correct/len(open_x)))
print ("mean={0:.5f}".format(open_mean/len(open_x)))

_x = closed_x
_g = closed_g
image_index = random.randrange(2 ** 32 - 1)
mod_index = image_index % len(_x)

n_classes = 27
step = 0.02
labels = np.arange(1, n_classes + 1, 1)
fig, axs = plt.subplots(2, 1, sharex=True)
fig.subplots_adjust(hspace=1)
smax = np.max(_x[mod_index])
axs[0].plot(labels, _x[mod_index], 'o--')
axs[0].set_yticks(np.arange(0, smax+0.1, step))
axs[0].set_ylim(0, smax)
axs[1].plot(labels, _g[mod_index], 'o--')
axs[1].set_yticks(np.arange(0, smax+0.1, step))
axs[1].set_ylim(0, smax)
plt.show()

_x = open_x
_g = open_g
image_index = random.randrange(2 ** 32 - 1)
mod_index = image_index % len(_x)

fig, axs = plt.subplots(2, 1, sharex=True)
fig.subplots_adjust(hspace=1)
smax = np.max(_x[mod_index])
axs[0].plot(labels, _x[mod_index], 'o--')
axs[0].set_yticks(np.arange(0, smax+0.1, step))
axs[0].set_ylim(0, smax)
axs[1].plot(labels, _g[mod_index], 'o--')
axs[1].set_yticks(np.arange(0, smax+0.1, step))
axs[1].set_ylim(0, smax)
plt.show()

exit()

