import tensorflow as tf

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/tensorflow/mnist/input_data", one_hot=True)

# Parameters
learning_rate = 0.01
training_epochs = 10
batch_size = 100
display_step = 1

# tf Graph Input
x = tf.placeholder(tf.float32, [None, 784])  # mnist data image of shape 28*28=784
y = tf.placeholder(tf.float32, [None, 10])  # 0-9 digits recognition => 10 classes

# Set model weights
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# Construct model
pred = tf.nn.softmax(tf.matmul(x, W))  # Softmax

# Minimize error using cross entropy
cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred), reduction_indices=1))

W_grad = - tf.matmul(tf.transpose(x), y - pred)
b_grad = - tf.reduce_mean(tf.matmul(tf.transpose(x), y - pred), reduction_indices=0)

new_W = W.assign(W - learning_rate * W_grad)
new_b = b.assign(b - learning_rate * b_grad)

init = tf.global_variables_initializer()

with tf.Session() as sess:
  sess.run(init)

  # Training cycle
  for epoch in range(training_epochs):
    avg_cost = 0.
    total_batch = int(mnist.train.num_examples / batch_size)
    # Loop over all batches
    for i in range(total_batch):
      batch_xs, batch_ys = mnist.train.next_batch(batch_size)
      # Fit training using batch data
      _1, _2 = sess.run([W_grad, b_grad], feed_dict={x: batch_xs, y: batch_ys})
      print (_2)
      _, _, c = sess.run([new_W, new_b, cost], feed_dict={x: batch_xs, y: batch_ys})

      # Compute average loss
      avg_cost += c / total_batch
    # Display logs per epoch step
    if (epoch + 1) % display_step == 0:
      print ("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))

  print ("Optimization Finished!")

  # Test model
  correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
  # Calculate accuracy for 3000 examples
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  print ("Accuracy:", accuracy.eval({x: mnist.test.images[:3000], y: mnist.test.labels[:3000]}))