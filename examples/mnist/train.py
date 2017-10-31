import network
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets('/tmp/tensorflow/mnist/input_data', one_hot=True)
net = network.Network([784, 30, 10])
net.SGD(mnist.train.images, 30, 10, 3.0, test_data=mnist.test.images)
