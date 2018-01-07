import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm

min_y = min_x = -5
max_y = max_x = 5
x_coords = np.random.uniform(min_x, max_x, (500, 1))
y_coords = np.random.uniform(min_y, max_y, (500, 1))
clazz = np.greater(y_coords, x_coords).astype(int)
delta = 0.5 / np.sqrt(2.0)
x_coords = x_coords + ((0 - clazz) * delta) + ((1 - clazz) * delta)
y_coords = y_coords + (clazz * delta) + ((clazz - 1) * delta)

def PlotClasses(span, delta, x, y, clazz, annotations=None):
  fig, ax = plt.subplots(figsize=(8, 8))
  ax.scatter(x, y, c=clazz, cmap=cm.coolwarm)
  ax.plot(span, 1 * span + 0, 'k-')
  ax.plot(span, 1 * span + delta, 'k,')
  ax.plot(span, 1 * span - delta, 'k,')
  if annotations:
    for i, txt in enumerate(annotations):
      ax.annotate(txt, (x[i], y[i]))
  plt.show()


x_range = np.linspace(min_x - delta, max_x + delta)
PlotClasses(x_range, delta, x_coords, y_coords, clazz)


def input_fn():
  """
  The function provided input for the SVM training.

  In real life code this function would probably read batches of data and return
  batches of IDs and feature columns. For us we simply generate all IDs in one
  go, and create both feature columns by reshaping x_coords and y_coords into
  a n x 1 vectors.
  """
  return {
           'example_id': tf.constant(map(lambda x: str(x + 1), np.arange(len(x_coords)))),
           'x': tf.constant(np.reshape(x_coords, [x_coords.shape[0], 1])),
           'y': tf.constant(np.reshape(y_coords, [y_coords.shape[0], 1])),
         }, tf.constant(clazz)


# Contrib libraries seem overly verbose. Only output errors.
tf.logging.set_verbosity(tf.logging.ERROR)

feature1 = tf.contrib.layers.real_valued_column('x')
feature2 = tf.contrib.layers.real_valued_column('y')
svm_classifier = tf.contrib.learn.SVM(
  feature_columns=[feature1, feature2],
  example_id_column='example_id')
svm_classifier.fit(input_fn=input_fn, steps=30)
metrics = svm_classifier.evaluate(input_fn=input_fn, steps=1)
print "Loss", metrics['loss'], "\nAccuracy", metrics['accuracy']

x_predict = np.random.uniform(min_x, max_x, (20, 1))
y_predict = np.random.uniform(min_y, max_y, (20, 1))


def predict_fn():
  return {
    'x': tf.constant(x_predict),
    'y': tf.constant(y_predict),
  }


pred = list(svm_classifier.predict(input_fn=predict_fn))
predicted_class = map(lambda x: x['classes'], pred)
annotations = map(lambda x: '%.2f' % x['logits'][0], pred)

PlotClasses(x_range, delta, x_predict, y_predict, predicted_class, annotations)

exit(1)