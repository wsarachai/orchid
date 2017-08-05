import os
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import PreProcess
import tensorflow as tf

DATA_TYPE = 'general'
DATA_SIZE = 224

ROOT_DIR = '/Volumes/Data/_Corpus-data/Orchids/orchid11/trains/'
DATASET_DIR = os.path.join(ROOT_DIR, 'dataset', DATA_TYPE)

image_path = os.path.join(DATASET_DIR, 'test/images' + str(DATA_SIZE) + '/5_100.jpg')
tfrecords_filename = os.path.join(DATASET_DIR, 'orchid11.tfrecords')

writer = tf.python_io.TFRecordWriter(tfrecords_filename)

cat_img = io.imread(image_path)
#io.imshow(cat_img)

cat_string = cat_img.tostring()

reconstructed_cat_1d = np.fromstring(cat_string, dtype=np.uint8)

reconstructed_cat_img = reconstructed_cat_1d.reshape(cat_img.shape)

print np.allclose(cat_img, reconstructed_cat_img)

#plt.show()