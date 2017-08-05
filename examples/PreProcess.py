'''
Author: David Crook
Module which contains functions for pre-processing image data
'''
import math
import os
import tensorflow as tf
import numpy as np
from scipy import misc
import CONSTANTS


def bytes_feature(value):
    '''
    Creates a TensorFlow Record Feature with value as a byte array
    '''
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def int64_feature(value):
    '''
    Creates a TensorFlow Record Feature with value as a 64 bit integer.
    '''
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def write_record(dest_path, df):
    '''
    Writes an actual TF record from a data frame
    '''
    writer = tf.python_io.TFRecordWriter(dest_path)
    for i in range(len(df)):
        example = tf.train.Example(features=tf.train.Features(feature={
            'example': bytes_feature(df[i]['image']),
            'label': int64_feature(df[i]['label'])
        }))
        writer.write(example.SerializeToString())
    writer.close()


def read_image_to_bytestring(path):
    '''
    Reads an image from a path and converts it
    to a flattened byte string
    '''
    img = misc.imread(path).astype(np.float32) / 255.0
    return img.reshape(CONSTANTS.IMAGE_SHAPE).flatten().tostring()


def write_records_from_file(image_dir, dest_folder, num_records):
    '''
    Takes a label file as a path and converts entries into a tf record
    for image classification.
    '''
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    img_arrs = [image_dir + i for i in os.listdir(image_dir)]

    labels = []

    for img in img_arrs:
        head, tail = os.path.split(img)
        filename, file_extension = os.path.splitext(tail)
        label = filename.split("_")[0]
        newlbl = {'label':int(label), 'image': img}
        labels.append(newlbl)

    start_idx = 0
    dlen = len(labels)
    ex_per_rec = math.ceil(dlen / num_records)
    for i in range(1, num_records):
        rec_path = dest_folder + str(i) + '.tfrecords'
        write_record(rec_path, labels[int(start_idx):int((ex_per_rec * i) - 1)])
        start_idx += ex_per_rec
        print('wrote record: ', i)
    final_rec_path = dest_folder + str(num_records) + '.tfrecords'
    write_record(final_rec_path, labels.loc[ex_per_rec * (num_records - 1):].reset_index())
    print('wrote record: ', num_records)
    print('finished writing records...')