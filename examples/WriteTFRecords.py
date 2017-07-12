"""
Author: David Crook
Copyright Microsoft Corporation 2017
"""
import PreProcess
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('image_dir', '/Users/sarachaii/Desktop/trains/resized3/',
                           'Directory to source of images')
tf.app.flags.DEFINE_string('data_dir', '/Users/sarachaii/Desktop/trains/orchid11_data3/',
                            'Directory to download data files and write the converted result')
def main():
    '''
    Main function which converts a label file into tf records
    '''
    PreProcess.write_records_from_file(FLAGS.image_dir, FLAGS.data_dir, 6)


if __name__ == "__main__":
    main()