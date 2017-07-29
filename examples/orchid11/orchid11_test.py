import os
import tensorflow as tf
import orchid11
import orchid11_input
from scipy.misc import imread

FLAGS = tf.app.flags.FLAGS


def main():
    sess = tf.InteractiveSession()

    # define placeholders
    with tf.name_scope('input'):
        _x = tf.placeholder(tf.float32, [None, FLAGS.image_buff_size])
        _y = tf.placeholder(tf.float32, [None, 11])

    with tf.name_scope('input_reshape'):
        x_image = tf.reshape(_x, [-1, FLAGS.image_size, FLAGS.image_size, FLAGS.image_channels])
        tf.summary.image('input', x_image, 11)

    output_layer, keep_prob = orchid11.deepnn(x_image)

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    saver.restore(sess, os.path.join(FLAGS.summaries_dir, "model.ckpt"))
    print("Model restored.")

    image_path = os.path.join(FLAGS.dataset_dir, 'test/images' + str(FLAGS.image_size) + '/6_92.jpg')
    print (image_path)

    pd_img = imread(image_path, flatten=False)
    pd_img = pd_img.astype('float32')
    pd_img = pd_img.reshape(-1, FLAGS.image_buff_size)
    pd_img = orchid11_input.preproc(pd_img)

    perc = output_layer.eval({_x: pd_img, keep_prob: 1.0})

    perc_max = tf.nn.relu(perc)
    perc_max = perc_max.eval()

    perc_sum = tf.reduce_sum(perc_max, 1)
    perc_sum = perc_sum.eval()

    #print (perc_max)
    #print (perc_sum)

    perc_ans = perc_max / perc_sum * 100

    lb = 0
    for p in perc_ans[0]:
        print ("label {0}: {1:2.2f}%".format(lb, p))
        lb += 1


if __name__ == '__main__':
    main()
