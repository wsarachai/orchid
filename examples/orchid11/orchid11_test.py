import os
import tensorflow as tf
import orchid11
import orchid11_env
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

    # find predictions on val set
    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            pred_temp = tf.equal(tf.argmax(output_layer, 1), tf.argmax(_y, 1))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(pred_temp, tf.float32))

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    SUMMARIES_DIR = os.path.join('/Users/sarachaii/Desktop/trains/logs', orchid11_env.DATA_TYPE, orchid11_env.SUMMARIES)
    saver.restore(sess, os.path.join(SUMMARIES_DIR, "model.ckpt"))
    print("Model restored.")

    batch_x, batch_y = orchid11_input.batch_creator('test')
    acc = sess.run([accuracy], feed_dict={_x: batch_x, _y: batch_y, keep_prob: 1.0})
    print('Accuracy is {0:2.2f}%'.format(acc[0]*100))

    image_path = os.path.join(FLAGS.dataset_dir, 'test/images' + str(FLAGS.image_size) + '/5_88.jpg')
    #image_path = os.path.join(FLAGS.dataset_dir, 'test/images' + str(FLAGS.image_size) + '/9_100.jpg')
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
