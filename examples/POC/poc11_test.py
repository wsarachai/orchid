import os
import tensorflow as tf
import poc11
import poc11_env

poc11_env.ON_TEST = True
import poc11_input

FLAGS = tf.app.flags.FLAGS


def main():
    sess = tf.InteractiveSession()

    _x = tf.placeholder(tf.float32, [None, FLAGS.image_buff_size])
    _y = tf.placeholder(tf.float32, [None, poc11_env.CLASS_NUM])

    output_layer = poc11.MLP(_x)

    # find predictions on val set
    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            pred_temp = tf.equal(tf.argmax(output_layer, 1), tf.argmax(_y, 1))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(pred_temp, tf.float32))

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    SUMMARIES_DIR = os.path.join('/Users/sarachaii/Desktop/trains/poc11-logs', poc11_env.DATA_TYPE, poc11_env.SUMMARIES)
    saver.restore(sess, os.path.join(SUMMARIES_DIR, "model.ckpt"))
    print("Model restored.")

    acc = sess.run([accuracy], feed_dict=poc11_input.feed_dict(False, _x, _y))
    print('Accuracy is {0:2.2f}%'.format(acc[0]*100))

    #image_path = os.path.join(FLAGS.dataset_dir, 'test/images' + str(FLAGS.image_size) + '/6_99.jpg')
    image_path = os.path.join(FLAGS.dataset_dir, 'test/images' + str(FLAGS.image_size) + '/8_100.jpg')
    #image_path = os.path.join(FLAGS.dataset_dir, 'test/images' + str(FLAGS.image_size) + '/9_100.jpg')

    print (image_path)

    vc_img = poc11_input.get_vactors(image_path)
    vc_img = vc_img.reshape(-1, FLAGS.image_buff_size)
    vc_img = poc11_input.preproc(vc_img)

    perc = output_layer.eval({_x: vc_img})

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
