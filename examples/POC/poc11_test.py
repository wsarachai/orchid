import os
import tensorflow as tf
import poc11
import poc11_input
import poc11_env
from scipy.misc import imread

FLAGS = tf.app.flags.FLAGS


def main():
    sess = tf.InteractiveSession()

    _x = tf.placeholder(tf.float32, [None, FLAGS.image_buff_size])
    _y = tf.placeholder(tf.float32, [None, poc11_env.CLASS_NUM])

    output_layer, keep_prob = poc11.MLP(_x)

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    saver.restore(sess, os.path.join(FLAGS.summaries_dir, "model.ckpt"))
    print("Model restored.")

    image_path = os.path.join(FLAGS.dataset_dir, 'test/images' + str(FLAGS.image_size) + '/6_92.jpg')
    print (image_path)

    vc_img = poc11_input.get_vactors(image_path)
    #vc_img = poc11_input.preproc(vc_img)

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
