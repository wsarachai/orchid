import os
import tensorflow as tf
import orchid11
from scipy.misc import imread


def main():
    sess = tf.InteractiveSession()

    # define placeholders
    with tf.name_scope('input'):
        _x = tf.placeholder(tf.float32, [None, orchid11.IMAGE_BUFF_SIZE])
        _y = tf.placeholder(tf.float32, [None, 11])

    with tf.name_scope('input_reshape'):
        x_image = tf.reshape(_x, [-1, orchid11.IMAGE_SIZE, orchid11.IMAGE_SIZE, orchid11.IMAGE_CHANNEL])
        tf.summary.image('input', x_image, 11)

    output_layer, keep_prob = orchid11.deepnn(x_image)

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    saver.restore(sess, os.path.join(orchid11.FLAGS.summaries_dir, "model.ckpt"))
    print("Model restored.")

    image_path = os.path.join(orchid11.FLAGS.data_dir, 'test/images' + str(orchid11.IMAGE_SIZE) + '/5_91.jpg')
    print (image_path)

    pd_img = imread(image_path, flatten=False)
    pd_img = pd_img.astype('float32')
    pd_img = pd_img.reshape(-1, orchid11.IMAGE_BUFF_SIZE)
    pd_img = orchid11.preproc(pd_img)

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
