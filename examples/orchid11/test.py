import tensorflow as tf


def main(arg):
    graph = tf.Graph()
    with graph.as_default():
        file_name = tf.placeholder(dtype=tf.string)
        splitf = tf.string_split([file_name], '/')
        v = splitf.values[-1]
        splitf = tf.string_split([v], '_')
        v = splitf.values[0]
        v = tf.string_to_number(v, tf.int32)
        res = tf.one_hot(v, 11)


    p = '/Users/sarachaii/Desktop/trains/resized3/4_20.jpg'

    with tf.Session(graph=graph) as sess:
        tf.global_variables_initializer().run()
        print (sess.run(res, feed_dict={file_name: p}))


if __name__ == '__main__':
    tf.app.run()
