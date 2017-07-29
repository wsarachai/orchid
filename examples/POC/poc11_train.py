import os
import tensorflow as tf
import poc11
import poc11_input
import poc11_env

FLAGS = tf.app.flags.FLAGS


def main():
    sess = tf.InteractiveSession()

    _x = tf.placeholder(tf.float32, [None, FLAGS.image_buff_size])
    _y = tf.placeholder(tf.float32, [None, poc11_env.CLASS_NUM])

    output_layer = poc11.MLP(_x)

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output_layer, labels=_y))

    optimizer = tf.train.GradientDescentOptimizer(FLAGS.learning_rate).minimize(cross_entropy)

    # find predictions on val set
    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            pred_temp = tf.equal(tf.argmax(output_layer, 1), tf.argmax(_y, 1))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(pred_temp, tf.float32))

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    tf.global_variables_initializer().run()

    for step in range(FLAGS.epochs):
        if step % 4 == 0:  # Record summaries and test-set accuracy
            acc = sess.run([accuracy], feed_dict=poc11_input.feed_dict(False, _x, _y))
            print('Accuracy at step %s: %s' % (step, acc))
        else:
            if step % 10 == 9:  # Record execution stats
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                _ = sess.run([optimizer],
                             feed_dict=poc11_input.feed_dict(True, _x, _y),
                             options=run_options,
                             run_metadata=run_metadata)
                print('Adding run metadata for', step)
            else:  # Record a summary
                _ = sess.run([optimizer], feed_dict=poc11_input.feed_dict(True, _x, _y))

    print ("\nTraining complete!")

    save_path = saver.save(sess, os.path.join(FLAGS.orchid_summaries_dir, "model.ckpt"))
    print("Model saved in file: %s" % save_path)


if __name__ == '__main__':
    main()
