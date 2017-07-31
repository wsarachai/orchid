import os
import tensorflow as tf
import poc11
import poc11_input
import poc11_env

FLAGS = tf.app.flags.FLAGS


def main():
    sess = tf.InteractiveSession()

    with tf.name_scope('input'):
        _x = tf.placeholder(tf.float32, [None, FLAGS.image_buff_size])
        _y = tf.placeholder(tf.float32, [None, poc11_env.CLASS_NUM])

    output_layer = poc11.MLP(_x)

    with tf.name_scope('total'):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output_layer, labels=_y))

    tf.summary.scalar('cross_entropy', cross_entropy)

    with tf.name_scope('train'):
        #optimizer = tf.train.GradientDescentOptimizer(FLAGS.learning_rate).minimize(cross_entropy)
        optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(cross_entropy)

    # find predictions on val set
    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            pred_temp = tf.equal(tf.argmax(output_layer, 1), tf.argmax(_y, 1))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(pred_temp, tf.float32))

    tf.summary.scalar('accuracy', accuracy)
    merged = tf.summary.merge_all()

    train_writer = tf.summary.FileWriter(FLAGS.orchid_summaries_dir + '/train', sess.graph)
    test_writer = tf.summary.FileWriter(FLAGS.orchid_summaries_dir + '/test')

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    tf.global_variables_initializer().run()

    for step in range(FLAGS.epochs):
        if step % 50 == 0:  # Record summaries and test-set accuracy
            summary, acc = sess.run([merged, accuracy], feed_dict=poc11_input.feed_dict(False, _x, _y))
            test_writer.add_summary(summary, step)
            print('Accuracy at step %s: %s' % (step, acc))
        else:
            if step % 500 == 9:  # Record execution stats
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                summary, _ = sess.run([merged, optimizer],
                             feed_dict=poc11_input.feed_dict(True, _x, _y),
                             options=run_options,
                             run_metadata=run_metadata)
                train_writer.add_run_metadata(run_metadata, 'step%03d' % step)
                train_writer.add_summary(summary, step)
                print('Adding run metadata for', step)
            else:  # Record a summary
                summary, _ = sess.run([merged, optimizer], feed_dict=poc11_input.feed_dict(True, _x, _y))
                train_writer.add_summary(summary, step)

    print ("\nTraining complete!")

    save_path = saver.save(sess, os.path.join(FLAGS.orchid_summaries_dir, "model.ckpt"))
    print("Model saved in file: %s" % save_path)

    train_writer.close()
    test_writer.close()


if __name__ == '__main__':
    main()
