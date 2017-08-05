import os
import tensorflow as tf
import orchid11
import orchid11_input
import orchid11_dataset

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

    with tf.name_scope('total'):
        #output_layer = tf.nn.softmax(output_layer)
        #cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(output_layer), reduction_indices=[1]))
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output_layer, labels=_y))

    tf.summary.scalar('cross_entropy', cross_entropy)

    with tf.name_scope('train'):
        optimizer = tf.train.GradientDescentOptimizer(FLAGS.learning_rate).minimize(cross_entropy)
        #optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cross_entropy)

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

    use_new_dataset = False
    if use_new_dataset:
        ord11 = orchid11_dataset.read_data_sets(one_hot=True)

    for step in range(FLAGS.epochs):
        if step % 40 == 0:  # Record summaries and test-set accuracy
            if use_new_dataset:
                batch_x, batch_y = ord11.validation.next_batch(FLAGS.batch_size)
            else:
                batch_x, batch_y = orchid11_input.batch_creator('test')
            summary, acc = sess.run([merged, accuracy], feed_dict={_x: batch_x, _y: batch_y, keep_prob: 1.0})
            test_writer.add_summary(summary, step)
            print('Accuracy at step %s: %s' % (step, acc))
        else:
            if use_new_dataset:
                batch_x, batch_y = ord11.train.next_batch(FLAGS.batch_size)
            else:
                batch_x, batch_y = orchid11_input.batch_creator('train')

            if step % 1000 == 9:  # Record execution stats
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                summary, _ = sess.run([merged, optimizer],
                                      feed_dict={_x: batch_x, _y: batch_y, keep_prob: FLAGS.dropout},
                                      options=run_options,
                                      run_metadata=run_metadata)
                train_writer.add_run_metadata(run_metadata, 'step%03d' % step)
                train_writer.add_summary(summary, step)
                print('Adding run metadata for', step)
            else:  # Record a summary
                summary, _ = sess.run([merged, optimizer], feed_dict={_x: batch_x, _y: batch_y, keep_prob: FLAGS.dropout})
                train_writer.add_summary(summary, step)

    print ("\nTraining complete!")

    save_path = saver.save(sess, os.path.join(FLAGS.orchid_summaries_dir, "model.ckpt"))
    print("Model saved in file: %s" % save_path)

    train_writer.close()
    test_writer.close()


if __name__ == '__main__':
    main()
