import os
import tensorflow as tf
import orchid11


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

    with tf.name_scope('total'):
        #output_layer = tf.nn.softmax(output_layer)
        #cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(output_layer), reduction_indices=[1]))
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output_layer, labels=_y))

    tf.summary.scalar('cross_entropy', cross_entropy)

    with tf.name_scope('train'):
        optimizer = tf.train.GradientDescentOptimizer(orchid11.LEARNING_RATE).minimize(cross_entropy)
        #optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cross_entropy)

    # find predictions on val set
    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            pred_temp = tf.equal(tf.argmax(output_layer, 1), tf.argmax(_y, 1))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(pred_temp, tf.float32))

    tf.summary.scalar('accuracy', accuracy)

    merged = tf.summary.merge_all()

    train_writer = tf.summary.FileWriter(orchid11.FLAGS.summaries_dir + '/train', sess.graph)
    test_writer = tf.summary.FileWriter(orchid11.FLAGS.summaries_dir + '/test')

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    tf.global_variables_initializer().run()

    for step in range(orchid11.FLAGS.epochs):
        if step % 40 == 0:  # Record summaries and test-set accuracy
            summary, acc = sess.run([merged, accuracy], feed_dict=orchid11.feed_dict(False, _x, _y, keep_prob))
            test_writer.add_summary(summary, step)
            print('Accuracy at step %s: %s' % (step, acc))
        else:
            #total_batch = int(train.shape[0] / FLAGS.batch_size)
            #for i in range(total_batch):
            if step % 1000 == 9:  # Record execution stats
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                summary, _ = sess.run([merged, optimizer],
                                      feed_dict=orchid11.feed_dict(True, _x, _y, keep_prob),
                                      options=run_options,
                                      run_metadata=run_metadata)
                train_writer.add_run_metadata(run_metadata, 'step%03d' % step)
                train_writer.add_summary(summary, step)
                print('Adding run metadata for', step)
            else:  # Record a summary
                summary, _ = sess.run([merged, optimizer], feed_dict=orchid11.feed_dict(True, _x, _y, keep_prob))
                train_writer.add_summary(summary, step)

    print ("\nTraining complete!")

    save_path = saver.save(sess, os.path.join(orchid11.FLAGS.summaries_dir, "model.ckpt"))
    print("Model saved in file: %s" % save_path)

    train_writer.close()
    test_writer.close()


if __name__ == '__main__':
    main()
