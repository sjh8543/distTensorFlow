import tensorflow as tf

def inference(x_ph, keep_prob):

    with tf.name_scope('hidden1'):
        weights = tf.Variable(tf.truncated_normal([data_num * NUM_CLASSES, NUM_HIDDEN1], stddev=stddev), name='weights')
        biases = tf.Variable(tf.zeros([NUM_HIDDEN1]), name='biases')
        hidden1 = tf.nn.relu(tf.matmul(x_ph, weights) + biases)

    with tf.name_scope('hidden2'):
        weights = tf.Variable(tf.truncated_normal([NUM_HIDDEN1, NUM_HIDDEN2], stddev=stddev), name='weights')
        biases = tf.Variable(tf.zeros([NUM_HIDDEN2]), name='biases')
        hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)

    # DropOut
    dropout = tf.nn.dropout(hidden2, keep_prob)

    with tf.name_scope('softmax'):
        weights = tf.Variable(tf.truncated_normal([NUM_HIDDEN2, NUM_CLASSES], stddev=stddev), name='weights')
        biases = tf.Variable(tf.zeros([NUM_CLASSES]), name='biases')
        y = tf.nn.softmax(tf.matmul(dropout, weights) + biases)

    return y


def loss(y, target):

    softmax_target = tf.nn.softmax(target)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(y, softmax_target, name='xentropy')
    loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')

    return loss


def training(sess, train_step, loss, x_train_array, y_train_array):

    summary_op = tf.merge_all_summaries()

    init = tf.initialize_all_variables()
    sess.run(init)

    summary_writer = tf.train.SummaryWriter(LOG_DIR, graph_def=sess.graph_def)

    for i in range(int(len(x_train_array) / bach_size)):
        batch_xs = getBachArray(x_train_array, i * bach_size, bach_size)
        batch_ys = getBachArray(y_train_array, i * bach_size, bach_size)
        sess.run(train_step, feed_dict={x_ph: batch_xs, y_ph: batch_ys, keep_prob: 0.8})
        ce = sess.run(loss, feed_dict={x_ph: batch_xs, y_ph: batch_ys, keep_prob: 1.0})

        summary_str = sess.run(summary_op, feed_dict={x_ph: batch_xs, y_ph: batch_ys, keep_prob: 1.0})
        summary_writer.add_summary(summary_str, i)

