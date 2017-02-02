from datetime import datetime
from datetime import timedelta 
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


#get mnist data set
cluster = tf.train.ClusterSpec({"ps":["ps-0.default.svc.cluster.local:2222","ps-1.default.svc.cluster.local:2222"],"worker":["worker-0.default.svc.cluster.local:2222","worker-1.default.svc.cluster.local:2222","worker-2.default.svc.cluster.local:2222"]})
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])
keep_prob = tf.placeholder(tf.float32)

with tf.device( tf.train.replica_device_setter( ps_device="/job:ps/task:0",cluster=cluster ) ):
    x_image = tf.reshape(x, [-1,28,28,1])
    W_conv1 = tf.Variable( tf.truncated_normal([5,5,1,32]))
    b_conv1 = tf.Variable( tf.constant(0.1,shape=[32]) )
    W_conv2 = tf.Variable( tf.truncated_normal([5,5,32,64]))
    b_conv2 = tf.Variable( tf.constant(0.1,shape=[64]))
    W_fc1 = tf.Variable( tf.truncated_normal([7*7*64,1024]))
    b_fc1 = tf.Variable( tf.constant(0.1,shape=[1024]))
    #readout layer to common neural network
    W_fc2 = tf.Variable( tf.truncated_normal([1024, 10]))
    b_fc2 = tf.Variable( tf.constant(0.1,shape=[10]))

    
with tf.device( tf.train.replica_device_setter( worker_device ="/job:worker/task:0" ,cluster=cluster ) ):
    h_conv1 = tf.nn.relu(tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1)
    h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')
    #second layer
    h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, W_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2)
    h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    y_conv= tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    cross_entropy = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(y_conv, y_) )
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    
with tf.device( tf.train.replica_device_setter( worker_device ="/job:worker/task:1" ,cluster=cluster ) ):
    h_conv1_1 = tf.nn.relu(tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1)
    h_pool1_1 = tf.nn.max_pool(h_conv1_1, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')
    #second layer
    h_conv2_1 = tf.nn.relu(tf.nn.conv2d(h_pool1_1, W_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2)
    h_pool2_1 = tf.nn.max_pool(h_conv2_1, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')

    h_pool2_flat_1 = tf.reshape(h_pool2_1, [-1, 7*7*64])
    h_fc1_1 = tf.nn.relu(tf.matmul(h_pool2_flat_1, W_fc1) + b_fc1)
    h_fc1_drop_1 = tf.nn.dropout(h_fc1_1, keep_prob)
    y_conv_1= tf.matmul(h_fc1_drop_1, W_fc2) + b_fc2

    cross_entropy_1 = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(y_conv_1, y_) )
    train_step_1 = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy_1)

with tf.device( tf.train.replica_device_setter( worker_device ="/job:worker/task:2" ,cluster=cluster ) ):
    h_conv1_2 = tf.nn.relu(tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1)
    h_pool1_2 = tf.nn.max_pool(h_conv1_2, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')
    #second layer
    h_conv2_2 = tf.nn.relu(tf.nn.conv2d(h_pool1_2, W_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2)
    h_pool2_2 = tf.nn.max_pool(h_conv2_2, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')

    h_pool2_flat_2 = tf.reshape(h_pool2_2, [-1, 7*7*64])
    h_fc1_2 = tf.nn.relu(tf.matmul(h_pool2_flat_2, W_fc1) + b_fc1)
    h_fc1_drop_2 = tf.nn.dropout(h_fc1_2, keep_prob)
    y_conv_2= tf.matmul(h_fc1_drop_2, W_fc2) + b_fc2

    cross_entropy_2 = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(y_conv_2, y_) )
    train_step_2 = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy_2)
    
init_op = tf.global_variables_initializer()

with tf.Session( "grpc://ps-0.default.svc.cluster.local:2222" ) as sess:   
    sess.run( init_op )
    train_writer = tf.summary.FileWriter('/log/experimental_take1',sess.graph)
    for i in range(5000):
        batch = mnist.train.next_batch(70)
        batch_1 = mnist.train.next_batch(70)
        batch_2 = mnist.train.next_batch(60)
        if i%100 == 0:
            print "train " +str(i)
            train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
            print("step %d, training accuracy %g"%(i, train_accuracy))
        sess.run( train_step , feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5} )
        sess.run( train_step_1 , feed_dict={x: batch_1[0], y_: batch_1[1], keep_prob: 0.5} )
        sess.run( train_step_2 , feed_dict={x: batch_2[0], y_: batch_2[1], keep_prob: 0.5} )
    print("test accuracy %g"%accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

