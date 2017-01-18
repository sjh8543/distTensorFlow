import sys
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

"""have to pass serrion url"""
session_url = sys.argv[0]

#get mnist data set
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
    
#initialize variables     
x = tf.placeholder(tf.float32,[None, 784])
#generate cross-entropy function
y_ = tf.placeholder( tf.float32,[None,10])

# assign parameter value to parameter server
with tf.device("/job:ps/task:0"):
    w = tf.Variable(tf.zeros([784,10]))
    b = tf.Variable(tf.zeros([10]))

with tf.device("/job:worker/task:0"):
    #compile model we would like to use in trainning 
    y = tf.nn.softmax(tf.matmul(x,w) + b)

with tf.device("/job:worker/task:1"):
    #initialize loss function & define gradient optimizer
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y),reduction_indices=[1]))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
        

#init = tf.initialize_all_variables()

with tf.Session("grpc://"+session_url) as sess: 
    init = tf.global_variables_initializer() 
    sess.run(init)
    for i in range(1000):
        batch_xs, batch_ys=mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x:batch_xs,y_:batch_ys})

    #evaluate process
    correct_prediction = tf.equal( tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print( sess.run( accuracy, feed_dict={x: mnist.test.images,y_: mnist.test.labels}) )
