import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


if __name__ == "__main__" :
    print 'test project'
    print 'get input data from the library'
    #get mnist data set
    mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
    
    #initialize variables     
    x = tf.placeholder(tf.float32,[None, 784])
    w = tf.Variable(tf.zeros([784,10]))
    b = tf.Variable(tf.zeros([10]))

    #compile model we would like to use in trainning 
    y = tf.nn.softmax(tf.matmul(x,w) + b)
    
    #generate cross-entropy function
    y_ = tf.placeholder( tf.float32,[None,10])
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y),reduction_indices=[1]))
    tf.summary.scalar('cross_entropy', cross_entropy)    
    
    #train step
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
    merged = tf.summary.merge_all()
#    init = tf.initialize_all_variables()
    sess = tf.Session()
    tf.global_variables_initializer().run(session=sess)
#    sess.run(init)

    train_writer = tf.summary.FileWriter('tmp/train',sess.graph)
    test_writer = tf.summary.FileWriter('tmp/test') 
    for i in range(1000):
        batch_xs, batch_ys=mnist.train.next_batch(100)
        summary, _ = sess.run([merged,train_step], feed_dict={x:batch_xs,y_:batch_ys})
        train_writer.add_summary(summary,i)

    #evaluate process
    correct_prediction = tf.equal( tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    summary,acc = sess.run( [merged,accuracy], feed_dict={x: mnist.test.images,y_: mnist.test.labels})
    test_writer.add_summary(summary)
    print acc

    print 'train end'     
