import numpy as np
import tensorflow as tf
from datetime import datetime

cluster = tf.train.ClusterSpec( {"local":["localhost:2222","localhost:2223"]} )

x = tf.placeholder( tf.float32 , 10000000 )

with tf.device("/job:local/task:1"):
    first_batch = tf.slice( x , [0] , [5000000] )
    mean1 = tf.reduce_mean( first_batch )

with tf.device("/job:local/task:0"):
    second_batch = tf.slice( x ,[5000000] , [-1] )
    mean2 = tf.reduce_mean( second_batch )
    mean = ( mean1 + mean2 ) / 2 

with tf.Session("grpc://localhost:2222") as sess :
    t = datetime.now()
    result = sess.run( mean , feed_dict={x:np.random.random(10000000)}) 
    t2 = datetime.now()
    t3 = t2 -t
    print result 
    print t3.microseconds
