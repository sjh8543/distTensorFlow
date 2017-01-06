import numpy as np
import tensorflow as tf
from datetime import datetime

x = tf.placeholder( tf.float32 , 100000000 )

#Generate Model
mean = tf.reduce_mean( x )

with tf.Session() as sess :

    t=datetime.now()
    result = sess.run(mean,feed_dict={x:np.random.random(100000000)})
    t2=datetime.now()
    t3=t2-t
    print result
    print t3.microseconds

