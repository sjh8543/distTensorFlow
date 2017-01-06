import tensorflow as tf

cluster = tf.train.ClusterSpec( {"local":["localhost:2222","localhost:2223"]} )

c = []
with tf.device("/job:local/task:0"):
    print "tatk:0"
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3])
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2])
    c.append(tf.matmul(a, b))


with tf.device("/job:local/task:1"):
    print "task:1"
    sum = tf.add_n(c)
    

with tf.Session("grpc://localhost:2222") as sess:
    tf.global_variables_initializer().run(session=sess) 

