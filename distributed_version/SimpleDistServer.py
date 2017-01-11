import tensorflow as tf

#Flags for defining the tf.train.ClusterSpec
tf.app.flags.DEFINE_string("ps_hosts","",
                           "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("worker_hosts","",
                           "Comma-separated list of hostname:port paris")

#Flags for defining the tf.train.Server
tf.app.flags.DEFINE_string("job_name","","One of 'ps' , 'worker'")
tf.app.flags.DEFINE_integer("task_index",0,"Index of task within the job") 

FLAGS = tf.app.flags.FLAGS

#define main function for ditributed server
def main(_):
    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")
    cluster = tf.train.ClusterSpec( {"ps":ps_hosts,"worker":worker_hosts} )
    server = tf.train.Server( cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index )

    print("Starting server #{}".format(FLAGS.task_index))

    server.start()
    server.join()

if __name__ == "__main__":
    tf.app.run()
