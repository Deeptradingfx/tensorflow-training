import numpy as np
import tensorflow as tf

x = tf.placeholder(tf.float32, shape=(1024, 1024))
y = tf.matmul(x, x)

with tf.Session() as sess:
    rand_array = np.random.rand(1024, 1024)
    print(sess.run(y, feed_dict={x : rand_array}))
