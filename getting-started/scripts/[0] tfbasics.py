#!/usr/bin/env python3

import tensorflow as tf

# instantiating a TF constant
x1 = tf.constant(5)
x2 = tf.constant(6)

# defining a TF operation
result = tf.multiply(x1, x2)

# an operation must be run in a session
print(result)

# start a TF session
with tf.Session() as sess:
    # run the defined operation
    output = sess.run(result)
    
    # display the result of the op
    print(output)

# the var will live even after the sess
print(output)

