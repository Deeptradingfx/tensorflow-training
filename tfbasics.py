#!/usr/bin/env python3

import tensorflow as tf

x1 = tf.constant(5)
x2 = tf.constant(6)

result = tf.multiply(x1, x2)

print(result)

# sess = tf.Session()
# print(sess.run(result))
# sess.close()

with tf.Session() as sess:
	output = sess.run(result)
	print(output)

print(output)

a = tf.constant(3.0, tf.float32)
b = tf.constant(4.0) # implicit tf.float32

print(a, b)

with tf.Session() as sess:
        print(sess.run([a, b]))
        c = tf.add(a, b)
        print('c:', c)
        print('sess.run(c):', sess.run(c))

node1 = tf.placeholder(tf.float32)
node2 = tf.placeholder(tf.float32)
adder_node = node1 + node2 # shortcut for tf.add(node1, node2)

with tf.Session() as sess:
        print(sess.run(adder_node, {node1: 3, node2: 4.5}))
        print(sess.run(adder_node, {node1: [1, 3], node2: [2, 4]}))

add_and_triple = adder_node * 3

with tf.Session() as sess:
        print(sess.run(add_and_triple, {node1: 3, node2: 4.5}))
