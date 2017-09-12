"""Implementation of Linear Regression"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__version__ = '0.1'
__author__ = 'Abien Fred Agarap'

import tensorflow as tf
import sys


class LinearRegression:

    def __init__(self, data_input):
        self.data_input = data_input

        def __graph__():
            weight = tf.Variable(initial_value=[.3], name='weight', dtype=tf.float32)
            x_input = tf.placeholder(dtype=tf.float32, name='x_input')
            bias = tf.Variable(initial_value=[-.3], name='bias', dtype=tf.float32)

            y = tf.placeholder(dtype=tf.float32, name='actual_values')

            linear_model = weight * x_input + bias

            loss = tf.reduce_sum(tf.square(linear_model - y))

            train_op = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

            self.weight = weight
            self.x_input = x_input
            self.bias = bias
            self.y = y
            self.linear_model = linear_model
            self.loss = loss
            self.train_op = train_op

        sys.stdout.write('\n<log> Building graph...')
        __graph__()
        sys.stdout.write('</log>\n')

    def train(self):
        init_op = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init_op)

            for step in range(1000):
                x_train, y_train = self.data_input

                feed_dict = {self.x_input: x_train, self.y: y_train}

                sess.run(self.train_op, feed_dict=feed_dict)

                curr_w, curr_b, curr_loss = sess.run([self.weight, self.bias, self.loss], feed_dict=feed_dict)

            print("W: {}, b: {}, loss: {}".format(curr_w, curr_b, curr_loss))


if __name__ == '__main__':
    data = [[1, 2, 3, 4], [0, -1, -2, -3]]

    model = LinearRegression(data_input=data)
    model.train()
