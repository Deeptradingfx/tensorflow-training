# MIT License
# 
# Copyright (c) 2017 Abien Fred Agarap
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


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
            # initialize weight matrix
            weight = tf.Variable(initial_value=[.3], name='weight', dtype=tf.float32)
            
            # placeholder for input data
            x_input = tf.placeholder(dtype=tf.float32, name='x_input')
            
            # initialize the bias matrix
            # bias is used to avoid dead neurons
            bias = tf.Variable(initial_value=[-.3], name='bias', dtype=tf.float32)
            
            # placeholder for actual data
            y = tf.placeholder(dtype=tf.float32, name='actual_values')
            
            # linear regression equation
            # y = Wx + b
            linear_model = weight * x_input + bias
            
            # determine the amount of error in the regression
            loss = tf.reduce_sum(tf.square(linear_model - y))
            
            # train the regression to reach optimal parameters W and b
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
                # load the data
                x_train, y_train = self.data_input
                
                # create input dictionary to feed to the train operation
                feed_dict = {self.x_input: x_train, self.y: y_train}
                
                # run the train operation with the previously-defined input dict
                sess.run(self.train_op, feed_dict=feed_dict)
                
                # get the learnt parameters and the error (loss)
                curr_w, curr_b, curr_loss = sess.run([self.weight, self.bias, self.loss], feed_dict=feed_dict)
                
            # print the learn parameters of the regression and its loss
            print("W: {}, b: {}, loss: {}".format(curr_w, curr_b, curr_loss))


if __name__ == '__main__':
    # arbitrary values to test
    data = [[1, 2, 3, 4], [0, -1, -2, -3]]

    model = LinearRegression(data_input=data)
    model.train()
