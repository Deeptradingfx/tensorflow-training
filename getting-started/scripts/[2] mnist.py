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

"""Softmax Regression for MNIST Classification"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__version__ = '0.1'
__author__ = 'Abien Fred Agarap'

import argparse
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import sys


class LogisticRegression:

    def __init__(self, data_input):
        self.data_input = data_input

        def __graph__():
            # input placeholder for features (x) and labels (y)
            x = tf.placeholder(tf.float32, [None, 784], name='x_input')
            y = tf.placeholder(tf.float32, [None, 10], name='actual_label')
            
            # parameters to learn
            w = tf.Variable(tf.zeros([784, 10]), name='weights')
            b = tf.Variable(tf.zeros([10]), name='biases')

            # numerically unstable version
            # y = tf.nn.softmax(tf.matmul(x, W) + b)
            # cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
            
            # get the predicted labels
            y_ = tf.matmul(x, w) + b
            y_ = tf.identity(y_, name='logits')
            
            # just for future use, i.e.
            # writing a classifier based on this model
            predictions = tf.nn.softmax(y, name='predictions')
            
            # get the loss of the training
            cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_))
            
            # train using SGD algorithm
            train_op = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(cross_entropy)
            
            # check if the actual labels and predicted labels match
            correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))
            
            # get the % of correct predictions
            accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            self.x = x
            self.w = w
            self.b = b
            self.y = y
            self.predictions = predictions
            self.y_ = y_
            self.cross_entropy = cross_entropy
            self.train_op = train_op
            self.accuracy_op = accuracy_op

        sys.stdout.write('\n<log> Building graph...')
        __graph__()
        sys.stdout.write('</log>\n')

    def train(self):
        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)

            for step in range(1000):
                # train by batch of 100
                batch_xs, batch_ys = self.data_input.train.next_batch(100)
                
                # input dictionary
                feed_dict = {self.x: batch_xs, self.y: batch_ys}
                
                # run the train operation
                _ = sess.run([self.train_op], feed_dict=feed_dict)
                
                # every 100th step and at step 0,
                # display the loss and accuracy of the model
                if step % 100 == 0:
                    loss, accuracy = sess.run([self.cross_entropy, self.accuracy_op], feed_dict=feed_dict)

                    print('step [{}] -- loss: {}, accuracy: {}'.format(step, loss, accuracy))

            feed_dict = {self.x: self.data_input.test.images, self.y: self.data_input.test.labels}
            
            # get the accuracy of the train model
            # using unseen data
            test_accuracy = sess.run(self.accuracy_op, feed_dict=feed_dict)
            print('Test Accuracy: {}'.format(test_accuracy))


def parse_args():
    parser = argparse.ArgumentParser(description='Logistic Regression for MNIST Classification')
    group = parser.add_argument_group('Arguments')
    group.add_argument('-d', '--dataset', required=True, type=str,
                       help='path of the MNIST dataset')
    arguments = parser.parse_args()
    return arguments


if __name__ == '__main__':
    args = parse_args()

    mnist = input_data.read_data_sets(args.dataset, one_hot=True)

    model = LogisticRegression(data_input=mnist)
    model.train()
