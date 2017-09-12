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

"""2 Convolutional Layers with Max Pooling for MNIST Classification"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__version__ = '0.1'
__author__ = 'Abien Fred Agarap'

import argparse
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import sys


class CNN:
    def __init__(self, data_input):
        self.data_input = data_input

        def __graph__():
            # [BATCH_SIZE, 784]
            x = tf.placeholder(dtype=tf.float32, shape=[None, 784], name='x_input')

            # [BATCH_SIZE, 10]
            y = tf.placeholder(dtype=tf.float32, shape=[None, 10], name='actual_label')

            # First layer
            W_conv1 = self.weight_variable([5, 5, 1, 32])
            b_conv1 = self.bias_variable([32])

            x_image = tf.reshape(x, [-1, 28, 28, 1])

            h_conv1 = tf.nn.relu(self.conv2d(x_image, W_conv1) + b_conv1)
            h_pool1 = self.max_pool_2x2(h_conv1)

            # Second layer
            W_conv2 = self.weight_variable([5, 5, 32, 64])
            b_conv2 = self.bias_variable([64])

            h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2) + b_conv2)
            h_pool2 = self.max_pool_2x2(h_conv2)

            # Fully-connected layer (Dense Layer)
            W_fc1 = self.weight_variable([7 * 7 * 64, 1024])
            b_fc1 = self.bias_variable([1024])

            h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
            h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

            # Dropout
            keep_prob = tf.placeholder(tf.float32)
            h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

            # Readout layer
            W_fc2 = self.weight_variable([1024, 10])
            b_fc2 = self.bias_variable([10])

            y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

            # Train and evaluate model (optimize)
            cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_conv))
            train_op = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cross_entropy)
            correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y, 1))
            accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            self.x = x
            self.y = y
            self.keep_prob = keep_prob
            self.y_conv = y_conv
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

            for index in range(20000):
                batch_x, batch_y = mnist.train.next_batch(50)

                feed_dict = {self.x: batch_x, self.y: batch_y, self.keep_prob: 0.5}

                sess.run(self.train_op, feed_dict=feed_dict)

                if index % 100 == 0:
                    feed_dict = {self.x: batch_x, self.y: batch_y, self.keep_prob: 1.0}

                    train_accuracy = sess.run(self.accuracy_op, feed_dict=feed_dict)

                    print('step: {}, training accuracy: {}'.format(index, train_accuracy))

            feed_dict = {self.x: self.data_input.test.images,
                         self.y: self.data_input.test.labels,
                         self.keep_prob: 1.0}

            test_accuracy = sess.run(self.accuracy_op, feed_dict=feed_dict)

            print('Test Accuracy: {}'.format(test_accuracy))

    @staticmethod
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    @staticmethod
    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    @staticmethod
    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    @staticmethod
    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def parse_args():
    parser = argparse.ArgumentParser(description='2 Convolutional Layer with Max Pooling for MNIST Classification')
    group = parser.add_argument_group('Arguments')
    group.add_argument('-d', '--dataset', required=True, type=str,
                             help='path of the MNIST dataset')
    arguments = parser.parse_args()
    return arguments

if __name__ == '__main__':
    args = parse_args()

    mnist = input_data.read_data_sets(args.dataset, one_hot=True)
    model = CNN(data_input=mnist)
    model.train()
