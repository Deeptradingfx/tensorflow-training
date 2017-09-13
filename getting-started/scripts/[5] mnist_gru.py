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

"""Implementation of GRU+SVM model for MNIST classification"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

__version__ = '0.2'
__author__ = 'Abien Fred Agarap'

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import time
import sys

BATCH_SIZE = 128
CELL_SIZE = 32
CHUNK_SIZE = 28
HM_EPOCHS = 10
LEARNING_RATE = 0.01
NUM_CHUNKS = 28
NUM_CLASSES = 10
SVM_C = 0.5


class GruSvm:
    def __init__(self, data_input, checkpoint_path, model_name, log_path):
        self.data_input = data_input
        self.checkpoint_path = checkpoint_path
        self.model_name = model_name
        self.log_path = log_path

        def __graph__():
            """Build the inference graph"""

            with tf.name_scope('input'):
                # [BATCH_SIZE, NUM_CHUNKS, CHUNK_SIZE]
                x_input = tf.placeholder(dtype=tf.float32, shape=[None, NUM_CHUNKS, CHUNK_SIZE], name='x_input')

                # []
                y_input = tf.placeholder(dtype=tf.float32, shape=[None, NUM_CLASSES], name='y_input')

            initial_state = tf.placeholder(dtype=tf.float32, shape=[None, CELL_SIZE], name='initial_state')

            cell = tf.contrib.rnn.GRUCell(CELL_SIZE)
            outputs, states = tf.nn.dynamic_rnn(cell, x_input, initial_state=initial_state, dtype=tf.float32)

            states = tf.identity(states, name='H')

            with tf.name_scope('training_ops'):
                with tf.name_scope('weights'):
                    xav_init = tf.contrib.layers.xavier_initializer
                    weight = tf.get_variable('weights', shape=[CELL_SIZE, NUM_CLASSES], initializer=xav_init())
                    self.variable_summaries(weight)
                with tf.name_scope('biases'):
                    bias = tf.get_variable('biases', initializer=tf.constant(0.1, shape=[NUM_CLASSES]))
                    self.variable_summaries(bias)
                with tf.name_scope('Wx_plus_b'):
                    final_state = tf.transpose(outputs, [1, 0, 2])
                    last = tf.gather(final_state, int(final_state.get_shape()[0]) - 1)
                    output = tf.matmul(last, weight) + bias
            tf.summary.histogram('pre-activations', output)

            with tf.name_scope('loss'):
                cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y_input))
            tf.summary.scalar('loss', cost)

            optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cost)

            with tf.name_scope('accuracy'):
                predicted_class = tf.identity(output, name='prediction')
                with tf.name_scope('correct_prediction'):
                    correct = tf.equal(tf.argmax(predicted_class, 1), tf.argmax(y_input, 1))
                with tf.name_scope('accuracy'):
                    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
            tf.summary.scalar('accuracy', accuracy)

            merged = tf.summary.merge_all()

            self.x_input = x_input
            self.y_input = y_input
            self.state = initial_state
            self.states = states
            self.loss = cost
            self.optimizer = optimizer
            self.accuracy_op = accuracy
            self.merged = merged

        sys.stdout.write('\n<log>Building Graph...')
        __graph__()
        sys.stdout.write('</log>\n')

    def train(self):
        """Train the model"""

        timestamp = str(time.asctime())
        train_writer = tf.summary.FileWriter(self.log_path + timestamp + '-training', graph=tf.get_default_graph())
        saver = tf.train.Saver(max_to_keep=10)

        current_state = np.zeros([BATCH_SIZE, CELL_SIZE])

        init_op = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init_op)

            checkpoint = tf.train.get_checkpoint_state(self.checkpoint_path)

            if checkpoint and checkpoint.model_checkpoint_path:
                saver.restore(sess, tf.train.latest_checkpoint(self.checkpoint_path))

            for epoch in range(HM_EPOCHS):
                epoch_loss = 0
                for _ in range(int(self.data_input.train.num_examples / BATCH_SIZE)):
                    epoch_x, epoch_y = self.data_input.train.next_batch(BATCH_SIZE)

                    epoch_x = epoch_x.reshape((BATCH_SIZE, NUM_CHUNKS, CHUNK_SIZE))

                    feed_dict = {self.x_input: epoch_x, self.y_input: epoch_y,
                                 self.state: current_state}

                    summary, _, epoch_loss, train_accuracy, next_state = sess.run(
                        [self.merged, self.optimizer, self.loss, self.accuracy_op, self.states],
                        feed_dict=feed_dict)

                if epoch % 10 == 0:
                    saver.save(sess, self.checkpoint_path + self.model_name, global_step=epoch)

                train_writer.add_summary(summary, epoch)
                print('Epoch : {} completed out of {}, loss : {}, accuracy : {}'.format(epoch, HM_EPOCHS,
                                                                                        epoch_loss, train_accuracy))

                current_state = next_state

            train_writer.close()

            saver.save(sess, self.checkpoint_path + self.model_name, global_step=epoch)

            x_ = self.data_input.test.images.reshape((-1, NUM_CHUNKS, CHUNK_SIZE))
            y_ = self.data_input.test.labels

            test_accuracy = sess.run(self.accuracy_op, feed_dict={self.x_input: x_, self.y_input: y_,
                                                                  self.state: np.zeros([10000, CELL_SIZE])})

            print('Accuracy : {}'.format(test_accuracy))

    @staticmethod
    def variable_summaries(var):
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)


def main(argv):
    mnist = input_data.read_data_sets(argv.dataset, one_hot=True)

    model = GruSvm(data_input=mnist, checkpoint_path=argv.checkpoint_path,
                   model_name=argv.model_name, log_path=argv.log_path)
    model.train()


def parse_args():
    parser = argparse.ArgumentParser(description='GRU for MNIST Classification')
    group = parser.add_argument_group('Arguments')
    group.add_argument('-d', '--dataset', required=True, type=str,
                       help='path of the MNIST dataset')
    group.add_argument('-c', '--checkpoint_path', required=True, type=str,
                       help='path to save the trained model')
    group.add_argument('-m', '--model_name', required=True, type=str,
                       help='name for the trained model')
    group.add_argument('-l', '--log_path', required=True, type=str,
                       help='path where to save program logs')
    arguments = parser.parse_args()
    return arguments


if __name__ == '__main__':
    args = parse_args()

    main(argv=args)
