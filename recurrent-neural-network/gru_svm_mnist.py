# Copyright 2017 Abien Fred Agarap. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Implementation of GRU+SVM model for MNIST classification"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import time

mnist = input_data.read_data_sets('/home/darth/Projects/Artificial Intelligence/tensorflow/tutorial/MNIST_data',
                                  one_hot=True)

BATCH_SIZE = 128
CELL_SIZE = 32
CHUNK_SIZE = 28
HM_EPOCHS = 10
LEARNING_RATE = 0.01
NUM_CHUNKS = 28
NUM_CLASSES = 10
SVM_C = 0.5

CHECKPOINT_PATH = 'checkpoint/'
MODEL_NAME = 'model.ckpt'


x = tf.placeholder(dtype=tf.float32, shape=[None, NUM_CHUNKS, CHUNK_SIZE], name='x_input')
y = tf.placeholder(dtype=tf.float32, name='y_input')


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


def recurrent_neural_network(x_input):

    x_input = tf.transpose(x_input, [1, 0, 2])
    x_input = tf.reshape(x_input, [-1, CHUNK_SIZE])
    x_input = tf.split(x_input, NUM_CHUNKS, 0)

    cell = tf.contrib.rnn.GRUCell(CELL_SIZE)

    outputs, states = tf.nn.static_rnn(cell, x_input, dtype=tf.float32)

    with tf.name_scope('training_ops'):
        with tf.name_scope('weights'):
            weight = tf.get_variable('weights', initializer=tf.random_normal([CELL_SIZE, NUM_CLASSES], stddev=0.01))
            variable_summaries(weight)
        with tf.name_scope('biases'):
            bias = tf.get_variable('biases', initializer=tf.constant(0.1, shape=[NUM_CLASSES]))
            variable_summaries(bias)
        with tf.name_scope('Wx_plus_b'):
            output = tf.matmul(outputs[-1], weight) + bias
        tf.summary.histogram('pre-activations', output)

    return output, weight, states


def train_neural_network(x):
    prediction, layer, states = recurrent_neural_network(x)

    with tf.name_scope('loss'):
        regularization_loss = 0.5 * tf.reduce_sum(tf.square(layer['weights']))
        hinge_loss = tf.reduce_sum(tf.square(tf.maximum(tf.zeros([BATCH_SIZE, NUM_CLASSES]),
                                                        1 - tf.cast(y, tf.float32) * prediction)))
        with tf.name_scope('loss'):
            cost = regularization_loss + SVM_C * hinge_loss
    tf.summary.scalar('loss', cost)

    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cost)

    with tf.name_scope('accuracy'):
        predicted_class = tf.sign(prediction)
        predicted_class = tf.identity(predicted_class, name='prediction')
        with tf.name_scope('correct_prediction'):
            correct = tf.equal(tf.argmax(predicted_class, 1), tf.argmax(y, 1))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
    tf.summary.scalar('accuracy', accuracy)

    merged = tf.summary.merge_all()

    timestamp = str(time.asctime())
    train_writer = tf.summary.FileWriter('logs/rnn/' + timestamp + '-training', graph=tf.get_default_graph())
    saver = tf.train.Saver(max_to_keep=10)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        checkpoint = tf.train.get_checkpoint_state(CHECKPOINT_PATH)

        if checkpoint and checkpoint.model_checkpoint_path:
            saver.restore(sess, tf.train.latest_checkpoint(CHECKPOINT_PATH))

        for epoch in range(HM_EPOCHS):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples / BATCH_SIZE)):
                epoch_x, epoch_y = mnist.train.next_batch(BATCH_SIZE)
                epoch_y[epoch_y == 0] = -1

                epoch_x = epoch_x.reshape((BATCH_SIZE, NUM_CHUNKS, CHUNK_SIZE))

                feed_dict = {x: epoch_x, y: epoch_y, }

                summary, _, c, train_accuracy = sess.run([merged, optimizer, cost, accuracy], feed_dict=feed_dict)

                epoch_loss = c

            if epoch % 10 == 0:
                saver.save(sess, CHECKPOINT_PATH + MODEL_NAME, global_step=epoch)

            train_writer.add_summary(summary, epoch)
            print('Epoch : {} completed out of {}, loss : {}, accuracy : {}'.format(epoch, HM_EPOCHS,
                                                                                    epoch_loss, train_accuracy))

        train_writer.close()

        saver.save(sess, CHECKPOINT_PATH + MODEL_NAME, global_step=epoch)

        x_ = mnist.test.images.reshape((-1, NUM_CHUNKS, CHUNK_SIZE))
        y_ = mnist.test.labels
        y_[y_ == 0] = -1

        test_accuracy = sess.run(accuracy, feed_dict={x: x_, y: y_})

        print('Accuracy : {}'.format(test_accuracy))


if __name__ == '__main__':
    train_neural_network(x)
