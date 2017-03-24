from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

tf.logging.set_verbosity(tf.logging.INFO)

def cnn_model_fn(features, labels, mode):
    '''Model function for CNN'''
    # Input Layer
    input_layer = tf.reshape(features, [-1, 760, 760, 3])

    # Conv layer #1
    conv1 = tf.layers.conv3d(
        inputs=input_layer,
        filters=96,
        kernel_size=5,
        padding='same',
        activation=tf.nn.relu)

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling3d(inputs=conv1, pool_size=3, strides=3)

    # Conv layer #2
    conv2 = tf.layers.conv3d(
        inputs=pool1,
        filters=256,
        kernel_size=5,
        padding='same',
        activation=tf.nn.relu)

    # Pooling Layer #2
    pool2 = tf.layers.max_pooling3d(inputs=conv2, pool_size=2, strides=2)

    # Conv layer #3
    conv3 = tf.layers.conv3d(
        inputs=pool2,
        filters=512,
        kernel_size=3,
        padding='same',
        activation=tf.nn.relu
    )

    # Conv layer #4
    conv4 = tf.layers.conv3d(
        inputs=conv3,
        filters=512,
        kernel_size=3,
        padding='same',
        activation=tf.nn.relu
    )

    # Conv layer #5
    conv5 = tf.layers.conv3d(
        inputs=conv4,
        filters=512,
        kernel_size=3,
        padding='same',
        activation=tf.nn.relu
    )

    # Pooling Layer #3
    pool3 = tf.layers.max_pooling3d(inputs=conv5, pool_size=3, strides=3)
    pool3_flat = tf.reshape(pool3, [-1, 3 * 3 * 512])
    
    # Dense Layer #1
    dense = tf.layers.dense(inputs=pool3_flat, units=4096, activation=tf.nn.relu)

    # Dropout: p = 0.5
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.5, training=mode == learn.ModeKeys.TRAIN)

    # Dense Layer #2
    dense2 = tf.layers.dense(inputs=dropout, units=4096, activation=tf.nn.relu)

    # Dropout: p = 0.5
    dropout2 = tf.layers.dropout(
        inputs=dense2, rate=0.5, training=mode == learn.ModeKeys.TRAIN)

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout2, units=4)


if __name__ == '__main__':
    tf.app.run()