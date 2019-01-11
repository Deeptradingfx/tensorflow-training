import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
import timeit

tf.set_random_seed(0)
np.random.seed(12)

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train.astype(np.float32), x_test.astype(np.float32)
y_train, y_test = tf.keras.utils.to_categorical(y_train), tf.keras.utils.to_categorical(y_test)

idx_perm = np.random.RandomState(3).permutation(x_train.shape[0])
x_train, y_train = x_train[idx_perm], y_train[idx_perm]

with tf.Graph().as_default() as graph:
	features = tf.placeholder(dtype=tf.float32, shape=[None, x_train.shape[1], x_train.shape[2]], name='features')
	labels = tf.placeholder(dtype=tf.float32, shape=[None, y_train.shape[1]], name='labels')

	feature_data = tf.data.Dataset.from_tensor_slices(features).batch(512)
	feature_itr = tf.data.Iterator.from_structure(feature_data.output_types, feature_data.output_shapes)
	next_example = feature_itr.get_next()
	feature_itr_op = feature_itr.make_initializer(feature_data, name='feature_init')

	label_data = tf.data.Dataset.from_tensor_slices(labels).batch(512)
	label_itr = tf.data.Iterator.from_structure(label_data.output_types, label_data.output_shapes)
	next_label = label_itr.get_next()
	label_itr_op = label_itr.make_initializer(label_data, name='label_init')

	epochs = 20
	
	x = tf.reshape(next_example, shape=[-1, x_train.shape[1], x_train.shape[2], 1])

	conv_1 = tf.layers.conv2d(inputs=x, filters=32, kernel_size=5, activation=tf.nn.relu)
	pool_1 = tf.layers.max_pooling2d(conv_1, 2, 2)
	conv_2 = tf.layers.conv2d(inputs=pool_1, filters=64, kernel_size=3, activation=tf.nn.relu)
	pool_2 = tf.layers.max_pooling2d(conv_2, 2, 2)
	fc_1 = tf.contrib.layers.flatten(pool_2)
	fc_1 = tf.layers.dense(inputs=fc_1, units=1024)
	dropout = tf.layers.dropout(inputs=fc_1, rate=0.55)
	logits = tf.layers.dense(inputs=dropout,
		units=y_train.shape[1],
		activation=None,
		activity_regularizer=tf.contrib.layers.l2_regularizer(scale=1e-2))
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=next_label))
	train_op = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss=loss)
	accuracy = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(logits), axis=1),
		tf.argmax(next_label, 1))

with tf.Session(graph=graph) as sess:
	start_time = timeit.default_timer()
	sess.run(tf.global_variables_initializer())
	sess.run(feature_itr_op, feed_dict={features: x_train})
	sess.run(label_itr_op, feed_dict={labels: y_train})

	for epoch in range(epochs):
		_, loss_value, accuracy_value = sess.run([train_op, loss, accuracy])
		print('Epoch {} -- Loss : {}, Accuracy : {}'.format(epoch, loss_value, accuracy_value))
	
	sess.run(feature_itr_op, feed_dict={features: x_test})
	sess.run(label_itr_op, feed_dict={labels: y_test})
	test_accuracy = sess.run(accuracy)
	print('Test Accuracy : {}'.format(test_accuracy))

	inputs = {'feature_placeholder': features}
	outputs = {'prediction': tf.nn.softmax(logits, name='predictions')}
	tf.saved_model.simple_save(sess, 'model', inputs, outputs)
	print('took {} seconds'.format(timeit.default_timer() - start_time))
