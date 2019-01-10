import numpy as np
import tensorflow as tf
from tensorflow.contrib import eager as tfe
from tensorflow.keras.datasets import mnist

# tf.enable_eager_execution()
tf.set_random_seed(1234)
np.random.seed(9876)


(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 784).astype(np.float32)
x_test = x_test.reshape(-1, 784).astype(np.float32)
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

idx_perm = np.random.RandomState(5).permutation(x_train.shape[0])
x_train, y_train = x_train[idx_perm], y_train[idx_perm]

# class NeuralNetwork(tf.keras.Model):
# 	def __init__(self):
# 		super(NeuralNetwork, self).__init__()
# 		self.hidden_layer_1 = tf.layers.Dense(512, input_shape=(784, ), activation=tf.nn.relu, use_bias=True)
# 		self.hidden_layer_2 = tf.layers.Dense(256, activation=tf.nn.relu, use_bias=True)
# 		self.dropout = tf.layers.Dropout()
# 		self.output_layer = tf.layers.Dense(10, use_bias=True, activation=None)

# 	def call(self, x):
# 		h1 = self.hidden_layer_1(x)
# 		h2 = self.hidden_layer_2(h1)
# 		h2 = self.dropout(h2)
# 		return self.output_layer(h2)

# model = NeuralNetwork()
# opt = tf.train.AdamOptimizer(learning_rate=1e-3)

# def loss(model, logits, labels):
# 	return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model(logits), labels=labels))

# def train_step(loss, model, opt, x, y):
# 	opt.minimize(lambda : loss(model, x, y), global_step=tf.train.get_or_create_global_step())

# dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))

# epochs = 1
# accuracy_history = np.zeros(epochs)
# writer = tf.contrib.summary.create_file_writer('tmp')

# with writer.as_default():
# 	with tf.contrib.summary.always_record_summaries():
# 		for epoch in range(epochs):
# 			accuracy = tfe.metrics.Accuracy()
# 			for x_batch, y_batch in tfe.Iterator(dataset.shuffle(1000).batch(512)):
# 				tf.contrib.summary.scalar('loss', loss(model, x_batch, y_batch))
# 				train_step(loss, model, opt, x_batch, y_batch)
# 				accuracy(tf.argmax(model(tf.constant(x_batch)), axis=1), tf.argmax(tf.constant(y_batch), axis=1))
# 				tf.contrib.summary.scalar('accuracy', accuracy.result().numpy())
# 			accuracy_history[epoch] = accuracy.result().numpy()

# test_accuracy = tf.contrib.metrics.accuracy(predictions=tf.argmax(model(tf.constant(x_test)), axis=1),
# 											labels=tf.argmax(tf.constant(y_test), axis=1))
# print('Test accuracy : {}'.format(test_accuracy.numpy()))

with tf.Graph().as_default() as graph:

	features = tf.placeholder(shape=[None, 784], dtype=tf.float32, name='features')
	labels = tf.placeholder(shape=[None, 10], dtype=tf.float32, name='labels')

	# dataset = tf.data.Dataset.from_tensor_slices((features, labels)).shuffle(1000).batch(512)
	feature_data = tf.data.Dataset.from_tensor_slices(features).batch(512)
	feature_itr = tf.data.Iterator.from_structure(feature_data.output_types, feature_data.output_shapes)
	next_example = feature_itr.get_next()
	feature_itr_op = feature_itr.make_initializer(feature_data, name='feature_init')

	label_data = tf.data.Dataset.from_tensor_slices(labels).batch(512)
	label_itr = tf.data.Iterator.from_structure(label_data.output_types, label_data.output_shapes)
	next_label = label_itr.get_next()
	label_itr_op = label_itr.make_initializer(label_data, name='label_init')

	epochs = 20

	model = tf.layers.dense(inputs=next_example, units=512, activation=tf.nn.relu, use_bias=True)
	model = tf.layers.dense(inputs=model, units=256, activation=tf.nn.relu, use_bias=True)
	model = tf.layers.dropout(inputs=model, rate=0.55)
	predictions = tf.layers.dense(inputs=model,
		units=10,
		activation=None,
		activity_regularizer=tf.contrib.layers.l2_regularizer(scale=1e-2))
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=predictions, labels=next_label))
	train_op = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss=loss)
	accuracy = tf.contrib.metrics.accuracy(predictions=tf.argmax(tf.nn.softmax(predictions), axis=1),
		labels=tf.argmax(next_label, axis=1))


with tf.Session(graph=graph) as sess:
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
	outputs = {'prediction': tf.nn.softmax(predictions, name='predictions')}
	tf.saved_model.simple_save(sess, 'model', inputs, outputs)

# model = tf.keras.models.Sequential()
# model.add(tf.layers.Dense(512, input_shape=(784, ), activation=tf.nn.relu, use_bias=True))
# model.add(tf.layers.Dense(256, activation=tf.nn.relu, use_bias=True))
# model.add(tf.layers.Dropout())
# model.add(tf.layers.Dense(10, activation=tf.nn.softmax, activity_regularizer=tf.keras.layers.ActivityRegularization(l2=1e-2)))
# model.compile(loss='categorical_crossentropy',
# 				optimizer=tf.train.AdamOptimizer(learning_rate=1e-3),
# 				metrics=['accuracy'])
# model.fit(x_train, y_train, epochs=20, batch_size=512)
# score = model.evaluate(x_test, y_test)
# print('Test accuracy : {}'.format(score[1]))

# tf.contrib.saved_model.save_keras_model(model=model, saved_model_path='model')