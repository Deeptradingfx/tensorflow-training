import cv2
import numpy as np
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants


camera = cv2.VideoCapture(0)

def grab_video_feed():
	grabbed, frame = camera.read()
	return frame if grabbed else None

# model = tf.contrib.saved_model.load_keras_model('model/1547123657')	
	
graph = tf.Graph()
with graph.as_default():
	with tf.Session(graph=graph) as sess:
		tf.saved_model.loader.load(sess, [tag_constants.SERVING], 'model')
		features_placeholder = graph.get_tensor_by_name('features:0')
		dataset_initializer = graph.get_operation_by_name('feature_init')
		prediction = graph.get_tensor_by_name('predictions:0')

		while True:
			frame = grab_video_feed()

			cv2.imshow('main', frame)

			frame = cv2.resize(frame, (784, 1), interpolation=cv2.INTER_CUBIC)
			frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

			sess.run(dataset_initializer, feed_dict={features_placeholder: frame})

			predicted_class = sess.run(prediction, feed_dict={features_placeholder: frame})
			print('Predicted class : {}'.format(np.argmax(predicted_class)))

# print('Predicted class : {}'.format(np.argmax(model.predict(frame))))

			if cv2.waitKey(1) & 0xFF == ord('q'):
				break

camera.release()
cv2.destroyAllWindows()