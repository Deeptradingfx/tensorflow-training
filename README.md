tensorflow-training
-------------------

## Dependencies
Install the library dependencies for this project by executing the `setup.sh`:

```
chmod +x setup.sh
./setup.sh
```

Or you may install the requirements manually:

```
pip install -r requirements.txt
```

## Contents
Scripts in this repository are implementations based on tutorials for TensorFlow.

* [getting-started](https://github.com/AFAgarap/tensorflow-training/tree/master/getting-started) is  based on the [Getting Started](https://www.tensorflow.org/get_started/) from the official website of [TensorFlow](https://www.tensorflow.org/).
* [image-classification](https://github.com/AFAgarap/tensorflow-training/tree/master/image-classification) is based on [Siraj](https://github.com/llSourcell/)'s YouTube tutorial, ["Build a TensorFlow Image Classifier in 5 Min"](https://www.youtube.com/watch?v=QfNvhPx5Px8).
* [recurrent-neural-network](https://github.com/AFAgarap/tensorflow-training/tree/master/recurrent-neural-network) contains the implementation of [my proposed GRU+SVM model](https://www.researchgate.net/publication/317016806_A_Neural_Network_Architecture_Combining_Gated_Recurrent_Unit_GRU_and_Support_Vector_Machine_SVM_for_Intrusion_Detection), but for MNIST classification, and uses `tf.nn.static_rnn()` instead of `tf.nn.dynamic_rnn()`.
* [time-series-prediction](https://github.com/AFAgarap/tensorflow-training/tree/master/time-series-prediction) is an implementation based on the [tutorial written by Lakshmanan V](https://medium.com/google-cloud/how-to-do-time-series-prediction-using-rnns-and-tensorflow-and-cloud-ml-engine-2ad2eeb189e8).
* [unfolding-rnn](https://github.com/AFAgarap/tensorflow-training/tree/master/unfolding-rnn) is based on the tutorial written by [Suriyadeepan Ram](https://github.com/suriyadeepan), ["Vanilla, GRU, LSTM RNNs from scratch in TensorFlow"](http://suriyadeepan.github.io/2017-02-13-unfolding-rnn-2/).
