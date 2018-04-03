from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__version__ = '0.1.0'
__author__ = 'Abien Fred Agarap'

import argparse
from cnn_keras import CNN
from sklearn.model_selection import train_test_split
from utils import load_dataset


def parse_args():
    parser = argparse.ArgumentParser(description='Training module for VGG-like CNN')
    group  = parser.add_argument_group('Arguments')
    group.add_argument('-d', '--dataset', required=True, type=str,
            help='the dataset to be used for training')
    group.add_argument('-a', '--activation', required=False, type=str,
            help='the activation function to be used by conv layers, default is "relu"')
    group.add_argument('-c', '--classifier', required=False, type=str,
            help='the classification function to be used, default is "softmax"')
    group.add_argument('-l', '--loss', required=False, type=str,
            help='the loss function to be used, default is "categorical_crossentropy"')
    group.add_argument('-o', '--optimizer', required=False, type=str,
            help='the optimization algorithm to be used, default is "adam"')
    group.add_argument('-t', '--model_name', required=False, type=str,
            help='the filename for trained model, default is "model.h5"')
    arguments = parser.parse_args()
    return arguments
    

def main(argv):
    
    features, labels = load_dataset(argv.dataset, one_hot=True)
    
    activation = argv.activation if argv.activation is not None else 'relu'
    classifier = argv.classifier if argv.classifier is not None else 'softmax'
    loss = argv.loss if argv.loss is not None else 'categorical_crossentropy'
    optimizer = argv.optimizer if argv.optimizer is not None else 'adam'

    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.30, stratify=labels)

    model = CNN(activation=activation, classifier=classifier, input_shape=features.shape[1:], loss=loss,
                num_classes=labels.shape[1], optimizer=optimizer, return_summary=True)
    model.train(batch_size=256, n_splits=10, epochs=32, validation_split=0., verbose=0, train_features=x_train, train_labels=y_train)
    report, conf_matrix = model.evaluate(batch_size=128, class_names=['Janella', 'Liza', 'Yen'], test_features=x_test, test_labels=y_test)
    print(report)
    print('=====')
    print(conf_matrix)
    
    model_name = argv.model_name if argv.model_name is not None else 'model.h5'
    model.save_model(model_name)


if __name__ == '__main__':
    args = parse_args()

    main(args)
