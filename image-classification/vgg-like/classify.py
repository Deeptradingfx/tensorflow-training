from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__version__ = '0.1.0'
__author__ = 'Abien Fred Agarap'

import argparse
from keras.models import load_model
from utils import load_image

def parse_args():
    parser = argparse.ArgumentParser(description='VGG-like CNN Classifier program')
    group = parser.add_argument_group('Arguments')
    group.add_argument('-i', '--image', required=True, type=str,
            help='the image to classify')
    group.add_argument('-m', '--model_name', required=True, type=str,
            help='the trained model to use')
    arguments = parser.parse_args()
    return arguments

def main(argv):
    
    image = load_image(argv.image)

    model = load_model(argv.model_name)
    probabilities = model.predict_proba(image)
    print(probabilities)

if __name__ == '__main__':
    args = parse_args()

    main(args)
