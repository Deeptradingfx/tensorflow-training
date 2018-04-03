from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__version__ = '0.1.0'
__author__ = 'Abien Fred Agarap'

from keras.utils import to_categorical
import numpy as np
import os
from PIL import Image
from sklearn.preprocessing import LabelEncoder


def load_dataset(dataset_path, one_hot=False):

    files_list = []

    for subdirectories, directories, files in os.walk(dataset_path):
        for file in files:
            if file.endswith('.jpg') or file.endswith('.jpeg'):
                files_list.append(os.path.join(subdirectories, file))

    images = np.array([[np.asarray(Image.open(file).resize((32, 32))), os.path.dirname(file).strip('./')] for file in files_list])

    features = images[:, 0]
    features = np.array([feature for feature in features])

    labels = images[:, 1]
    labels = LabelEncoder().fit_transform(labels)

    del images

    if one_hot:
        labels = to_categorical(labels)

    return features, labels

def load_image(image_path):
    image = Image.open(image_path)
    image = image.resize((32, 32))
    image = np.asarray(image)
    image = np.reshape(image, (-1, 32, 32, 3))
    return image
