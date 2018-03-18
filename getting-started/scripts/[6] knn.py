# MIT License
# 
# Copyright (c) 2018 Abien Fred Agarap
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

import math
import numpy as np
import operator
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def euclidean_distance(first_instance, second_instance, length):
    distance = 0
    for index in range(length):
        distance += pow((first_instance[index] - second_instance[index]), 2)
    return math.sqrt(distance)

def get_neighbors(training_data, test_data, k):
    distances = []
    length = len(test_data) - 1
    for index in range(len(training_data)):
        distance = euclidean_distance(test_data, training_data[index], length)
        distances.append((training_data[index], distance))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for index in range(k):
        neighbors.append(distances[index][0])
    return neighbors

def get_response(neighbors):
    class_votes = {}
    for index in range(len(neighbors)):
        response = neighbors[index][-1]
        if response in class_votes:
            class_votes[response] += 1
        else:
            class_votes[response] = 1
    sorted_votes = sorted(class_votes.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_votes[0][0]

def get_accuracy(test_data, predictions):
    correct = 0
    for index in range(len(test_data)):
        if test_data[index][-1] == predictions[index]:
            correct += 1
    return (float(correct) / float(len(test_data))) * 100.00

def main():
    features = datasets.load_iris().data
    labels = datasets.load_iris().target

#    features = StandardScaler().fit_transform(features)

    train_features, test_features, train_labels, test_labels = train_test_split(features,
                                                                                labels,
                                                                                test_size=0.3,
                                                                                stratify=labels)
    scaler = StandardScaler()
    scaler.fit(train_features)
    train_features = scaler.transform(train_features)
    test_features = scaler.transform(test_features)

    train_data = np.c_[train_features, train_labels]
    test_data = np.c_[test_features, test_labels]
    predictions = []
    k = 3
    for index in range(len(test_data)):
        neighbors = get_neighbors(train_data, test_data[index], k)
        result = get_response(neighbors)
        predictions.append(result)
        print('predicted={}, actual={}'.format(result, test_data[index][-1]))
    accuracy = get_accuracy(test_data, predictions)
    print('accuracy : {}'.format(accuracy))

if __name__ == '__main__':
    main()
