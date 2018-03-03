#!/usr/bin/python3

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

"""
SGD implementation for 2-layer neural network with sigmoid function

You may refer to the accompanying documentation on the SGD algorithm: `sgd-algorithm.md`.
"""
import numpy as np
from numpy.random import randn

N, D_in, H, D_out = 64, 1000, 100, 10
x, y = randn(N, D_in), randn(N, D_out)
w1, w2 = randn(D_in, H), randn(H, D_out)

for t in range(5000):                   
	h = 1 / (1 + np.exp(-x.dot(w1)))
	y_pred = h.dot(w2)
	y_pred[y_pred <= 0] = 0
	loss = np.square(y_pred - y).sum()
	print('step {}, loss : {}'.format(t, loss))

	grad_y_pred = 2.0 * (y_pred - y)
	grad_w2 = h.T.dot(grad_y_pred)
	grad_h = grad_y_pred.dot(w2.T)
	grad_w1 = x.T.dot(grad_h * h * (1 - h))

	w1 -= 1e-4 * grad_w1
	w2 -= 1e-4 * grad_w2
