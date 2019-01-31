#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 12:24:56 2019

@author: adhamlin
"""
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD
import numpy as np
# reproducability
np.random.seed(444)

# 4 possible A-B sets array
X = np.array([[0, 0],
             [0, 1],
             [1, 0],
             [1, 1]])
y = np.array([[0], [1], [1], [0]])

# define the neural network
# provided model by keras
model = Sequential()
"""
First layer of neurons, composed of two neurons, fed by two inputs.
"""
model.add(Dense(2, input_dim=2))
"""
Define their activation function as a sigmoid function in the sequence.
"""
model.add(Activation('sigmoid'))

"""
Define the output layer which is composed of one neuron
with the same sigmoid activation function
"""
model.add(Dense(1))
model.add(Activation('sigmoid'))

# Stochastic Gradient Descent (sgd)
sgd = SGD(lr=0.1)  # learning rate equal to 0.1
# minimized loss function
model.compile(loss='mean_squared_error', optimizer=sgd)


"""
Using X and y as training examples.

(batch_size=1):
After every training example is fed into the network the weights are updated.

The number of epochs equals the number of times the whole training set
will be used to train the neural network.
"""
# run training by the fit() method
model.fit(X, y, batch_size=1, epochs=5000)

# print the predicted values for the 4 possible input examples
# if __name__ == '__main__':
print("Results expected are: 0, 1, 1, 0")
print(model.predict(X))
to_round = model.predict(X)
print("Rounding results to obtain binary values: ")
print(np.around(to_round))
