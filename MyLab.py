#!/usr/bin/env python
# coding: utf-8


from keras.datasets import mnist
from keras.utils import np_utils
import matplotlib.pyplot as plt
import matplotlib.image as mping
import numpy as np
import sys 
from   random   import *
from   datetime import datetime


def get_data_from_Keras():
    numbers = 10
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1]*X_train.shape[2])
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1]*X_test.shape[2])
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    Y_train = np_utils.to_categorical(Y_train, numbers)
    Y_test = np_utils.to_categorical(Y_test, numbers)
    return (X_train, Y_train), (X_test, Y_test)


def normalize(X_train, X_test):
    X_train /= 255
    X_test /= 255
    return X_train, X_test



class neural_network:
    def __init__(self, inputs = 784, hiddens=30, outputs=10, lr=0.1):
        self.inputs_count  = inputs
        self.hiddens_count = hiddens
        self.outputs_count = outputs
        self.lr = lr
        self.W1 = np.random.normal(0.0, pow(self.hiddens_count, -0.5), (self.hiddens_count, self.inputs_count))
        self.W2 = np.random.normal(0.0, pow(self.outputs_count, -0.5), (self.outputs_count, self.hiddens_count))
        self.b1 = np.zeros((self.hiddens_count, 1))
        self.b2 = np.zeros((self.outputs_count, 1))
        self.batch_size = 0      
        self.n = 0

    def forward(self, X):
        res = {}

        X = X.T
        WX = self.W1.dot(X) + self.b1
        X = self.ReLU(WX)
        res[0] = X
        res[1] = WX

        WX = self.W2.dot(X) + self.b2
        X = self.softmax(WX)
        res[2] = X

        return X, res
    def predict(self, X, Y):
        U, coef = self.forward(X)
        crossentropy = -np.sum(Y * np.log(U.T)) / X.shape[0]

        U = np.argmax(U, axis=0)
        Y = np.argmax(Y, axis=1)
        accuracy = (U == Y).mean()

        return crossentropy, accuracy

    def fit(self, X, Y, batch_size=128, number_epochs=1000):
        np.random.seed(1)
        self.batch_size = batch_size
        self.n = X.shape[0]

        epoch_percent = 0
        for epoch in range(number_epochs):
            X, Y = self.unisonShuffle(X, Y)
            for i in range(0, self.n, self.batch_size):
                U, coef = self.forward(X[i:i + self.batch_size])
                dW1, dW2, db1, db2 = self.calDerivatives(X[i:i + self.batch_size], Y[i:i + self.batch_size], coef)
                self.W1 = self.W1 + self.lr * dW1
                self.W2 = self.W2 + self.lr * dW2
                self.b1 = self.b1 + self.lr * db1
                self.b2 = self.b2 + self.lr * db2
                
            crossentropy, accuracy = self.predict(X, Y)
            print()
            print('Epoch = ', epoch)
            print('train_accurancy = ', accuracy)
            
    def calDerivatives(self, X, Y, coef):
        delta_2 = Y.T - coef[2] 
        dW2 = delta_2.dot(coef[0].T) / self.batch_size
        db2 = np.sum(delta_2, axis=1, keepdims=True) / self.batch_size

        delta_1 = self.W2.T.dot(delta_2) * self.ReLUDerivative(coef[1])
        dW1 = delta_1.dot(X) / self.batch_size
        db1 = np.sum(delta_1, axis=1, keepdims=True) / self.batch_size

        return dW1, dW2, db1, db2

    @staticmethod
    def ReLU(X):
        return X * (X > 0)

    @staticmethod
    def ReLUDerivative(X):
        return 1. * (X > 0)

    @staticmethod
    def softmax(X):
        expX = np.exp(X)
        return expX / expX.sum(axis=0, keepdims=True)

    @staticmethod
    def unisonShuffle(a, b):
        random_state = np.random.get_state()
        np.random.shuffle(a)
        np.random.set_state(random_state)
        np.random.shuffle(b)
        return a, b

def run(hidden_size=30, batch_size=128, num_epochs=20, lr=0.1):
    (train_x, train_y), (test_x, test_y) = get_data_from_Keras()
    train_x, test_x = normalize(train_x, test_x)
    
    time_start = datetime.now()
    net = neural_network(hiddens = hidden_size, lr=lr)
    net.fit(train_x, train_y, batch_size=batch_size, number_epochs=num_epochs)

    time   = datetime.now() - time_start
    test   = net.predict(test_x, test_y)
    return time, test


if __name__ == '__main__':
    if  len(sys.argv) > 0:
        time, test = run(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]))
    else:
        time, test = run()
    print() 
    print('time      =', time)
    print('accuracy  =', test[1])
    #print('test loss =', test[0])
    print()
