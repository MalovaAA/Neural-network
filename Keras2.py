#!/usr/bin/env python
# coding: utf-8

from keras.datasets import mnist
from keras.utils import np_utils
from keras import optimizers
from keras.models import Model
from keras.layers import Input, Dense
from datetime import datetime
import sys 

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

def NNKeras(hidden_size=30, batch_size=128, num_epochs=20, lr=0.1):
    (X_train, Y_train), (X_test, Y_test) = get_data_from_Keras()
    X_train, X_test = normalize(X_train, X_test)

    time_start = datetime.now()
    inp = Input(shape=(X_train.shape[1],))  # Our input is a 1D vector of size 784
    hidden_1 = Dense(hidden_size, activation='relu')(inp)  # First hidden ReLU layer
    out = Dense(Y_train.shape[1], activation='softmax')(hidden_1)  # Output softmax layer
    model = Model(input=inp, output=out)  # To define a model, just specify its input and output layers
    sgd = optimizers.SGD(lr=lr, momentum=0.0, nesterov=False)

    model.compile(loss='categorical_crossentropy',  # using the cross-entropy loss function
                  optimizer=sgd,  # using the SGD optimiser
                  metrics=['accuracy'])  # reporting the accuracy

    model.fit(X_train, Y_train,  # Train the model using the training set...
              batch_size=batch_size, nb_epoch=num_epochs,
              verbose=2)

    delta_time = datetime.now() - time_start
    score_train = model.evaluate(X_train, Y_train, verbose=0)
    score_test = model.evaluate(X_test, Y_test, verbose=0)
    return score_train, score_test, delta_time

if __name__ == '__main__':
    if  len(sys.argv) > 0:
        score_train, score_test, delta_time = NNKeras(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]))
    else:
        score_train, score_test, delta_time = NNKeras()
    print('Delta time =', delta_time)
    print('Test loss:', score_test[0])
    print('Test accuracy:', score_test[1])




