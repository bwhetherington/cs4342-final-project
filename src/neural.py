import numpy as np
from numpy.random import random
from scipy.special import softmax
import math


def relu(x):
    return np.maximum(0, x)


def relu_prime(x):
    return x > 0


def gT(dy, W2, z1):
    return np.dot(dy.T, W2) * relu_prime(z1.T)


def error(yhat, y):
    n = len(y)
    return (-1 / n) * np.sum(y * np.log(yhat))


def percent_correct(yhat, y):
    pred_indices = yhat.argmax(axis=1)
    label_indices = y.argmax(axis=1)
    correct = pred_indices == label_indices
    return correct.mean()


def split_batches(n, batch_size):
    num_batches = math.ceil(n / batch_size)
    indices = np.arange(n)
    np.random.shuffle(indices)

    index_set = []

    for i in range(num_batches):
        start = i * batch_size
        end = min(start + batch_size, n)
        section = indices[start:end]
        index_set.append(section)

    return index_set


class NeuralNetwork:
    def __init__(self, hidden_neurons):
        self.input_neurons = None
        self.hidden_neurons = hidden_neurons
        self.output_neurons = None

    # def sizes(self):
    #     W1 = self.input_neurons * self.hidden_neurons
    #     b1 = self.hidden_neurons
    #     W2 = self.hidden_neurons * self.output_neurons
    #     b2 = self.output_neurons
    #     return W1, b1, W2, b2

    def init_weights(self):
        self.W1 = 2 * (random(size=(self.hidden_neurons, self.input_neurons)) /
                       self.input_neurons**0.5) - 1.0/self.input_neurons**0.5
        self.b1 = 0.01 * np.ones((self.hidden_neurons, 1))
        self.W2 = 2 * (random(size=(self.output_neurons, self.hidden_neurons)) /
                       self.hidden_neurons**0.5) - 1.0/self.hidden_neurons**0.5
        self.b2 = 0.01 * np.ones((self.output_neurons, 1))

    def train(self, X_train, Y_train, X_test=None, Y_test=None, learning_rate=0.05, batch_size=40, num_epochs=50, learning_rate_decay=True):
        # Initialize network
        self.input_neurons = X_train.shape[1]
        self.output_neurons = Y_train.shape[1]
        self.init_weights()

        actual_lr = learning_rate

        # Train network
        for epoch in range(num_epochs):
            if learning_rate_decay:
                actual_lr = learning_rate * (1 / (math.log(epoch + 1) + 1))

            indices = split_batches(len(X_train), batch_size)
            for batch in indices:
                X_batch = X_train[batch]
                Y_batch = Y_train[batch]
                gW1, gb1, gW2, gb2 = (
                    actual_lr * x for x in self.grad(X_batch, Y_batch))
                self.W1 -= gW1
                self.b1 -= gb1
                self.W2 -= gW2
                self.b2 -= gb2

            # Assess model after each epoch
            if X_test is not None and Y_test is not None:
                label = '[{}/{}]'.format(epoch + 1, num_epochs)
                self.assess(X_test, Y_test, label=label)

    def grad(self, X, y):
        n = len(X)
        z1, h1, _, yhat = self.process(X)
        X = X.T
        y = y.T

        dy = yhat - y
        g = gT(dy, self.W2, z1).T
        gW2 = dy.dot(h1.T) / n
        gb2 = dy.mean(axis=1).reshape(-1, 1)
        gW1 = g.dot(X.T) / n
        gb1 = g.mean(axis=1).reshape(-1, 1)

        return gW1, gb1, gW2, gb2

    def error(self, X, y):
        """
        Given training images X, associated labels Y, and a vector of combined weights
        and bias terms w, compute and return the cross-entropy (CE) loss. You might
        want to extend this function to return multiple arguments (in which case you
        will also need to modify slightly the gradient check code below).
        """
        yhat = self.decision_function(X).T
        return error(yhat, y)

    def process(self, X):
        X = X.T
        z1 = self.W1.dot(X) + self.b1
        h1 = relu(z1)
        z2 = self.W2.dot(h1) + self.b2
        yhat = softmax(z2, axis=0)
        return z1, h1, z2, yhat

    def decision_function(self, X):
        _, _, _, yhat = self.process(X)
        return yhat

    def assess(self, X, y, label=''):
        yhat = self.decision_function(X).T
        err = error(yhat, y)
        acc = percent_correct(yhat, y)
        print('{} fCE={} %={}'.format(label, err, acc))
