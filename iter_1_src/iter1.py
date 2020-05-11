import numpy as np
import datetime
import keras
from keras import layers
import pandas as pd

NUM_HANDS = 10

def shuffle_pairwise(a, b):
    '''Returns a pair of shuffled copies of the given arrays, shuffled in the same way.'''
    p = np.random.permutation(len(a))
    return a[p], b[p]

def prepare_y(hands):
    n = len(hands)
    y = np.zeros((n, NUM_HANDS))
    for i, hand in zip(np.arange(n), hands):
        y[i][hand] = 1
    return y
    # return hands 

def split_data(X, Y):
    n = len(X)
    mid = int(n / 2)
    X_train = X[:mid]
    Y_train = Y[:mid]
    X_test = X[mid:]
    Y_test = Y[mid:]
    return X_train, Y_train, X_test, Y_test

def load_data(file):
    d = pd.read_csv(file)
    # print(d)
    y = prepare_y(np.array(d.hand))        # Labels
    X = one_hot(np.array(d.iloc[:, :10]))  # Features
    return X, y

def get_predictions(W, images):
    # images: _x785
    # W: 785x10
    # intermediate layer for softmax corresponding to exp(Z)
    expZ = np.exp(images.dot(W)) # dimension: _x10
    return expZ / np.sum(expZ, axis=1).reshape(-1,1)

def get_gradient_CE(images, predictions, labels):
    # images: nx785
    # predictions: nx10
    # labels: nx10
    n = len(images)
    return images.T.dot(predictions - labels) / n # dimension: 785x10

def train_epoch_SGD(W, trainingImages, trainingLabels, epsilon, batchSize):
    '''
    Trains the weights over one epoch.
    '''
    for i in range(0, 5000, batchSize):
        images = trainingImages[i:i+batchSize]
        labels = trainingLabels[i:i+batchSize]
        predictions = get_predictions(W, images)
        W -= get_gradient_CE(images, predictions, labels) * epsilon
    return W


# Given training and testing data, learning rate epsilon, and a specified batch size,
# conduct stochastic gradient descent (SGD) to optimize the weight matrix W (785x10).
# Then return W.
def softmaxRegression (trainingImages, trainingLabels, testingImages, testingLabels, epsilon = None, batchSize = None, num_epochs=1):
    # trainingImages: 5000x785
    # trainingLabels: 5000x10
    W = np.random.normal(size=(trainingImages.shape[1], trainingLabels.shape[1])) # initialize weights to random numbers

    for e in range(num_epochs):
        W = train_epoch_SGD(W, trainingImages, trainingLabels, epsilon or .1, batchSize or 100)
        if (num_epochs - e) <= 20:
            print(f"{num_epochs-e}: fPC = {fPC(W, testingImages, testingLabels)}")
    return W

def appendOnes (images):
    return np.vstack((images.T, np.ones(images.shape[0]))).T

def fPC(W, images, labels):
    '''Percent correct.'''
    predictions = get_predictions(W, images)
    return np.mean(np.argmax(predictions, axis=1) == np.argmax(labels, axis=1))

def fCE(W, images, labels):
    '''Cross-entropy loss.'''
    n = len(images)
    predictions = get_predictions(W, images)
    return -np.sum(labels * np.log(predictions)) / n

card_size = 4 + 13

def load_test(file):
    d = pd.read_csv(file)
    X = np.array(d.iloc[:, 1:])
    return one_hot(X)

def one_hot(X):
    """
    One hot encode the input data
    """
    # Reshape Y
    n = len(X)
    hands = X.reshape((n, 5, 2))

    out = np.zeros((n, 5 * (4 + 13)))

    for i, hand in zip(range(len(hands)), hands):
        for j, card in zip(range(len(hand)), hand):
            suit_slots = np.zeros(4)
            value_slots = np.zeros(13)

            suit, value = card - 1

            suit_slots[suit] = 1
            value_slots[value] = 1

            set_one_card(out[i], j, (suit_slots, value_slots))

    return out

def set_one_card(out, index, data):
    suit, value = data
    start = card_size * index
    end = start + card_size
    out[start:start + 4] = suit
    out[start + 4:end] = value

if __name__ == "__main__":
    # Load data
    X, Y = load_data('train.csv')
    trainingImages, trainingLabels, testingImages, testingLabels = split_data(X,Y)
    print(Y.shape)
    # shuffle data
    trainingImages, trainingLabels = shuffle_pairwise(trainingImages, trainingLabels)

    # Append a constant 1 term to each example to correspond to the bias terms
    trainingImages = appendOnes(trainingImages)
    testingImages = appendOnes(testingImages)

    # do regression (time it)
    start = datetime.datetime.now()
    W = softmaxRegression(trainingImages, trainingLabels, testingImages, testingLabels, epsilon=0.1, batchSize=100, num_epochs=1024)
    stop = datetime.datetime.now()

    T = load_test('test.csv')
    T = appendOnes(T)
    predictions = get_predictions(W, T)
    np.savetxt("resp1.csv", predictions, fmt = '%d', header = 'id,hand', comments = '', delimiter=",")
    print('Time to train: %.2f seconds' % (stop - start).total_seconds())

    # print fCE and fPC
    print(f"Test accuracy (PC): {fPC(W, testingImages, testingLabels)}")
    print(f"Test loss (CE): {fCE(W, testingImages, testingLabels)}")
