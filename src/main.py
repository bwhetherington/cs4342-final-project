import pandas as pd
import numpy as np
import neural
import sys

import keras
from keras import layers


NUM_HANDS = 10


def prepare_y(hands):
    n = len(hands)
    y = np.zeros((n, NUM_HANDS))
    for i, hand in zip(np.arange(n), hands):
        y[i][hand] = 1
    return y
    # return hands


card_size = 4 + 13


def set_one_card(out, index, data):
    suit, value = data
    start = card_size * index
    end = start + card_size
    out[start:start + 4] = suit
    out[start + 4:end] = value


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


def load_data(file):
    d = pd.read_csv(file)
    # print(d)
    y = prepare_y(np.array(d.hand))        # Labels
    X = one_hot(np.array(d.iloc[:, :10]))  # Features
    return X, y


def load_test(file):
    d = pd.read_csv(file)
    X = np.array(d.iloc[:, 1:])
    return one_hot(X)


def create_submission(file, yhat):
    yhat = yhat.argmax(axis=1)

    pass


def write_csv(file, df):
    with open(file, 'w') as f:
        f.write(df.to_csv())


def split_data(X, Y):
    n = len(X)
    mid = int(n / 2)
    X_train = X[:mid]
    Y_train = Y[:mid]
    X_test = X[mid:]
    Y_test = Y[mid:]
    return X_train, Y_train, X_test, Y_test


def augment(X, Y):
    resamples = 10
    n, f = X.shape

    # We will include 3 shuffled versions of each hand for each hand in the input data
    X_aug = np.zeros((n * resamples, f))
    Y_aug = np.zeros((n * resamples, 10))

    for i, x, y in zip(np.arange(n), X, Y):
        start = i * resamples
        Y_aug[start:start + resamples] = y
        for j in range(resamples):
            x_aug = np.copy(x).reshape((5, 17))
            np.random.shuffle(x_aug)
            x_aug = x_aug.flatten()
            X_aug[start + j] = x_aug

    return X_aug, Y_aug


def main():
    X, Y = load_data('train.csv')
    X, Y = augment(X, Y)
    X_train, Y_train, X_test, Y_test = split_data(X, Y)

    model = keras.Sequential([
        layers.Dense(70, name='hidden1', activation='relu', input_shape=(85,)),
        layers.Dense(60, name='hidden2', activation='relu'),
        layers.Dense(30, name='hidden3', activation='relu'),
        layers.Dense(10, name='output', activation='softmax')
    ], 'poker_predictor')

    model.summary()

    model.compile(
        optimizer=keras.optimizers.RMSprop(),  # Optimizer
        # Loss function to minimize
        loss=keras.losses.CategoricalCrossentropy(),
        # List of metrics to monitor
        metrics=["categorical_accuracy"],
    )

    model.fit(
        x=X_train,
        y=Y_train,
        batch_size=40,
        epochs=50,
        validation_data=(X_test, Y_test)
    )
    model.save_weights('model.h5')

    # model.load_weights('model.h5')

    # nn = neural.NeuralNetwork(hidden_neurons=70)
    # nn.train(X_train, Y_train, X_test=X_test,
    #          Y_test=Y_test, batch_size=40, learning_rate=0.1, num_epochs=75, learning_rate_decay=False)

    # X_submit = load_test('test.csv')
    # np.save("X_submit.npy", X_submit)

    X_submit = np.load("X_submit.npy")

    predictions = np.array(model.predict(X_submit)).argmax(axis=1)
    indices = np.arange(len(predictions)) + 1
    data = np.vstack((indices, predictions)).T
    d = pd.DataFrame(data=data, columns=["id", "hand"])
    d.to_csv('submission.csv', mode='w', index=False)


if __name__ == '__main__':
    main()
