import numpy as np

import tensorflow as tf
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import Sequential

import model_loader
from misc_utils import ModelKey

time_points = np.linspace(0, 8 * np.pi, 8000)

seq_length = 5


def gen_X_y():
    dataX = []
    dataY = []
    for i in range(0, len(time_points) - seq_length, 1):
        seq_in = np.sin(time_points)[i:i + seq_length]
        seq_out = np.cos(time_points)[i]
        dataX.append([seq_in])
        dataY.append(seq_out)

    X = np.reshape(dataX, (len(dataX), seq_length, 1))  # numpy as np
    y = np.reshape(dataY, (len(dataY), 1))
    print(f'{X} {y}')
    return X, y


def fit_normal():
    X, y = gen_X_y()
    model = Sequential()
    model.add(tf.keras.layers.Dense(units=X.shape[1], activation='relu', input_shape=(X.shape[1],)))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(units=16, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(units=49, activation='softmax'))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(units=7, activation='relu'))
    model.add(tf.keras.layers.Dense(1))
    optimizer = tf.keras.optimizers.RMSprop(0.001)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])
    model.fit(X[:6000], y[:6000], epochs=200, batch_size=10, verbose=2, validation_split=0.3)
    model_loader.save(ModelKey.ANN_NORMAL_SINE.value, model)


def fit_rnn():
    X, y = gen_X_y()
    model = Sequential()
    model.add(LSTM(16, input_shape=(X.shape[1], X.shape[2])))

    model.add(Dense(y.shape[1], activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='adam')

    model.fit(X[:6000], y[:6000], epochs=200, batch_size=10, verbose=2, validation_split=0.3)

    model_loader.save(ModelKey.RNN_SINE.value, model)


if __name__ == '__main__':
    # fit_rnn()
    fit_normal()
