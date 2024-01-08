import math
from random import random

import numpy
import tensorflow as tf
from keras import Model

import misc_utils
from ann_model_creator import AnnModelCreator


def setup_model(model: Model):
    model.add(tf.keras.layers.Dense(units=1, activation='relu', input_shape=(1,)))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(units=16, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(units=49, activation='softmax'))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(units=7, activation='relu'))
    model.add(tf.keras.layers.Dense(1))
    optimizer = tf.keras.optimizers.RMSprop(0.001)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])


def gen_mock_x_args(x_list: list, y_list: list) -> list:
    cur_len = len(x_list)
    deg = float(cur_len) * 0.1
    sin_y = numpy.sin(math.radians(deg))
    return [sin_y]


def val_func(x: list) -> float:
    deg = math.asin(x[0])
    shift_deg = math.degrees(deg) + 90
    sin_y = numpy.sin(math.radians(shift_deg))
    return sin_y


if __name__ == '__main__':
    ann_model_creator = AnnModelCreator(
        y_val_func=val_func,
        model_consumer=setup_model,
        gen_mock_x_args_func=gen_mock_x_args
    )
    ann_model_creator.save(misc_utils.ModelKey.CUSTOM_SINE)
