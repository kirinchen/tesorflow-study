from random import random, randint

import tensorflow as tf
from keras import Model

import misc_utils
from ann_model_creator import AnnModelCreator


def setup_model(model: Model):
    model.add(tf.keras.layers.Dense(units=6, activation='relu', input_shape=(6,)))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(units=36, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(units=128, activation='softmax'))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(units=64, activation='relu'))
    model.add(tf.keras.layers.Dense(1))
    optimizer = tf.keras.optimizers.RMSprop(0.001)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])


def gen_mock_x_args(x_list: list, y_list: list) -> list:
    cur_len = len(x_list)
    mod_num = cur_len % 10
    x0 = random()
    x1 = random()
    x2 = random()
    x3 = random()
    x4 = randint(0, 3)
    x5 = randint(0, 3)
    return [x0, x1, x2, x3, x4, x5]


def _gen_func_float(x: list) -> float:
    idx_a = x[4]
    idx_b = x[5]
    ans = 0
    ans += x[idx_a] * 2
    ans += x[idx_b] * 3
    return ans


if __name__ == '__main__':
    ann_model_creator = AnnModelCreator(
        y_val_func=_gen_func_float,
        model_consumer=setup_model,
        gen_mock_x_args_func=gen_mock_x_args
    )
    ann_model_creator.save(misc_utils.ModelKey.CUSTOM_IDX)
