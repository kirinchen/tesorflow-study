from random import random, randint

import tensorflow as tf
from keras import Model

import misc_utils
from ann_model_creator import AnnModelCreator


#
# print(_gen_func_float([0.2, 0.8, 0.7]))
#
#
# def _gen_x_y(count: int):
#     x_list = []
#     y_list = []
#     while len(x_list) < count:
#         x = [random(), random(), random()]
#         x_list.append(x)
#         y = _gen_func_float(x)
#         y_list.append(y)
#
#     return np.array(x_list), np.array(y_list)
#
#
# #
# (X_train, y_train), (X_test, y_test) = (_gen_x_y(pow(10, 5)), _gen_x_y(pow(10, 3)))
#
# #
# # X_train = X_train / 255.0
# # X_test = X_test / 255.0
# #
# # X_train = X_train.reshape(-1, 28 * 28)
# # X_test = X_test.reshape(-1, 28 * 28)
# print(X_train.shape)
# print(X_test.shape)
# #
# model = tf.keras.models.Sequential()
# model.add(tf.keras.layers.Dense(units=3, activation='relu', input_shape=(3,)))
# model.add(tf.keras.layers.Dropout(0.3))
# model.add(tf.keras.layers.Dense(units=7, activation='relu'))
# model.add(tf.keras.layers.Dense(1))
# optimizer = tf.keras.optimizers.RMSprop(0.001)
# model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])
# model.summary()
# model.fit(X_train, y_train, epochs=10)
# test_accuracy = model.evaluate(X_test, y_test)
#
# print("Test accuracy: {}".format(test_accuracy))
#
# print('-----SAVE & Reload model')
#
# model_loader.save("custom", model)


def setup_model(model: Model):
    model.add(tf.keras.layers.Dense(units=3, activation='relu', input_shape=(3,)))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(units=7, activation='relu'))
    model.add(tf.keras.layers.Dense(units=28, activation='sigmoid'))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(units=32, activation='softmax'))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(units=7, activation='relu'))
    # model.add(tf.keras.layers.Dense(units=49, activation='linear'))
    model.add(tf.keras.layers.Dense(1))
    optimizer = tf.keras.optimizers.RMSprop(0.001)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])


def gen_mock_x_args(x_list: list, y_list: list) -> list:
    cur_len = len(x_list)
    mod_num = cur_len % 10
    x1 = random()
    x2 = random()
    x3 = randint(0, 1)
    return [x1, x2, x3]


def gen_func_float(x: list) -> float:
    ans = 0
    ans += x[0] * 2
    ans += x[1] * 3
    ans = ans if x[2] == 0 else ans + 100

    return ans


if __name__ == '__main__':
    ann_model_creator = AnnModelCreator(
        y_val_func=gen_func_float,
        model_consumer=setup_model,
        gen_mock_x_args_func=gen_mock_x_args,
        train_count=99999,
        test_count=11111,
        epochs=100

    )
    ann_model_creator.save(misc_utils.ModelKey.CUSTOM_FLOAT)
