from random import random
import numpy as np
import tensorflow as tf

import model_loader
from misc_utils import _gen_func

print(_gen_func([0.2, 0.8, 0.7]))


def _gen_x_y(count: int):
    x_list = []
    y_list = []
    while len(x_list) < count:
        x = [random(), random(), random()]
        x_list.append(x)
        y = _gen_func(x)
        y_list.append(y)

    return np.array(x_list), np.array(y_list)


#
(X_train, y_train), (X_test, y_test) = (_gen_x_y(pow(10, 5)), _gen_x_y(pow(10, 3)))

#
# X_train = X_train / 255.0
# X_test = X_test / 255.0
#
# X_train = X_train.reshape(-1, 28 * 28)
# X_test = X_test.reshape(-1, 28 * 28)
print(X_train.shape)
print(X_test.shape)
#
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(units=3, activation='relu', input_shape=(3,)))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Dense(units=7, activation='relu'))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Dense(units=10, activation='softmax'))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
model.summary()
model.fit(X_train, y_train, epochs=10)
test_loss, test_accuracy = model.evaluate(X_test, y_test)

print("Test accuracy: {}".format(test_accuracy))

print('-----SAVE & Reload model')

model_loader.save("custom", model)
#
# test_loss, test_accuracy = model.evaluate(X_test, y_test)
#
# print("Test accuracy: {}".format(test_accuracy))
#
# print('-----SAVE & Reload model')
#
# model_json = model.to_json()
# with open("fashion_model.json", "w") as json_file:
#     json_file.write(model_json)
#
# model.save_weights("fashion_model.h5")
#
# model_new: Model = tf.keras.models.model_from_json(json_string=model_json)
# model_new.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
# model_new.summary()
# model_new.load_weights('fashion_model.h5')
# test_loss, test_accuracy = model_new.evaluate(X_test, y_test)
#
# print("Test accuracy: {}".format(test_accuracy))
