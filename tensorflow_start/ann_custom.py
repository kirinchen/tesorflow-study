import tensorflow as tf
from keras import Model

import misc_utils
import model_loader
from ann_model_creator import AnnModelCreator


def setup_model(model: Model):
    model.add(tf.keras.layers.Dense(units=3, activation='relu', input_shape=(3,)))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(units=7, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(units=10, activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])


ann_model_creator = AnnModelCreator(y_val_func=misc_utils._gen_func, model_consumer=setup_model)
ann_model_creator.save(misc_utils.ModelKey.CUSTOM)

# print(_gen_func([0.2, 0.8, 0.7]))

#
# (X_train, y_train), (X_test, y_test) = (_gen_x_y(pow(10, 5)), _gen_x_y(pow(10, 3)))
#
# print(X_train.shape)
# print(X_test.shape)
#
# model = tf.keras.models.Sequential()
# model.add(tf.keras.layers.Dense(units=3, activation='relu', input_shape=(3,)))
# model.add(tf.keras.layers.Dropout(0.3))
# model.add(tf.keras.layers.Dense(units=7, activation='relu'))
# model.add(tf.keras.layers.Dropout(0.3))
# model.add(tf.keras.layers.Dense(units=10, activation='softmax'))
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
# model.summary()
#
# model.fit(X_train, y_train, epochs=10)
# test_loss, test_accuracy = model.evaluate(X_test, y_test)
#
# print("Test accuracy: {}".format(test_accuracy))
#
# print('-----SAVE & Reload model')
#
# model_loader.save("custom", model)
