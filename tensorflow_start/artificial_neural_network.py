import numpy as np
import datetime
import tensorflow as tf
from keras.datasets import fashion_mnist
from keras.engine.training_v1 import Model


(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

X_train = X_train / 255.0
X_test = X_test / 255.0

X_train = X_train.reshape(-1, 28 * 28)
X_test = X_test.reshape(-1, 28 * 28)
print(X_train.shape)
print(X_test.shape)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(units=128, activation='relu', input_shape=(784,)))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Dense(units=10, activation='softmax'))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
model.summary()
model.fit(X_train, y_train, epochs=5)

test_loss, test_accuracy = model.evaluate(X_test, y_test)

print("Test accuracy: {}".format(test_accuracy))

print('-----SAVE & Reload model')

model_json = model.to_json()
with open("fashion_model.json", "w") as json_file:
    json_file.write(model_json)

model.save_weights("fashion_model.h5")

model_new: Model = tf.keras.models.model_from_json(json_string=model_json)
model_new.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
model_new.summary()
model_new.load_weights('fashion_model.h5')
test_loss, test_accuracy = model_new.evaluate(X_test, y_test)

print("Test accuracy: {}".format(test_accuracy))
