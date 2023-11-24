from random import random
from typing import Callable

import numpy as np
import tensorflow as tf
from keras import Model

import model_loader
from misc_utils import ModelKey


class AnnModelCreator:

    def __init__(self, y_val_func: Callable[[list], any], model_consumer: Callable[[Model], any]):
        print(y_val_func([0.2, 0.8, 0.7]))
        self.y_val_func: Callable[[list], any] = y_val_func
        self.model_consumer: Callable[[Model], any] = model_consumer
        self.X_train, self.y_train = self.gen_x_y(pow(10, 5))
        self.X_test, self.y_test = self.gen_x_y(pow(10, 3))
        model = tf.keras.models.Sequential()
        self.model_consumer(model)
        model.summary()
        model.fit(self.X_train, self.y_train, epochs=10)
        self.evaluate_info = model.evaluate(self.X_test, self.y_test)
        print("evaluate_info>>>>>>>")
        print(self.evaluate_info)
        print("<<<<<<<<<evaluate_info")

        self.model = model

    def save(self, key: ModelKey):
        model_loader.save(key.value, self.model)

    def gen_x_y(self, count: int):
        x_list = []
        y_list = []
        while len(x_list) < count:
            x = [random(), random(), random()]
            x_list.append(x)
            y = self.y_val_func(x)
            y_list.append(y)

        return np.array(x_list), np.array(y_list)
