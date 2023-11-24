from enum import Enum
import tensorflow as tf

from keras.engine.training_v1 import Model

SAVE_PATH_PREFIX = "model/"


class ModelExtension(Enum):
    JSON = 'json'
    H5 = 'h5'


def get_path(key: str, extension: ModelExtension) -> str:
    return SAVE_PATH_PREFIX + key + "." + extension.value


def save(file_key: str, model: Model):
    model_json = model.to_json()
    with open(get_path(file_key, ModelExtension.JSON), "w") as json_file:
        json_file.write(model_json)
    model.save_weights(get_path(file_key, ModelExtension.H5))


def load(file_key: str) -> Model:
    with open(get_path(file_key, ModelExtension.JSON), "r") as json_file:
        model_json = json_file.read()
    model: Model = tf.keras.models.model_from_json(json_string=model_json)
    model.load_weights(get_path(file_key, ModelExtension.H5))
    return model
