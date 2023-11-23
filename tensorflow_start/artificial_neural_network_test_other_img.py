import numpy as np
import tensorflow as tf
from keras.engine.training_v1 import Model
from matplotlib import image


def _row_val(row):
    _row = row.astype('float32')
    avgv = ((_row[0] + _row[1] + _row[2]) / 3)/255
    return avgv


def convert_img_array(img_ary):
    pre_list = []
    for col in img_ary:
        pre_list.append([_row_val(row) for row in col])
    new_image = np.array(pre_list)
    return [new_image.reshape(1, 28 * 28)]


with open("fashion_model.json", "r") as json_file:
    model_json = json_file.read()
model: Model = tf.keras.models.model_from_json(json_string=model_json)
model.load_weights('fashion_model.h5')

image_ary = image.imread('boots.jpg')
# summarize shape of the pixel array
print(image_ary.dtype)
print(image_ary.shape)

labels = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]
wb_img_ary = convert_img_array(image_ary)
predication = model.predict(wb_img_ary)
print(predication)
p_idx= np.argmax(predication[0])
print(p_idx)
print(labels[p_idx])
