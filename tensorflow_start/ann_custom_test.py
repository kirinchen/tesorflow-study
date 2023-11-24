import numpy as np

import misc_utils
import model_loader
from misc_utils import _gen_func

model = model_loader.load("custom")
test_args = [0.2, 0.8, 0.7]
predication = model.predict([test_args])
print(predication)
p_idx = np.argmax(predication[0])
print(p_idx)
ans = _gen_func(test_args)
print(ans)
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
