import math

import numpy
import numpy as np

import ann_custom_sine
import misc_utils
import model_loader
import plt_utils
import rnn_sine

X, y = rnn_sine.gen_X_y()

model = model_loader.load_by_key(misc_utils.ModelKey.RNN_SINE)
predict_y = model.predict(X)
print(f'predict_y={predict_y}')
plt_utils.show_array(y, [plt_utils.SerMeta(val_array=predict_y, color='blue', label='predict')])
# plt_utils.show_array(predict_y)
