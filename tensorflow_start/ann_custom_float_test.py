import numpy as np

import ann_custom_float
import misc_utils
import model_loader

model = model_loader.load_by_key(misc_utils.ModelKey.CUSTOM_FLOAT)
test_args = [0.8, 0.8, 0.9]
predication = model.predict([test_args])
print(predication)
p_idx = np.argmax(predication[0])
print(p_idx)
ans = ann_custom_float.gen_func_float(test_args)
print(ans)
