import numpy as np

import misc_utils
import model_loader

model = model_loader.load_by_key(misc_utils.ModelKey.CUSTOM_FLOAT)
test_args = [0.8, 0.8, 0.8]
predication = model.predict([test_args])
print(predication)
p_idx = np.argmax(predication[0])
print(p_idx)
ans = misc_utils._gen_func_float(test_args)
print(ans)
