import numpy as np

import ann_custom_index
import misc_utils
import model_loader

if __name__ == '__main__':
    model = model_loader.load_by_key(misc_utils.ModelKey.CUSTOM_IDX)
    test_args = [0.5, 0.9, 0.8, 0.4, 0, 2]
    predication = model.predict([test_args])
    print(f'predication={predication}')
    p_idx = np.argmax(predication[0])
    print(f'p_idx={p_idx}')
    ans = ann_custom_index._gen_func_float(test_args)
    print(f'ans={ans}')
