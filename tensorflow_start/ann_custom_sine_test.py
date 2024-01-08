import math

import numpy
import numpy as np

import ann_custom_sine
import misc_utils
import model_loader


def show_rad_deg(prefix: str, y: float):
    print(f'{prefix}={y}')
    _rad = math.asin(y)
    print(f'{prefix}_rad={_rad}')
    _deg = math.degrees(_rad)
    print(f'{prefix}degrees={_deg}')


model = model_loader.load_by_key(misc_utils.ModelKey.CUSTOM_SINE)
sin_90 = numpy.sin(math.radians(90))
test_args = [float(sin_90)]
predication = model.predict([test_args])

ans = ann_custom_sine.val_func(test_args)
show_rad_deg('ans',ans)
show_rad_deg('predication',predication[0])