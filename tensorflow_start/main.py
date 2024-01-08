import math

import numpy
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}' + tf.__version__)  # Press Ctrl+Shift+B to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    sin_90 = numpy.sin(math.radians(90))
    print(sin_90)
    deg = math.asin(sin_90)
    print(deg)
    print(math.degrees(deg))
    time_points = np.linspace(0, 8 * np.pi, 8000)
    print(time_points)

    x = np.arange(1, 11)
    y = x * x

    plt.title("Line graph")
    plt.xlabel("X axis")
    plt.ylabel("Y axis")
    plt.plot(x, y, color ="red")
    plt.show()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
