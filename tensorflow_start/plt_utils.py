from typing import List

from matplotlib import pyplot as plt


class SerMeta:

    def __init__(self, val_array: list, color: str, label: str):
        self.val_array: list = val_array
        self.color: str = color
        self.label: str = label


def show_array(y: list, extend_list: List[SerMeta] = []):
    x_idx_array = [idx for idx, val in enumerate(y)]
    # plt.title("Line graph")
    # plt.xlabel("X axis")
    # plt.ylabel("Y axis")
    # plt.plot(x_idx_array, y, color="red")
    # plt.show()

    fig, ax1 = plt.subplots(figsize=(8, 8))
    ax1.plot(x_idx_array, y, color='red')
    ax1.set_xlabel("idx")
    ax1.set_ylabel("y", color='red', fontsize=14)
    ax1.tick_params(axis="y", labelcolor='red')

    for extend in extend_list:
        ax2 = ax1.twinx()
        ax2.plot(x_idx_array, extend.val_array, color=extend.color)
        ax2.set_ylabel(extend.label, color=extend.color, fontsize=14)
        ax2.tick_params(axis="y", labelcolor=extend.color)
    fig.suptitle("Temperature down, price up", fontsize=20)
    # fig.autofmt_xdate()
    plt.show()