from enum import Enum
from typing import List


def _gen_func(x: List[float]) -> int:
    ans = 0
    ans += x[0] * 2
    ans += x[1] * 3
    ans += x[2] * 5
    return int(ans)


def _gen_func_float(x: List[float]) -> float:
    ans = 0
    ans += x[0] * 2
    ans += x[1] * 3
    ans += x[2] * 5
    if x[0] < 0.2:
        ans = 100
    return ans


class ModelKey(Enum):
    CUSTOM = 'CUSTOM'
    CUSTOM_FLOAT = 'CUSTOM_FLOAT'
    CUSTOM_SINE = 'CUSTOM_SINE'
