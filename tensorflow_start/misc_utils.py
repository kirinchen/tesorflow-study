from typing import List


def _gen_func(x: List[float]) -> int:
    ans = 0
    ans += x[0] * 2
    ans += x[1] * 3
    ans += x[2] * 5

    return int(ans)
