import sys
import math


def bar(message, now: int, total: int):
    """
    :param message: string to print.
    :param now: the i-th iteration.
    :param total: total iteration num.
    :return:
    """
    rate = now / total
    rate_num = int(rate * 40)
    rate_nums = math.ceil(rate * 100)
    r = '\r%s:[%s%s]%d%%\t\t%d/%d' % (message, "=" * rate_num,
                                      " " * (40 - rate_num), rate_nums, now, total)
    if now == total:
        r += "\n"
    sys.stdout.write(r)
    sys.stdout.flush()
