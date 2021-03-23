import math
from functools import reduce


def sum_dict_values(*args):
    def reducer(accumulator, element):
        for key, value in element.items():
            accumulator[key] = accumulator.get(key, 0) + value
        return accumulator

    return reduce(reducer, args)


# noinspection PyStringFormat
def truncate(number: float, step) -> str:
    if step > 0:
        digits = int(-math.log10(step))
    else:
        return str(number)
    return f'%.{digits}f' % (int(number*10**digits)/10**digits)
