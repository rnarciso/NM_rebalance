import os
import math
import logging
import pickle5 as pickle
import pandas as pd
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


def safe_save(self, datafile=None) -> bool:

    if datafile is None and hasattr(self, 'filename'):
        datafile = getattr(self, 'filename')
    try:
        make_bak_file(datafile)
        if hasattr(self, 'to_pickle'):
            self.to_pickle(datafile)
        else:
            pickle.dump(self, open(datafile, 'wb'))
        return True
    except Exception as e:
        logging.error(e)
        return False


def make_bak_file(datafile):
    if os.path.exists(datafile):
        if os.path.exists(os.path.splitext(datafile)[0] + '.bak'):
            os.remove(os.path.splitext(datafile)[0] + '.bak')
        os.rename(datafile, os.path.splitext(datafile)[0] + '.bak')


def next_date(date, days=1):
    return pd.Timestamp(date) + pd.Timedelta(days, 'days')
