import os
import math
import logging
import pickle5 as pickle
import pandas as pd
from functools import reduce

PICKLE_PROTOCOL = 4


def downgrade_pickle(filename):
    pickle.dump(pickle.load(open(filename, 'rb')), open(filename, 'wb'), protocol=PICKLE_PROTOCOL)


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


# noinspection PyBroadException
def is_serializable(obj):
    try:
        pickle.loads(pickle.dumps(obj))
        return True
    except:
        return False


def safe_save(self, datafile=None) -> bool:
    if datafile is None and hasattr(self, 'filename'):
        datafile = getattr(self, 'filename')
    try:
        make_bak_file(datafile)
        if hasattr(self, 'to_pickle'):
            self.to_pickle(datafile, protocol=PICKLE_PROTOCOL)
        else:
            pickle.dump(self, open(datafile, 'wb'), protocol=PICKLE_PROTOCOL)
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


def readable_kline(klines):
    """
    1499040000000,      // Open time
    "0.01634790",       // Open
    "0.80000000",       // High
    "0.01575800",       // Low
    "0.01577100",       // Close
    "148976.11427815",  // Volume
    1499644799999,      // Close time
    "2434.19055334",    // Quote asset volume
    308,                // Number of trades
    "1756.87402397",    // Taker buy base asset volume
    "28.46694368",      // Taker buy quote asset volume
    "17928899.62484339" // Ignore.
    """
    kline = pd.DataFrame(klines).apply(pd.to_numeric)
    columns = ['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 'Quote asset volume',
               'Number of trades', 'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore']
    kline.columns = columns
    kline['Open time'] = pd.to_datetime(kline['Open time'] * 10**6)
    return kline


def tz_remove_and_normalize(date):
    try:
        try:
            return pd.Timestamp(date).normalize().tz_convert(None)
        except TypeError:
            return pd.Timestamp(date).normalize()
    except ValueError:
        return tz_remove_and_normalize('now')


