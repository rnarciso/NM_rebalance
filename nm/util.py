import os
import sys
import math
import logging
import pandas as pd
from functools import reduce

if sys.version_info[:3] > (3, 7):
    import pickle5 as pickle
else:
    import pickle

AT_SIGN = ' às '
AVG_SLIPPAGE = 0.0045755
CLOSE = 'Close'
CLOSE_TIME = 'Close time'
COIN_MARKET_COLUMNS = ['volume_24h', 'percent_change_1h', 'percent_change_24h', 'percent_change_7d', 'market_cap']
COIN_HISTORY_FILE = 'history.dat'
DATE = 'date'
DEFAULT_COINS_IN_HISTORY_DATA = ['BTC']
DIFF = 'diff'
EXCHANGE_OPENING_DATE = '17 Aug, 2017'
KEYFILE = '.keys'
MAKER_PREMIUM = 0.1 / 100
MINIMUM_TIME_OFFSET = 2000
NM_COLUMNS = ['symbol', 'price', 'NM1', 'NM2', 'NM3', 'NM4', 'date']
NM_MAX = 4
NM_TIME_ZONE = 'Brazil/East'
NM2_RANGE = 17
NM4_RANGE = 20
NM_REPORT_DEFAULT_URL = 'http://127.0.0.1/nmREPORT.asp?NM='
NMDATA_FILE = 'nm_index.dat'
OPEN = 'Open'
OPEN_TIME = 'Open time'
ORDER_AMOUNT_REDUCING_FACTOR = 5 / 100
PICKLE_PROTOCOL = 5
QUOTE_ASSET = 'USDT'
RISK_FREE_DAILY_IRATE = 0.0001596535874
SINCE = '20191231'
SYMBOL = 'symbol'
STATEMENT_FILE = 'statement.dat'
UPDATED = 'atualizado'
UPDATED_ON: str = f'{UPDATED} em'
TA_DATA_FILE = 'ta_data.dat'
TOP_N_MAX = 4
YIELD = 'yield'
YIELD_FILE = 'yield.dat'

# Following constants are imported from Client later on
SIDE_SELL, SIDE_BUY, TIME_IN_FORCE_GTC, ORDER_STATUS_FILLED, ORDER_TYPE_LIMIT, ORDER_TYPE_LIMIT_MAKER, \
    ORDER_TYPE_MARKET = [None]*7


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
    return f'%.{digits}f' % (int(number * 10 ** digits) / 10 ** digits)


def is_serializable(obj):
    # noinspection PyPep8,PyBroadException
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
    kline['Open time'] = pd.to_datetime(kline['Open time'] * 10 ** 6)
    return kline


def tz_remove_and_normalize(date):
    try:
        try:
            return pd.Timestamp(date).normalize().tz_convert(None)
        except TypeError:
            return pd.Timestamp(date).normalize()
    except ValueError:
        return tz_remove_and_normalize('now')


def trim_run(method, *args, **kwargs):
    if len(args) >= 1 and isinstance(args[0], dict):
        kwargs.update(args[0])
        args = args[1:]
    elif len(kwargs) < 1:
        return method(*args)
    kwargs = {k.lower(): v for k, v in kwargs.items()}
    try:
        if hasattr(method, '__wrapped__'):
            method_params = method.__wrapped__.__code__
        else:
            method_params = method.__code__
        method_params = method_params.co_varnames[:method_params.co_argcount]
    except AttributeError:
        logging.error(f' Unable to retrieve arguments from function {method}.')
        method_params = ()

    kwargs = {k: kwargs.get(k.lower()) for k in method_params if k.lower() in kwargs.keys()}

    return method(*args, **kwargs)
