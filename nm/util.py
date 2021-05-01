import os
import sys
import math
import logging
import pandas as pd
from tqdm import tqdm
from functools import reduce
from time import time as sys_time

if sys.version_info[:3] < (3, 7):
    import pickle5 as pickle
else:
    import pickle

# noinspection PyBroadException
try:
    # noinspection PyPep8Naming,PyUnresolvedReferences,PyPackageRequirements
    from config import data_files_path as DATA_FOLDER
except Exception as e:
    DATA_FOLDER = ''

AT_SIGN = ' às '
ADJUSTED_CLOSE = 'Adjusted close'
AVG_SLIPPAGE = 1.1641814/100
CLOSE = 'Close'
CLOSE_TIME = 'Close time'
COIN_MARKET_COLUMNS = ['volume_24h', 'percent_change_1h', 'percent_change_24h', 'percent_change_7d', 'market_cap']
COIN_HISTORY_FILE = os.path.join(DATA_FOLDER, 'history.dat')
DATE = 'date'
DEFAULT_COINS_IN_HISTORY_DATA = ['BTC']
DEPOSIT = 'Deposit'
DIFF = 'diff'
EXCHANGE_OPENING_DATE = '17 Aug, 2017'
FEES = 'Fees'
FEES_DATA_FILE = os.path.join(DATA_FOLDER, 'fees.dat')
KEYFILE = '.keys'
INFOFILE = os.path.join(DATA_FOLDER, 'info.dat')

MAKER_PREMIUM = 0.1 / 100
MINIMUM_TIME_OFFSET = 2000
NM_COLUMNS = ['symbol', 'price', 'NM1', 'NM2', 'NM3', 'NM4', 'date']
NM_INFO_XLSX = '~/Downloads/NM_Guathan.xlsx'
NM_MAX = 4
NM_REPORT_DEFAULT_URL = 'http://127.0.0.1/nmREPORT.asp?NM='
NM_TIME_ZONE = 'Brazil/East'
NM2_RANGE = 17
NM4_RANGE = 20
NMDATA_FILE = os.path.join(DATA_FOLDER, 'nm_index.dat')
OPEN = 'Open'
OPEN_TIME = 'Open time'
ORDER_AMOUNT_REDUCING_FACTOR = 5 / 100
PICKLE_PROTOCOL = 5
QUOTE_ASSET = 'USDT'
QUOTE_VALUE = 'Quote Value'
RISK_FREE_DAILY_IRATE = 0.0001596535874
SINCE = '20191231'
SLIPPAGE = 'Slippage'
STATEMENT_FILE = os.path.join(DATA_FOLDER, 'statement.dat')
SYMBOL = 'symbol'
UPDATED = 'atualizado'
UPDATED_ON: str = f'{UPDATED} em'
TA_DATA_FILE = os.path.join(DATA_FOLDER, 'ta_data.dat')
TO_DATE = 'to_date'
TOP_N_MAX = 4
YIELD = 'yield'
YIELD_FILE = os.path.join(DATA_FOLDER, 'yield.dat')

# Following constants are imported from Client later on
SIDE_SELL, SIDE_BUY, TIME_IN_FORCE_GTC, TIME_IN_FORCE_IOC, ORDER_STATUS_FILLED, ORDER_TYPE_LIMIT, \
    ORDER_TYPE_LIMIT_MAKER, ORDER_TYPE_MARKET, ORDER_STATUS_PARTIALLY_FILLED, ORDER_STATUS_NEW,\
    ORDER_STATUS_PENDING_CANCEL, ORDER_STATUS_CANCELED, ORDER_STATUS_EXPIRED, ORDER_STATUS_REJECTED \
    = [None]*14


def adjust(from_date, to_date, default_date=None):
    if from_date is None:
        try:
            from_date = default_date
            if pd.Timestamp(from_date) is pd.NaT:
                raise ValueError
        except ValueError:
            from_date = SINCE
        from_date = next_date(from_date)
    else:
        from_date = tz_remove_and_normalize(from_date)
    if tz_remove_and_normalize(from_date) < tz_remove_and_normalize('utc'):
        if to_date is None:
            to_date = tz_remove_and_normalize('utc')
        elif isinstance(to_date, int):
            to_date = next_date(from_date, to_date)
        else:
            to_date = next_date(to_date)
        if to_date < from_date:
            date = to_date
            to_date = from_date
            from_date = date
        from_date = tz_remove_and_normalize(from_date)
        to_date = tz_remove_and_normalize(to_date)
    return from_date, to_date


def data_frame_decimal_convert(nmdf):
    nmdf[nmdf.columns[0]] = pd.to_datetime(nmdf[nmdf.columns[0]], dayfirst=True)
    nmdf[nmdf.columns[1]] = pd.to_numeric(nmdf[nmdf.columns[1]].str.replace('%', '').str.replace(',', '.'))
    return nmdf.rename({nmdf.columns[0]: 'date', nmdf.columns[1]: 'yield'}, axis='columns').set_index('date')


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
    except Exception:
        return False


def safe_save(self, datafile=None) -> bool:
    if datafile is None and hasattr(self, 'filename'):
        datafile = getattr(self, 'filename')
    # noinspection PyShadowingNames
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


def log_error(error):
    exc_type, exc_obj, exc_tb = sys.exc_info()
    if exc_tb is not None:
        module_name = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        logging.error(f'{error} Type: {exc_type}, Module: {module_name}, Line # {exc_tb.tb_lineno}')
    else:
        logging.error(f'{error} Type: {exc_type}')


def make_bak_file(datafile):
    if os.path.exists(datafile):
        if os.path.exists(os.path.splitext(datafile)[0] + '.bak'):
            os.remove(os.path.splitext(datafile)[0] + '.bak')
        os.rename(datafile, os.path.splitext(datafile)[0] + '.bak')


def next_date(date, days=1):
    return tz_remove_and_normalize(date) + pd.Timedelta(days, 'days')


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
    if date == 'utc':
        return pd.Timestamp.now('utc').normalize().tz_convert(None)
    try:
        try:
            return pd.Timestamp(date).normalize().tz_convert(None)
        except TypeError:
            return pd.Timestamp(date).normalize()
    except ValueError:
        return tz_remove_and_normalize('utc')


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


def patch_time(offset):
    return sys_time() + offset


tqdm.pandas()
