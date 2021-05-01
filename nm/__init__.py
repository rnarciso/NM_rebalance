import ta
import json
import time
import random
import asyncio
import pathlib
import warnings
import numpy as np
from nm.util import *
from functools import partial
# noinspection PyPackageRequirements
from binance.client import Client
from collections.abc import Iterable
from itertools import combinations
# noinspection PyPackageRequirements
from binance.exceptions import BinanceAPIException

# import constants from Client
for const in globals().copy().keys():
    if globals()[const] is None and Client.__dict__.get(const) is not None:
        globals()[const] = Client.__dict__[const]


class Fees:

    def __init__(self, account=None, default_fee_type='taker', datafile=None, load=False, ):
        self._df = None
        self._binance_api = account
        self._filename = datafile
        self.default_fee_type = default_fee_type
        if load:
            self._df = self.load(datafile)

    def __repr__(self):
        return self.df.__repr__()

    def __str__(self):
        return self.df.__str__()

    def __getattr__(self, attr):
        if attr[0] != '_' and not hasattr(super(), attr) and hasattr(self.df.T, attr):
            setattr(self, attr, getattr(self.df.T, attr))
            return getattr(self, attr)
        else:
            super().__getattribute__(attr)

    def __setitem__(self, key, value):
        if not hasattr(super(), key) and hasattr(self.df.T, key):
            setattr(self.df.T, key, value)
        else:
            super().__setattr__(key, value)

    def __getitem__(self, item):
        if item in self.df.T.keys():
            return self.df.loc[item, self.default_fee_type]
        else:
            raise KeyError

    @property
    def binance_api(self):
        if self._binance_api is None:
            self._binance_api = BinanceAccount()
        return self._binance_api

    @property
    def filename(self):
        if self._filename is None:
            try:
                # noinspection PyPep8Naming,PyShadowingNames
                from config import fees_file as FEES_DATA_FILE
            except (ImportError, ModuleNotFoundError):
                # noinspection PyGlobalUndefined
                global FEES_DATA_FILE
            self._filename = FEES_DATA_FILE
        return self._filename

    @filename.setter
    def filename(self, value):
        self._filename = value

    def for_symbols(self, symbols, fee_type=None):
        if fee_type is None:
            return self.df[self.df.index.isin([symbols])][self.default_fee_type]
        else:
            return self.df[self.df.index.isin([symbols])][fee_type]

    # noinspection PyShadowingNames
    @property
    def df(self):
        try:
            if self._df is None:
                self._df = self.load(self.filename)
            if tz_remove_and_normalize('now') - self.last_update > pd.Timedelta(1, 'day'):
                self._df = pd.DataFrame.from_dict(self.get_trade_fee()['tradeFee']).set_index('symbol')
                self.save()
        except Exception as e:
            log_error(e)
        return self._df

    @df.setter
    def df(self, value):
        self._df = value

    @property
    def index(self):
        return self.df.index

    # noinspection PyShadowingNames
    @property
    def last_update(self):
        try:
            file_name = pathlib.Path(self.filename)
            assert file_name.exists(), f'No such file: {file_name}'
            return pd.Timestamp.fromtimestamp(file_name.stat().st_mtime)
        except AssertionError as e:
            logging.debug(e)
            return tz_remove_and_normalize(EXCHANGE_OPENING_DATE)

    @property
    def loc(self):
        return self.df.loc

    # noinspection PyUnboundLocalVariable,PyShadowingNames
    def load(self, datafile=None):
        if datafile is None:
            datafile = self.filename
        try:
            return pd.read_pickle(datafile)
        except FileNotFoundError:
            return dict()
        except ValueError:
            downgrade_pickle(datafile)
            return self.load(datafile)
        except Exception as e:
            log_error(e)

    def save(self):
        if self.filename is None:
            self.filename = FEES_DATA_FILE
        safe_save(self.df, self.filename)


class SymbolInfo:
    def __init__(self, account=None, datafile=None, load=False):
        self._aggregated_info = None
        self._binance_api = account
        self._filename = datafile
        if load:
            self._aggregated_info = self.load(datafile)

    def __setitem__(self, key, value):
        self.aggregated_info.update({key: value})

    def __getitem__(self, item):
        if item not in self.info.keys():
            self.get_info_from_binance(item)
        return self.info[item]

    def __getattr__(self, attr):
        if attr[0] != '_' and not hasattr(super(), attr):
            if attr in self.aggregated_info.keys():
                return self.aggregated_info[attr]
        else:
            return super().__getattribute__(attr)

    # noinspection PyShadowingNames
    def get_info_from_binance(self, pair):
        try:
            info = self.binance_api.get_symbol_info(pair)
            symbol = info.pop('symbol')
            for item in ['filters', 'orderTypes', 'permissions']:
                self.aggregated_info.setdefault(item, {})[symbol] = info.pop(item)
            self.aggregated_info.setdefault('info', {})[symbol] = info
            self.save()
        except BinanceAPIException as e:
            log_error(e)

    def __repr__(self):
        return pd.DataFrame(self.info).T.__repr__()

    def __str__(self):
        return pd.DataFrame(self.info).T.__str__()

    @property
    def aggregated_info(self):
        if self._aggregated_info is None:
            self._aggregated_info = self.load(self.filename)
        return self._aggregated_info

    @property
    def binance_api(self):
        if self._binance_api is None:
            self._binance_api = BinanceAccount()
        return self._binance_api

    @property
    def filename(self):
        if self._filename is None:
            try:
                # noinspection PyPep8Naming,PyShadowingNames
                from config import info_file as INFOFILE
            except (ImportError, ModuleNotFoundError):
                # noinspection PyGlobalUndefined
                global INFOFILE
            self._filename = INFOFILE
        return self._filename

    @filename.setter
    def filename(self, value):
        self._filename = value

    # noinspection PyShadowingNames
    @property
    def filters(self):
        try:
            return {k: {v: {kk: v for kk, v in dd.items() if kk != 'filterType'}
                    for dd in l for kk, v in dd.items() if kk == 'filterType'}
                    for k, l in self.aggregated_info['filters'].items()}
        except Exception as e:
            log_error(e)
            return {}

    def for_symbols(self, symbols, attribute):
        pass

    @property
    def info(self):
        return self.aggregated_info.get('info')

    # noinspection PyShadowingNames
    def min_notional(self, symbol):
        try:
            offline_results = [float(v) for s, l in self.aggregated_info.get('filters', {}).items() for d2 in l
                               for k, v in d2.items() if s == symbol and d2.get('filterType') == 'MIN_NOTIONAL' and
                               k == 'minNotional']
            if len(offline_results) < 1:
                self.get_info_from_binance(symbol)
                return self.min_notional(symbol)
            else:
                return np.mean(offline_results)
        except Exception as e:
            logging.debug(e)
            return 10.0

    # noinspection PyShadowingNames
    @property
    def last_update(self):
        try:
            file_name = pathlib.Path(self.filename)
            assert file_name.exists(), f'No such file: {file_name}'
            return pd.Timestamp.fromtimestamp(file_name.stat().st_mtime)
        except AssertionError as e:
            logging.debug(e)
            return tz_remove_and_normalize(EXCHANGE_OPENING_DATE)

    # noinspection PyUnboundLocalVariable,PyShadowingNames
    def load(self, datafile=None):
        if datafile is None:
            datafile = self.filename
        try:
            return pd.read_pickle(datafile)
        except FileNotFoundError:
            return dict()
        except ValueError:
            downgrade_pickle(datafile)
            return self.load(datafile)
        except Exception as e:
            log_error(e)

    def save(self):
        if self.filename is None:
            self.filename = FEES_DATA_FILE
        safe_save(self.aggregated_info, self.filename)


class Deposits:
    def __init__(self):
        self._data = pd.DataFrame()
        self._symbols = None
        self._binance_api = None

    def __repr__(self):
        return self._data.__repr__()

    def __str__(self):
        return self._data.__str__()

    def __getattr__(self, attr):
        if attr[0] != '_' and not hasattr(super(), attr) and hasattr(self.df, attr):
            setattr(self, attr, getattr(self.df, attr))
            return getattr(self, attr)
        else:
            super().__getattribute__(attr)

    def __setitem__(self, key, value):
        if not hasattr(super(), key) and hasattr(self.df, key):
            setattr(self._df, key, value)
        else:
            super().__setattr__(key, value)

    def __getitem__(self, item):
        if item in self.df.keys():
            return self.df[item]
        else:
            raise KeyError

    @property
    def binance_api(self):
        if self._binance_api is None:
            self._binance_api = BinanceAccount(connect=True)
        return self._binance_api

    @property
    def df(self):
        return self._data

    @property
    def symbols(self):
        if self._symbols is None:
            self._symbols = [i.get('symbol') for i in self.binance_api.get_all_tickers()]
        return self._symbols

    # noinspection PyShadowingNames
    def add(self, date, value, currency=QUOTE_ASSET, nm_index=1):
        date = pd.Timestamp(date)
        transaction = pd.DataFrame()
        transaction.loc[date, 'Amount'] = value
        transaction.loc[date, 'NM'] = str(nm_index)
        transaction.loc[date, 'Currency'] = currency
        if currency == QUOTE_ASSET:
            transaction.loc[date, 'Quote Value'] = value
        elif f'{QUOTE_ASSET}{currency}' in self.symbols:
            market_data = readable_kline(self.binance_api.get_historical_klines(f'{QUOTE_ASSET}{currency}',
                                                                                Client.KLINE_INTERVAL_1DAY,
                                                                                *[date.strftime('%Y-%m-%d')] * 2))
            transaction.loc[date, 'Quote Value'] = value / market_data['High'].iloc[0]
        else:
            try:
                market_data = self.binance_api.get_historical_klines(f'{currency}{QUOTE_ASSET}',
                                                                     Client.KLINE_INTERVAL_1DAY,
                                                                     *[date.strftime('%Y-%m-%d')] * 2)
                transaction.loc[date, 'Quote Value'] = value * market_data['Low'].iloc[0]
            except BinanceAPIException as e:
                log_error(e)
        self._data = self._data.append(transaction).reset_index().groupby(['index', 'NM']).sum().reset_index(
                ).set_index('index')
        return self

    def deposit(self, *args, **kwargs):
        return self.add(*args, **kwargs)

    def reset(self):
        self._data = pd.DataFrame()


class Backtest:
    def __init__(self, advisor=None):
        if advisor is None:
            advisor = NMData()
        self.advisor = advisor
        self.fees = Fees()
        self._coin_data = None
        self.yield_df_columns = [f'NM{i}' for i in range(1, NM_MAX + 1)]
        self._binance_api = None
        for column in self.yield_df_columns.copy():
            for suffix in ['', ' time']:
                for feature in ['bottom', 'top']:
                    self.yield_df_columns.append(f'{column} {feature}{suffix}')

    def __repr__(self):
        return f'<Backtest container class at {hex(id(self))}>'

    @property
    def binance_api(self):
        if self._binance_api is None:
            self._binance_api = BinanceAccount(connect=True)
        return self._binance_api

    @property
    def coin_data(self):
        if self._coin_data is None:
            self._coin_data = CoinData()
        return self._coin_data

    @staticmethod
    def default_to_date(to_date):
        if to_date is None:
            to_date = tz_remove_and_normalize('utc')
        else:
            to_date = pd.Timestamp(to_date)
        return to_date

    def index_yield_for_date(self, nm_index, date, top_n=4):
        try:
            return self.coin_data.yield_for_coins(self.advisor.get(nm_index, date)[:top_n].index,
                                                  next_date(date, -1), date)['yield'].mean()
        except KeyError:
            return 0

    def nm_index_yield_for_period(self, nm_index, from_date, to_date=None, top_n=4, fees=False, slippage=None,
                                  interval=1, return_df=False):
        df = pd.DataFrame()
        last_coins = []
        from_date = tz_remove_and_normalize(from_date)
        if to_date is None:
            to_date = tz_remove_and_normalize('utc')
        elif isinstance(to_date, int):
            to_date = next_date(from_date, to_date)
        accrued_yield = 0
        for date in tqdm(pd.date_range(from_date, to_date, freq=f'{interval}D')):
            try:
                coins = self.advisor.get(nm_index, date).index[:top_n]
                yield_for_period = self.coin_data.yield_for_coins(coins, date)
            except KeyError:
                yield_for_period = 0
                coins = []
            if fees:
                try:
                    # noinspection PyUnresolvedReferences
                    fee_for_date = self.fees.for_symbols([f'{c}{QUOTE_ASSET}' for c in set(coins) - set(last_coins)])
                except NameError:
                    fee_for_date = self.fees.for_symbols([f'{c}{QUOTE_ASSET}' for c in coins])
                fee_for_date = np.average(fee_for_date) * 2 if date > from_date else 1
                if slippage is not None:
                    fee_for_date *= (1 + slippage)
                yield_for_period *= (1 - fee_for_date)
                last_coins = coins

            accrued_yield += yield_for_period
            if return_df:
                df.loc[date, f'NM index {nm_index} /{interval}D'] = yield_for_period
                # noinspection SpellCheckingInspection
                df.loc[date, f'NM index {nm_index} ACUM.'] = accrued_yield

        if return_df:
            return df.shift(1).dropna()
        else:
            return accrued_yield

    def account_yield_for_period(self, accounts, from_date, *kwargs):
        if isinstance(accounts, dict):
            accounts = [accounts]
        return np.average([self.nm_index_yield_for_period(account.get('index'),
                                                          from_date, *kwargs) for account in accounts])

    def yield_simulation(self, deposits: Deposits, to_date=None, fees=True, slippage=AVG_SLIPPAGE, top_n=4,
                         interval=1):

        def add_deposits(row):
            dfs.setdefault(row['NM'], pd.DataFrame()).loc[row.name, DEPOSIT] = row[QUOTE_VALUE]

        # noinspection PyShadowingNames
        def add_yield(row):
            date = row.name
            date = next_date(date, - 1)
            to_date = row.to_date
            if not any([isinstance(d, type(pd.NaT)) for d in (date, to_date)]):
                coins = tuple(self.advisor.get(nm, date).index[:top_n])
                row[f'NM{nm} {YIELD}'] = self.coin_data.yield_for_coins(coins, date, next_date(date, interval - 1))
                row[f'NM{nm} coins'] = ','.join(coins)
                row[OPEN] = previous_row.get(ADJUSTED_CLOSE, 0)
                row[CLOSE] = row[OPEN] * (1 + row[f'NM{nm} {YIELD}']) + row[DEPOSIT]
                if fees:
                    changes = set(row[f'NM{nm} coins'].split(',')).symmetric_difference(
                            previous_row.get(f'NM{nm} coins', '').split(','))
                    change_percentage = len(changes) / top_n
                    row[FEES] = self.fees.for_symbols([f'{coin_name}{QUOTE_ASSET}' for coin_name in changes]).mean()
                    if np.isnan(row.get(FEES, np.nan)):
                        row[FEES] = 0.001
                    row[FEES] *= row[CLOSE] * change_percentage
                    if slippage is not None:
                        row[SLIPPAGE] = row[CLOSE] * slippage * change_percentage
                    else:
                        row[SLIPPAGE] = 0
                else:
                    row[FEES] = 0
                    row[SLIPPAGE] = 0
                row[ADJUSTED_CLOSE] = row[CLOSE] - (row[FEES] + row[SLIPPAGE])
                previous_row.update(row)
            return row

        dfs = {}
        to_date = self.default_to_date(to_date)
        deposits.apply(add_deposits, axis=1)
        for nm in dfs.keys():
            dfs[nm] = pd.concat([dfs[nm], pd.DataFrame(index=pd.date_range(start=dfs[nm].index.min(),
                                                                           end=to_date, freq=f'{interval}D',
                                                                           closed='right'))]
                                ).fillna(0).reset_index().groupby('index').sum()
            dfs[nm][TO_DATE] = dfs[nm].index
            dfs[nm][TO_DATE] = dfs[nm][TO_DATE].shift(-1)
            dfs[nm] = dfs[nm].dropna()
            previous_row = {}
            dfs[nm] = dfs[nm].progress_apply(add_yield, axis=1)
        consolidated_df = pd.concat(dfs.values()).reset_index().groupby('index').sum()
        consolidated_df.index = consolidated_df.index.shift(1, freq='D')
        consolidated_df.index.name = None
        leading_columns = [OPEN, DEPOSIT, CLOSE, FEES, SLIPPAGE]
        trailing_columns = [ADJUSTED_CLOSE]
        consolidated_df = consolidated_df[leading_columns + list(set(consolidated_df.columns) -
                                          (set(leading_columns) | set(trailing_columns))) + trailing_columns]

        return consolidated_df

    @staticmethod
    def predict_slippage(consolidated_df, real_final_balance, final_balance_reference=-1,
                         initial_slippage=AVG_SLIPPAGE):

        # noinspection PyShadowingNames
        def readjust_closure(df, slippage, previous_slippage=AVG_SLIPPAGE):

            df = df.copy()
            df['slippage_percentage'] = df[SLIPPAGE] / df[CLOSE] / previous_slippage
            df['combined_yield'] = (df[CLOSE] - df[DEPOSIT]) / df[OPEN] - 1
            df = df.fillna(0)
            previous_row = {}

            def new_slippage(row):
                row[OPEN] = previous_row.get(ADJUSTED_CLOSE, 0)
                row[CLOSE] = row[OPEN] * (1 + row['combined_yield']) + row[DEPOSIT]
                row[SLIPPAGE] = row[CLOSE] * row['slippage_percentage'] * slippage
                row[ADJUSTED_CLOSE] = row[CLOSE] - (row[FEES] + row[SLIPPAGE])
                previous_row.update(row)
                return row

            df.apply(new_slippage, axis=1)
            return df

        previous_slippage = initial_slippage
        while True:
            calculated_final_balance = consolidated_df[ADJUSTED_CLOSE].iloc[final_balance_reference]
            if abs(calculated_final_balance / real_final_balance - 1) < 0.5 / 100:
                print(f'Slippage of {previous_slippage*100:.2f}% is accurate enough!')
                break
            else:
                print(f'Slippage of {previous_slippage*100:.2f}% is not accurate, calculated '
                      f'final balance is {calculated_final_balance}!')
                if final_balance_reference == -1:
                    final_balance_reference = None
                else:
                    final_balance_reference -= 1
                total_slippage_cost = consolidated_df.Slippage[:final_balance_reference].sum()
                slippage_to_real_balance = calculated_final_balance - real_final_balance + total_slippage_cost
                predicted_slippage = slippage_to_real_balance * previous_slippage / total_slippage_cost

                consolidated_df = readjust_closure(consolidated_df, slippage=predicted_slippage)  # TODO fix this


class BinanceAccount:
    _time_offset: int

    # noinspection PyShadowingNames
    def __init__(self, key_name: str = None, connect=False, include_locked=False, config=None):
        self._balance = {}
        self._client = None
        if isinstance(key_name, dict):
            self._config = key_name
            key_name = config.get('account_name')
        if isinstance(config, dict):
            self._config = config
        if key_name is not None:
            try:
                from config import accounts
                for account in accounts:
                    if key_name == account.get('account_name'):
                        self._config = account
                        break
                else:
                    logging.error(f'Unable to find account {key_name} description on config file!')
                    return
            except Exception as e:
                log_error(e)
            self._key_name = key_name
        self._include_locked_asset_in_balance = include_locked
        self._info = None
        self._min_notational = None
        self._lotsize = None
        self._time_offset = 0
        self.fees = Fees(self)
        self.connected = False
        if connect:
            self.connect(key_name)

    @property
    def time_offset(self):
        if self._time_offset is None:
            self._time_offset = (self._client.get_server_time().get("serverTime") - (sys_time() * 10 ** 3)) // 10 ** 3
            time.time = partial(patch_time, self._time_offset)
        return self._time_offset

    @time_offset.setter
    def time_offset(self, value):
        self._time_offset = value
        time.time = partial(patch_time, self._time_offset)

    # noinspection PyShadowingNames
    def __getattr__(self, attr):
        if attr[0] != '_' and not hasattr(super(), attr):
            try:
                if hasattr(self, '_config') and self._config is not None and attr in self._config.keys():
                    setattr(self, attr, self._config[attr])
                    return self._config[attr]
                else:
                    if self._client is None:
                        self.connect()
                    if hasattr(self._client, attr):
                        setattr(self, attr, getattr(self._client, attr))
                        return getattr(self._client, attr)
            except BinanceAPIException as e:
                log_error(e)
        super().__getattribute__(attr)

    # noinspection PyShadowingNames
    def avg_price(self, amount, market, side=SIDE_BUY, add_fee=True):
        try:
            book_orders = self.get_order_book(symbol=market).get('asks' if side == SIDE_BUY else 'bids')
        except Exception as e:
            log_error(e)
            return 0.0, 0.0
        if len(book_orders) > 0:
            amount_left = amount
            amount_bought = 0.0
            cost = 0.0
            avg_price = 0.0
            for price, amount in book_orders:
                amount = float(amount)
                price = float(price)
                amount = amount if amount < amount_left else amount_left
                avg_price = avg_price * amount_bought + price * amount
                cost += price * (amount if amount < amount_left else amount_left)
                amount_bought += amount
                amount_left -= amount
                avg_price /= amount_bought
                if not amount_left > 0:
                    break
            if add_fee:
                fee = self.fees[market]
                cost *= 1 + fee

            return dict(price=avg_price, quote_amount=cost)

    # noinspection PyShadowingNames
    @property
    def balance(self):
        pd.options.display.float_format = '{:,.2f}'.format
        df = pd.DataFrame(data=pd.Series(self._balance) if len(self._balance) > 0 else
                          pd.Series(self.refresh_balance()).sort_values(), columns=['Amount'])
        try:
            prices = {i.get('symbol'): i.get('price') for i in self.get_all_tickers()}
            df[f'{QUOTE_ASSET} Value'] = df.apply(lambda row: float(prices.get(f'{row.name}{QUOTE_ASSET}', '0')) *
                                                  row['Amount'] if row.name != QUOTE_ASSET else row['Amount'], axis=1)
            df['%'] = df[f'{QUOTE_ASSET} Value'] / df[f'{QUOTE_ASSET} Value'].sum() * 100
        except Exception as e:
            log_error(e)
        return df

    @property
    def client(self):
        if self._client is None:
            self._client = self.connect(self.key_name)
        return self._client

    # noinspection PyShadowingNames
    @property
    def config(self):
        if self._config is None:
            try:
                from config import accounts
                self._config = accounts[0]
            except Exception as e:
                log_error(e)
        return self._config

    @config.setter
    def config(self, values):
        self._config = values

    # noinspection PyUnboundLocalVariable,PyShadowingNames,PyShadowingNames,PyShadowingNames
    def connect(self, key_name: str = None):
        if not hasattr(self, '_config') or self._config is None:
            if key_name is None:
                key_name = 'binance'
            try:
                with open(KEYFILE, 'r') as file:
                    __keys__ = file.read()
                api_key = json.loads(__keys__)[f'{key_name}_key']
                api_secret = json.loads(__keys__)[f'{key_name}_api_secret']
            except FileNotFoundError:
                logging.info(' Key file not found!')
            except json.JSONDecodeError:
                log_error(' Invalid Key file format!')
            except Exception as e:
                log_error(e)
        else:
            api_key = self._config.get('api_key')
            api_secret = self._config.get('api_secret')
        try:
            logging.info(' Connecting ')
            try:
                # noinspection PyUnboundLocalVariable
                self._client = Client(api_key, api_secret)
                logging.debug(f'Time offset set to {self.time_offset} ms')
                self.connected = True
            except NameError:
                try:
                    self._client = Client()
                    self.connected = True
                except BinanceAPIException:
                    self.connected = False
            except Exception as e:
                log_error(e)
                self.connected = False

            return self._client
        except Exception as e:
            log_error(e)

    def convert_small_balances(self, base_asset='BNB'):
        balance = self.balance
        small_balances = [asset for asset in balance.index
                          if balance.loc[asset, f'{QUOTE_ASSET} Value'] < self.minimal_order(f'{asset}{QUOTE_ASSET}')]
        try:
            bnb_index = small_balances.index(base_asset)
            small_balances.pop(bnb_index)
        except ValueError:
            pass
        if len(small_balances) > 0:
            try:
                self.transfer_dust(asset=','.join(small_balances))
            except BinanceAPIException as e:
                if str(e).find('Only can be requested once within 6 hours') < 0:
                    log_error(f'{e}. Assets: {small_balances}.')

    @property
    def key_name(self):
        if hasattr(self, '_keyname'):
            return self._key_name

    @key_name.setter
    def key_name(self, value):
        self._key_name = value

    @property
    def info(self):
        if self._info is None:
            self._info = SymbolInfo(self)
        return self._info

    @property
    def lotsize(self):
        if self._lotsize is None:
            self._lotsize = {k:dd for k, d in self.info.filters.items() for kk, dd in d.items() if kk == 'LOT_SIZE'}
        return self._lotsize

    def min_amount(self, pair):
        try:
            return self.minimal_order(pair)  / float(self.get_avg_price(symbol=pair)['price'])
        except (KeyError, TypeError):
            return self.account.minimal_order(pair)

    # noinspection PyShadowingNames
    @property
    def min_notational(self):
        try:
            return {k: v for k, d in self.info.filters.items() for kk, v in d.items() if kk == 'MIN_NOTIONAL'}
        except Exception as e:
            log_error(e)
            return {}

    def minimal_order(self, pair):
        if self._info is None:
            self._info = SymbolInfo(self)
        return self._info.min_notional(pair)

    # noinspection PyShadowingNames
    def step_size(self, pair, recursive=False):
        try:
            return self.info.filters[pair]['LOT_SIZE']['stepSize']
        except KeyError:
            try:
                self.info.get_info_from_binance(pair)
                if not recursive:
                    return self.step_size(pair, True)
                else:
                    raise KeyError
            except Exception as e:
                logging.debug(e)
                return '0.001'

    # noinspection PyShadowingNames
    def refresh_balance(self, include_locked: bool = None):
        retries = 3
        while retries > 0:
            try:
                assets = self.get_account()
                break
            except BinanceAPIException as e:
                logging.debug(e)
                self.time_offset = None
                retries -= 1
        else:
            log_error(' Unable to retrieve balances from Binance!')
            return pd.Series(self._balance).sort_values()
        if include_locked is None:
            include_locked = self._include_locked_asset_in_balance
        if include_locked:
            self._balance = {a['asset']: float(a['locked']) + float(a['free']) for a in assets['balances'] if
                             float(a['locked']) > 0.0 or float(a['free']) > 0.0}
        else:
            self._balance = {a['asset']: float(a['free']) for a in assets['balances'] if float(a['free']) > 0.0}
        return pd.Series(self._balance).sort_values()

    # noinspection PyShadowingNames
    def round_price(self, value, pair, recursive=False):
        try:
            decimals = int(round(-math.log10(float(self.info.filters[pair]['PRICE_FILTER']['tickSize'])), 0))
            return round(value, decimals)
        except KeyError:
            try:
                self.info.get_info_from_binance(pair)
                if not recursive:
                    return self.round_price(value, pair, True)
                else:
                    raise KeyError
            except Exception as e:
                logging.debug(e)
                return round(value, 2)

    def round_amount(self, value, pair):
        decimals = int(round(-math.log10(float(self.step_size(pair)))))
        return round(value, decimals)


class CoinData:
    def __init__(self, datafile=None, load=False):
        self._assets = None
        self._binance_api = None
        self._filename = datafile
        self._history = None
        if load:
            self._history = self.load(datafile)

    def __repr__(self):
        return self.history.__repr__()

    def __str__(self):
        return self.history.__str__()

    @property
    def binance_api(self):
        if self._binance_api is None:
            self._binance_api = BinanceAccount(connect=True)
        return self._binance_api

    @property
    def assets(self):
        if self._assets is None:
            self._assets = NMData().assets
        return self._assets

    @property
    def filename(self):
        if self._filename is None:
            try:
                # noinspection PyPep8Naming,PyShadowingNames
                from config import coin_file as COIN_HISTORY_FILE
            except (ImportError, ModuleNotFoundError):
                # noinspection PyGlobalUndefined
                global COIN_HISTORY_FILE
            self._filename = COIN_HISTORY_FILE
        return self._filename

    def get(self, coin, from_date=None, to_date=None):
        try:
            history = self.history[self.history[SYMBOL] == coin]
        except KeyError:
            return pd.DataFrame()
        return history[from_date:to_date]

    def history_for(self, coin):
        return self.history[self.history[SYMBOL] == coin]

    @property
    def history(self):
        if self._history is None:
            self._history = self.load(self._filename)
        return self._history

    @history.setter
    def history(self, value):
        self._history = value

    # noinspection PyShadowingNames
    @property
    def last_update(self):
        try:
            filename = pathlib.Path(self.filename)
            assert filename.exists(), f'No such file: {filename}'
            return pd.Timestamp.fromtimestamp(filename.stat().st_mtime)
        except AssertionError as e:
            logging.debug(e)
            return tz_remove_and_normalize(EXCHANGE_OPENING_DATE)

    # noinspection PyUnboundLocalVariable,PyShadowingNames
    def load(self, datafile=None):
        if datafile is None:
            datafile = self.filename
        try:
            return pd.read_pickle(datafile)
        except FileNotFoundError:
            return pd.DataFrame()
        except ValueError:
            downgrade_pickle(datafile)
            return self.load(datafile)
        except Exception as e:
            log_error(e)

    def reset(self, confirm=False):
        self.history = pd.DataFrame()
        if confirm:
            self.save()

    # noinspection PyShadowingNames
    def update(self, assets: list = None, from_date=None, to_date=None):
        if assets is None:
            assets = NMData().assets
            if assets is None:
                assets = DEFAULT_COINS_IN_HISTORY_DATA
        from_date, to_date = adjust(from_date, to_date, pd.Timestamp(EXCHANGE_OPENING_DATE))
        if from_date == to_date:
            to_date = tz_remove_and_normalize('utc')
        symbols = [i.get('symbol') for i in self.binance_api.get_all_tickers()]
        for asset in tqdm(assets):
            if QUOTE_ASSET in asset:
                asset = asset.replace(QUOTE_ASSET, '')
            try:
                old_data = self.history_for(asset)
                last_date = old_data.index.max() if old_data.index.max() > from_date else from_date
            except (KeyError, TypeError):
                old_data = pd.DataFrame()
                last_date = from_date
            symbol = f'{asset}{QUOTE_ASSET}'
            if symbol not in symbols:
                symbol = f'{QUOTE_ASSET}{asset}'
            try:
                new_data = readable_kline(self.binance_api.get_historical_klines(symbol,
                                                                                 Client.KLINE_INTERVAL_1DAY,
                                                                                 last_date.strftime('%Y-%m-%d'),
                                                                                 to_date.strftime(
                                                                                         '%Y-%m-%d'))).set_index(
                        'Open time')
                new_data[SYMBOL] = asset
                new_data['Close time'] = pd.to_datetime(new_data['Close time'] * 10 ** 6)
                merged_data = old_data[~(old_data.index >= old_data.index.max())].append(new_data)
                try:
                    self.history = self.history[self.history[SYMBOL] != asset].append(merged_data)
                except KeyError:
                    self.history = self.history.append(merged_data)
            except ValueError:
                logging.info(f' No data for {symbol} from {from_date} to {to_date}.')
            except Exception as e:
                log_error(e)
        if SYMBOL in self.history.columns:
            self.history = self.history.sort_values(SYMBOL).sort_index()
            self.save()
        return self.history

    # noinspection PyShadowingNames
    def update_single_date(self, assets: list = None, date=None):
        if isinstance(assets, str):
            try:
                date = pd.Timestamp(assets)
                assets = None
            except ValueError:
                assets = [assets]
        if date is None:
            date = tz_remove_and_normalize('utc')
            if assets is None:
                assets = self.assets
        else:
            if assets is None and pd.Timestamp(date).normalize() != pd.Timestamp(self.history.index.max()).normalize():
                assets = [a for a in set(self.assets).difference(self.history[date: date][SYMBOL].unique())
                          if date > self.history_for(a).index.min()]
        if len(assets) > 0:
            try:
                return self.update(assets, from_date=date, to_date=date)
            except Exception as e:
                log_error(e)
        return pd.DataFrame()

    def save(self):
        safe_save(self.history, self.filename)

    def yield_for_coin(self, coin, from_date=None, to_date=None):
        self.need_update(coin, to_date)
        df = self.get(coin, from_date, to_date)
        return df.iloc[-1]['Close'] / df.iloc[0]['Close'] - 1

    def need_update(self, coin, to_date):
        if isinstance(coin, Iterable) and not isinstance(coin, str):
            for coin in tuple(coin):
                self.need_update(coin, to_date)
            return
        elif to_date is not None:
            df_for_coin = self.history[self.history[SYMBOL] == coin]
            if len(df_for_coin) < 1 or (tz_remove_and_normalize(to_date) - df_for_coin[CLOSE_TIME].max() >
                                        pd.Timedelta(1, 'day')):
                self.update([coin])

    def yield_for_coins(self, coins, from_date=None, to_date=None, return_df=False):
        if to_date is None:
            to_date = from_date
        self.need_update(coins, to_date)
        df = pd.DataFrame()
        for coin in coins:
            df.loc[coin, 'yield'] = self.yield_for_coin(coin, from_date, next_date(to_date))
            df.loc[coin, 'from'] = pd.Timestamp(from_date)
            df.loc[coin, 'to'] = pd.Timestamp(to_date) if to_date is not None else to_date
        if return_df:
            return df
        else:
            return df['yield'].mean()


class NMData:
    _nm_url: str

    def __init__(self, nm_url=None, load=True, datafile=None):
        if nm_url is None:
            try:
                from config import nm_url
            except (ImportError, ModuleNotFoundError):
                log_error("Invalid config.py file, 'nm_url' not specified!")
                self._nm_url = NM_REPORT_DEFAULT_URL
        self._nm_url = nm_url
        self._df = None
        self._coin_data = None
        self._ta_data = None
        self._filename = datafile
        self.subset = ['price'] + [f'NM{i}' for i in range(1, 5)]
        if load and datafile is not None:
            self._df = self.load(datafile)
        if tz_remove_and_normalize('utc') - self.last_update > pd.Timedelta(1, 'day'):
            self.get_nm_data()

    def __repr__(self):
        return f'<NMData container class at {hex(id(self))}:\n{self.df.__repr__()}' \
               f'\n\nLast update on {self.last_update}>'

    def __str__(self):
        return self.df.__str__()

    def __setitem__(self, key, value):
        if not hasattr(super(), key) and hasattr(self.df, key):
            setattr(self._df, key, value)
        else:
            super().__setattr__(key, value)

    def __getitem__(self, item):
        if item in self.df.keys():
            return self.df[item]
        else:
            raise KeyError

    @property
    def assets(self):
        if self.df is not None and len(self.df) > 0:
            return self.df.symbol.unique()
        else:
            return list()

    @property
    def coins(self):
        if self._coin_data is None:
            self._coin_data = CoinData()
        return self._coin_data

    @property
    def df(self):
        if self._df is None:
            self._df = self.load()
            if self._df is None:
                self._df = pd.DataFrame()
        return self._df

    @df.setter
    def df(self, value):
        self._df = value

    @property
    def filename(self):
        if self._filename is None:
            try:
                # noinspection PyUnresolvedReferences
                from config import nm_data_file as datafile
            except (ImportError, ModuleNotFoundError):
                # noinspection PyGlobalUndefined
                global NMDATA_FILE
                datafile = NMDATA_FILE
            self._filename = datafile
        return self._filename

    @filename.setter
    def filename(self, value):
        self._filename = value

    def import_from_excel(self, filename=NM_INFO_XLSX):
        imported_nm = pd.concat([df[['Data', 'NM1', 'NM2', 'NM3', 'NM4', 'Moeda']]
                                for df in (pd.read_excel(filename, f'NM{i + 1}') for i in tqdm(range(4)))]).rename(
                                {'Data': 'date', 'Moeda': 'symbol'}, axis='columns').drop_duplicates().reset_index(
                                drop=True)
        imported_nm.symbol = imported_nm.symbol.str.replace(QUOTE_ASSET, '')
        imported_nm.date = pd.to_datetime(imported_nm.date, errors='coerce')
        imported_nm = imported_nm[~imported_nm.date.isna()]
        imported_nm.index = pd.DatetimeIndex(
                imported_nm.date).tz_localize('utc').tz_convert(NM_TIME_ZONE).tz_localize(None)
        imported_nm = imported_nm.drop('date', axis=1)
        history = self.df
        if len(history) > 0:
            if 'date' in history.columns:
                history = history.set_index('date')[['symbol', 'NM1', 'NM2', 'NM3', 'NM4']]
            else:

                history = history[['symbol', 'NM1', 'NM2', 'NM3', 'NM4']]
            history = pd.concat([imported_nm, history]).drop_duplicates()
            history.index = pd.DatetimeIndex(history.index)
        else:
            history = imported_nm
        self.df = history
        return history

    # noinspection PyShadowingNames
    @property
    def last_update(self):
        try:
            if 'date' in self.df.columns:
                self.df = self.df.set_index('date')
            last_update = self.df.index.max()
            if not isinstance(last_update, pd.Timestamp):
                raise ValueError
        except ValueError:
            filename = pathlib.Path(self.filename)
            assert filename.exists(), f'No such file: {filename}'
            last_update = pd.Timestamp.fromtimestamp(filename.stat().st_mtime)
        except AssertionError as e:
            logging.debug(e)
            last_update = tz_remove_and_normalize(EXCHANGE_OPENING_DATE)
        return last_update

    # noinspection PyShadowingNames
    def load(self, datafile=None):
        if datafile is None:
            datafile = self.filename
        try:
            self._df = pd.read_pickle(datafile)
        except FileNotFoundError:
            pass
        except ValueError:
            downgrade_pickle(datafile)
            return self.load(datafile)
        except Exception as e:
            log_error(e)
        return self._df

    def save(self):
        if self.filename is None:
            self.filename = NMDATA_FILE
        safe_save(self.df, self.filename)

    def sort(self, by=None):
        if by is None:
            by = ['date', 'symbol']
            self.df = self.df.sort_values(by)
            return self.df

    @property
    def ta_data(self):
        if self._ta_data is None:
            self._ta_data = TAData()
        return self._ta_data

    def to_numeric(self):
        self.df = self.df.applymap(partial(pd.to_numeric, errors='ignore'))

    def get(self, nm_index=1, date='utc'):
        date = next_date(date, -1)
        df = self.df
        if 'date' in df.columns:
            df = df.set_index('date')
        retries = 3
        while retries > 0:
            retries -= 1
            try:
                nm_for_date = df.sort_index().loc[date.strftime('%Y%m%d')].groupby('symbol').last()
            except KeyError:
                nm_for_date = pd.DataFrame()
            if len(nm_for_date) < 1:
                if len(self.get_nm_data()) > 0:
                    continue
                else:
                    break
            else:
                return nm_for_date[f'NM{nm_index}'].sort_values(ascending=False)
        raise IndexError

    # noinspection PyShadowingNames
    def get_nm_data(self, url=None):
        if url is None:
            url = self._nm_url

        try:
            if 'date' in self.df.columns:
                self.df = self.df.set_index('date')
            max_date = self.df.index.max()
            if not isinstance(max_date, pd.Timestamp):
                raise ValueError
        except (AttributeError, KeyError, TypeError, ValueError):
            max_date = tz_remove_and_normalize(EXCHANGE_OPENING_DATE)
        df = pd.DataFrame()
        for i in range(1, NM_MAX + 1):
            try:
                nm_df = pd.read_html(url + str(i), decimal=',', thousands='.')[0]
                date = pd.to_datetime(nm_df.iloc[-1][0].replace(UPDATED_ON, '').replace(AT_SIGN, ' '), dayfirst=True)
                if date > max_date:
                    nm_df['date'] = date
                    nm_df.columns = NM_COLUMNS
                    nm_df = nm_df.drop(index=[0, 1, nm_df.index.max()], columns=['price'])
                    nm_df = nm_df.applymap(partial(pd.to_numeric, errors='ignore'))
                    df = df.append(nm_df)
                else:
                    break
            except Exception as e:
                log_error(e)
        if len(df) > 0:
            self.df = pd.concat([self.df, df.set_index('date')])
            self.df = self.df.drop_duplicates()
            if self.filename is not None and len(self.df) > 0:
                self.save()
        return self.df

    def drop_duplicates(self):
        self.df = self.df.drop_duplicates(subset=self.subset)

    def find_nm(self, nm_to_find: pd.DataFrame(), nm_index=1, top_n=4, acceptable_error=0.5 / 100):

        def find_nm(row):
            date = row.name
            yield_for_date = row['yield']
            coin_yields_for_date = self.ta_data.yield_for_date(date)
            list_n = top_n - 1
            valid_combinations = []
            while list_n > 0:
                coins = self.get(nm_index, next_date(date)).index
                if len(coins) < 1:
                    coins = discovered_nm.get(next_date(date), [])
                for required_coins in combinations(coins, list_n):
                    required_coins_yield = np.average([y for c, y in coin_yields_for_date.items()
                                                       if c in required_coins])
                    other_coins = coin_yields_for_date.index.unique().difference(required_coins)
                    min_yield = (top_n - list_n) * np.min([y for c, y in coin_yields_for_date.items()
                                                          if c in other_coins])
                    min_yield = (required_coins_yield * len(required_coins) + min_yield) / top_n
                    max_yield = (top_n - list_n) * np.max([y for c, y in coin_yields_for_date.items()
                                                           if c in other_coins])
                    max_yield = (required_coins_yield * len(required_coins) + max_yield) / top_n
                    if max_yield >= yield_for_date >= min_yield:
                        coin_yields_for_date = self.ta_data.yield_for_date(date)
                        for coins in set(combinations(other_coins, top_n - list_n)):
                            coins = set(required_coins).union(coins)
                            combined_yield = np.average([y for c, y in coin_yields_for_date.items() if c in coins])
                            if abs(combined_yield / yield_for_date - 1) <= acceptable_error:
                                # noinspection PyUnresolvedReferences
                                logging.info(
                                        f' {coins}, match: '
                                        f' {np.average([y for c, y in coin_yields_for_date.items() if c in coins])}'
                                        )
                                discovered_nm[date] = coins
                                valid_combinations.append(coins)
                        else:
                            if len(valid_combinations) > 0:
                                list_n = 0
                                break
                list_n -= 1

            return valid_combinations

        nm_to_find = data_frame_decimal_convert(nm_to_find).sort_index(ascending=False)
        discovered_nm = {}
        nm_to_find[f'suggested NM{nm_index}'] = nm_to_find.progress_apply(find_nm, axis=1)
        return nm_to_find

    def find_best_sharpe_range(self, nm_index, date, max_days=21):
        coin_list = self.get(nm_index, date).index
        nm_index_for_date = self.get(nm_index, date)[f'NM{nm_index}'].to_dict()
        test = pd.DataFrame()
        for day in tqdm(range(2, max_days)):
            for coin in coin_list:
                test.loc[coin, f'sharpe_{day}_days'] = self.ta_data.sharpe(coin, days_range=day,
                                                                           from_date=pd.Timestamp(date) -
                                                                           pd.Timedelta(day + 1, 'days')
                                                                           )[:date].iloc[-1]
        for column in test.columns:
            test.loc['matches', column] = sum([1 if test.index.values[i] == coin else 0 for i, coin in
                                              enumerate(test[column].sort_values(ascending=False).index)])
            test.loc['rms', column] = np.average([abs(test[column].get(coin, 0) - nm_index_for_date.get(coin, 0)
                                                      ) for coin in test.index.values if coin in coin_list])

        # if len(test) > 1:
        #     print(f'Best match: {test.T.matches.sort_values(ascending=False).index[0]}.')

        return test.T.rms.sort_values()

    def tech_data(self, nm_index, date=None, n=None):

        def add_ta(row):
            coin = row[SYMBOL]
            from_date = row[TO_DATE] - pd.Timedelta(100, 'days')
            to_date = next_date(row[TO_DATE])
            try:
                return self.coins.history.asof(to_date).add_ta(coin, from_date=from_date).drop(SYMBOL)
            except IndexError:
                return pd.DataFrame()

        if date is None:
            return self.coins.history[[TO_DATE, SYMBOL, f'NM{nm_index}']].iloc[:n].join(
                    self.coins.history[[TO_DATE, SYMBOL]].iloc[:n].progress_apply(add_ta, axis=1))
        else:
            history = self.coins.history.set_index('date')[date:date]
            if len(history) > 0:
                history = history.reset_index()

                return history[[SYMBOL, f'NM{nm_index}']].iloc[:n].join(
                        history[[TO_DATE, SYMBOL]].iloc[:n].progress_apply(add_ta, axis=1)).set_index(SYMBOL)
            else:
                return pd.DataFrame()

    def yield_for_date(self, nm_index, date, top_n=4):
        date = next_date(date, -1)
        coins = self.get(nm_index, date).index[:top_n]
        return self.coins.yield_for_coins(coins, from_date=date)


class Rebalance:
    def __init__(self, account_name, **kwargs):
        if isinstance(account_name, str):
            self.account = BinanceAccount(account_name)
        elif isinstance(account_name, dict):
            if 'portfolio' in account_name.keys():
                self.account = account_name.get('portfolio')
        elif isinstance(account_name, BinanceAccount):
            self.account = account_name
        else:
            self.account = BinanceAccount()
        self._loop = asyncio.get_event_loop()
        self.nm_data = NMData()
        self.fees = Fees()
        self.market_orders = True
        self.market_maker = False
        self.pending_orders = []
        self.subaccount = False
        self.top_n = 4
        self.maker_order_book_position = {}
        self.set_attributes_from_config(**kwargs)

    @property
    def nm_index(self):
        return self.account.index

    def create_orders(self, target=None, market_orders=None, refresh_mean_price=False, threshold=1/100, verbose=True,
                      maker_order=None):
        balance = self.account.balance.copy()
        if not self.subaccount:
            last_nm_coins = self.nm_data.get(self.nm_index, next_date(tz_remove_and_normalize('utc'), -1)
                                             ).index[:self.top_n]
            balance = balance[balance.index.isin(last_nm_coins)]
        if target is None:
            target = self.nm_data.get(self.nm_index).index[:self.top_n]
        target = self.trim_target(target)
        if market_orders is None:
            market_orders = self.market_orders
        if maker_order is None:
            maker_order = self.market_maker
        quote_asset_value_ = f'{QUOTE_ASSET} Value'
        quote_value = balance[quote_asset_value_].sum()
        assets = set(balance.index).union(target)
        new_balance = pd.DataFrame(balance, index=assets)
        new_balance['Target %'] = pd.Series(target) * 100
        new_balance = new_balance.fillna(0)
        new_balance['% diff'] = new_balance['Target %'] - new_balance['%']
        if abs(new_balance['% diff']).max() > threshold * 100:
            if refresh_mean_price or any(new_balance['Amount'] == 0):
                new_balance['Mean Price'] = pd.Series({coin: float(self.account.get_avg_price(
                    symbol=f'{coin}{QUOTE_ASSET}').get('price', 1)) if coin != QUOTE_ASSET else 1.0
                    for coin in tqdm(assets, desc='Retrieving average prices for assets')})
            else:
                new_balance['Mean Price'] = new_balance[f'{QUOTE_ASSET} Value'] / new_balance['Amount']
            new_balance['Order Size'] = quote_value * (new_balance['% diff'] / 100) / new_balance['Mean Price']
            new_balance['minNotional'] = new_balance.apply(lambda row: self.account.minimal_order(
                    f'{row.name}{QUOTE_ASSET}'), axis=1)
            while True:
                net_result = (new_balance['Order Size'] * new_balance['Mean Price']).sum()
                if net_result > 0:
                    buys = new_balance['Order Size'] > 0.01
                    logging.info(' Trimming orders to fit total account value!')
                    ratio = math.ceil(net_result / (new_balance.loc[buys, quote_asset_value_] *
                                                    new_balance.loc[buys, '% diff'] / 100).sum() * 1000) / 1000
                    new_balance.loc[buys, 'Order Size'] = new_balance.loc[buys, 'Order Size'] * (1 - ratio)
                else:
                    break
            if QUOTE_ASSET in new_balance.index:
                new_balance = new_balance.drop(QUOTE_ASSET)
            new_balance = new_balance[abs(new_balance['Order Size'] * new_balance['Mean Price'])
                                      > new_balance.minNotional]
            if len(new_balance) > 0:
                new_balance = new_balance.sort_values('Order Size')
                tqdm.pandas(desc='Testing proposed orders')
                new_balance['orders'] = new_balance.progress_apply(lambda row: self.order_trim(
                        f'{row.name}{QUOTE_ASSET}',
                        abs(row['Order Size']),
                        side=SIDE_BUY if row['Order Size'] > 0 else SIDE_SELL,
                        order_type=ORDER_TYPE_MARKET if market_orders else (
                            ORDER_TYPE_LIMIT_MAKER if maker_order else ORDER_TYPE_LIMIT),
                        price=self.price_for_amount(
                                symbol=f'{row.name}{QUOTE_ASSET}',
                                amount=abs(row['Order Size']),
                                side=SIDE_BUY if row['Order Size'] > 0 else SIDE_SELL
                                ) if maker_order else row['Mean Price'],
                        order_time=TIME_IN_FORCE_IOC),
                        axis=1)
                tqdm.pandas()
                if verbose:
                    new_balance[f'Target {QUOTE_ASSET} Value'] = (new_balance['Amount'] + new_balance['Order Size']
                                                                  ) * new_balance['Mean Price']
                    quote_value = new_balance[f'Target {QUOTE_ASSET} Value'].sum()
                    new_balance['Target %'] = new_balance[f'Target {QUOTE_ASSET} Value'] / quote_value * 100
                    print(new_balance[['Amount', '%', 'Mean Price', 'Order Size', 'Target %',
                          f'Target {QUOTE_ASSET} Value']])
                return new_balance['orders'].values
        logging.debug(' No orders required for rebalancing!')
        return []

    # noinspection PyShadowingNames
    def fit_market_order(self, market=f'BTC{QUOTE_ASSET}', quote_amount=None, side=SIDE_BUY, add_fee=True):
        # noinspection PyShadowingNames
        def return_dict(base_amount, avg_price):
            return dict(amount=base_amount, price=avg_price)

        if quote_amount is None:
            try:
                quote_amount = self.account.balance.get('Amount', {}).get(
                        self.account.get_symbol_info(market).get('quoteAsset'), 0.0)
            except AttributeError:
                quote_amount = 0.0

        if quote_amount > 0:
            try:
                book_orders = self.account.get_order_book(symbol=market).get('asks' if side == SIDE_BUY else 'bids')
            except Exception as e:
                log_error(e)
                return return_dict(0.0, 0.0)

            if len(book_orders) > 0:
                quote_amount_left = quote_amount
                base_amount = 0.0
                avg_price = 0.0
                fee = 0.0
                if add_fee:
                    fee = self.fees[market]
                for price, amount in book_orders:
                    amount = float(amount)
                    price = float(price)
                    if price * amount > quote_amount_left:
                        amount = quote_amount_left / price
                    quote_amount_left -= price * amount * (1 + fee)
                    avg_price = avg_price * base_amount + price * amount
                    base_amount += amount
                    avg_price /= base_amount
                    if not quote_amount_left > 0:
                        break
                return return_dict(base_amount, avg_price)
        else:
            return return_dict(0.0, 0.0)

    def mark_down_orders(self, orders):
        for order in orders:
            try:
                pair = order.get('symbol')
                amount = float(order.get('quantity'))
                side = order.get('side')
                maker = order.get('side') == ORDER_TYPE_LIMIT_MAKER
                order['price'] = str(
                    self.account.round_price(self.price_for_amount(pair, amount=amount, side=side, maker=maker), pair))
                order.pop('orderId')
                order.pop('status')
            except KeyError:
                pass
            except Exception as e:
                log_error(e)
        return orders

    def min_amount(self, symbol):
        self.account.minimal_order()

    def order_status(self, orderId):
        while True:
            try:
                order = {'status': o.get('status') for o in self.account.get_open_orders()
                         if o.get('orderId', -1) == orderId}
                break
            except BinanceAPIException as e:
                log_error(e)
        return order.get('status', 'CLOSED')

    # noinspection PyShadowingNames
    def order_trim(self,
                   pair,
                   amount,
                   price=0,
                   side=SIDE_SELL,
                   order_type=ORDER_TYPE_MARKET,
                   order_time=TIME_IN_FORCE_GTC,
                   validate=True
                   ):

        if side is None:
            side = SIDE_SELL
        if order_type is None:
            order_type = ORDER_TYPE_MARKET
        if order_time is None:
            order_time = TIME_IN_FORCE_GTC
        amt_str = str(self.account.round_amount(amount, pair))
        price_str = str(self.account.round_price(price, pair))
        order = dict(symbol=pair, side=side, type=order_type, quantity=amt_str)

        if order_type not in [ORDER_TYPE_MARKET]:
            order['price'] = price_str
        if order_type not in [ORDER_TYPE_MARKET, ORDER_TYPE_LIMIT_MAKER]:
            order['timeInForce'] = order_time

        if validate:
            # noinspection PyBroadException
            try:
                self.account.create_test_order(**order)
                order['validated'] = True
            except Exception as e:
                log_error(f' Invalid order: {order}, error: {e}')
                order['validated'] = False

        return order

    # noinspection PyShadowingNames
    def place_order(self, order, return_number_only=False, test_order=False):

        def valid_order_params(order):
            params = ['symbol',
                      'side',
                      'type',
                      'timeInForce',
                      'quantity',
                      'quoteOrderQty',
                      'price',
                      'newClientOrderId',
                      'icebergQty',
                      'newOrderRespType',
                      'recvWindow']

            if order['type'] != ORDER_TYPE_LIMIT:
                params.pop(params.index('timeInForce'))
                params.pop(params.index('icebergQty'))
            if order['type'] != ORDER_TYPE_MARKET:
                params.pop(params.index('quoteOrderQty'))
            else:
                params.pop(params.index('price'))
                if 'quoteOrderQty' in order.keys():
                    params.pop(params.index('quantity'))
            return params

        if test_order:
            min_notional = self.account.minimal_order(order.get('symbol'))
            order['quantity'] = self.account.round_amount((min_notional + 1) / float(order.get('price', 1)) +
                                                          float(self.account.step_size(order.get('symbol'))),
                                                          order['symbol'])
            if order['quantity'] < float(self.account.lotsize[order.get('symbol')].get('minQty', 0)):
                order['quantity'] = float(self.account.lotsize[order.get('symbol')]['minQty'])

        try:
            order.pop('validated')
            logging.debug(f'order keys: {order.keys()}')
            if isinstance(order['quantity'], float):
                order['quantity'] = str(self.account.round_amount(order['quantity'], order['symbol']))
            if isinstance(order['price'], float):
                order['price'] = str(self.account.round_price(order['price'], order['symbol']))
        except KeyError:
            pass
        try:
            status = self.account.create_order(**{k: v for k, v in order.items() if k in valid_order_params(order)})
        except BinanceAPIException as e:
            status = dict(orderId=e.code, status=e.message, symbol=order.get('symbol'))
        logging.debug(f'\n{status}\n')
        if return_number_only:
            return status.get('orderId', -1)
        else:
            return {k: v for k, v in status.items() if k in ('orderId', 'symbol', 'status')}

    async def _async_place_order(self, order_queue, timeout=5, open_orders_timeout=60):
        def reprice_maker_order(order):
            if order.pop('maker_position', -1) > 5:
                order['type'] = ORDER_TYPE_MARKET
                order.pop('price')
            else:
                self.maker_order_book_position[order['symbol']] = order.get('maker_position', -2) + 1
                order['price'] = self.price_for_amount(order['symbol'], amount=float(order['quantity']), maker=True)
                order['maker_position'] = self.maker_order_book_position.get(order['symbol'], -1)

        def reprice_limit_order(order):
            if order.get('limit_retries', 5) < 1:
                order['type'] = ORDER_TYPE_MARKET
                order.pop('limit_retries')
                order.pop('price')
            else:
                order['price'] = self.price_for_amount(order['symbol'], amount=float(order['quantity']))
                order['limit_retries'] = order.get('limit_retries', 5) - 1

        def trim_market_order(order):
            if order.get('limit_retries', 5) < 1:
                self.account.refresh_balance()
                order['quoteOrderQty'] = self.account.balance.loc[QUOTE_ASSET, 'Amount']
                order.pop('limit_retries')
            else:
                try:
                    order['quoteOrderQty'] = float(order['price']) * float(order['quantity'])
                except KeyError:
                    order['quoteOrderQty'] *= 95/100
                order['limit_retries'] = order.get('limit_retries', 5) - 1
            try:
                order.pop('price')
                order.pop('quantity')
            except KeyError:
                pass

        while not order_queue.empty():
            order = await order_queue.get()
            logging.info(f'Place order for {order.get("symbol").replace(QUOTE_ASSET, "")}')
            # noinspection PyPep8Naming
            status = self.place_order(order);            order.update(status)
            orderId = order.get('orderId', -1)
            if orderId < 1:
                logging.info(f'order failed: {order.get("status")}')
                if order.get('status').find('match and take') < 0 <= order.get('status').find('balance'):
                    while True:
                        logging.info(f'Waiting for orders to be fullfilled')
                        open_orders = [o for o in self.account.get_open_orders()]
                        open_order_ids = [o.get('orderId', -1) for o in open_orders]
                        self.pending_orders = [o for o in self.pending_orders if o in open_order_ids]
                        our_open_orders = [o for o in open_orders if open_order_ids in self.pending_orders]
                        for order in our_open_orders:
                            if (pd.Timestamp.now('utc').astimezone(None) - pd.Timestamp.utcfromtimestamp(
                                        order['time'] // 1000)).seconds > open_orders_timeout:
                                self.account.cancel_order(symbol=order['symbol'], orderId=order['orderId'])
                                logging.debug(f'Order {order} cancelled!')
                        if len(our_open_orders) > 0:
                            await asyncio.sleep(timeout)
                        else:
                            break
            else:
                self.pending_orders += [orderId]
                if order['type'] == ORDER_TYPE_LIMIT_MAKER:
                    reprice_maker_order(order)
                elif order['type'] == ORDER_TYPE_LIMIT:
                    reprice_limit_order(order)
                else:
                    trim_market_order(order)
                logging.debug(f'Replacing {order} in queue!')
                await order_queue.put(order)

    async def _async_place_orders(self, orders, workers=4):
        work_queue = asyncio.Queue()
        for order in orders:
            await work_queue.put(order)
        await asyncio.gather(*[asyncio.create_task(self._async_place_order(work_queue)) for w in range(workers)])

    def place_orders(self, orders):
        return self._loop.run_until_complete(self._async_place_orders(orders))

    def sequencial_place_orders(self, orders, retries=5, open_orders_timeout=60):
        orders_to_recycle = []
        for order in tqdm(orders, desc='placing orders'):
            self.account.refresh_balance()
            if order['side'] == SIDE_BUY:
                if order['type'] != ORDER_TYPE_LIMIT:
                    price = self.price_for_amount(order.get('symbol'), float(order.get('quantity')), SIDE_BUY,
                                                  maker=order['type'] == ORDER_TYPE_LIMIT_MAKER)
                else:
                    price = float(order['price'])
                max_price = self.account.balance.loc['USDT', 'Amount'] / float(order['quantity'])
                if price < max_price:
                    order.update(self.place_order(order))
            else:
                if self.account.balance.loc[order['symbol'].replace(QUOTE_ASSET, ''), 'Amount'] >= float(
                        order['quantity']):
                    order.update(self.place_order(order))
        else:
            for order in orders:
                if order.get('orderId', -1) < 1:
                    orders_to_recycle += [order]
        while True:
            open_orders = [o for o in self.account.get_open_orders() if o.get('orderId') in
                           [o.get('orderId', -1) for o in orders]+[-1]]
            if len(open_orders) > 1:
                for order in tqdm(open_orders, desc='Waiting for open orders to be filled'):
                    if (pd.Timestamp.now('utc').astimezone(None) - pd.Timestamp.utcfromtimestamp(
                            order['time'] // 1000)).seconds > open_orders_timeout:
                        orders_to_recycle += [{k if k != 'origQty' else 'quantity': v for k, v in order.items()
                                               if k in ('symbol', 'price', 'type', 'side', 'origQty', 'orderId',
                                               'status')}]
                        self.account.cancel_order(symbol=order['symbol'], orderId=order['orderId'])
                time.sleep(5)
            else:
                break
        orders_to_recycle = [o for o in self.refresh_order_status(orders_to_recycle) if o.get('status') != 'FILLED']
        if len(orders_to_recycle) > 0:
            self.place_orders(self.recycle_orders(orders_to_recycle), retries=retries)

    # def place_orders(self, orders):
    #     # noinspection PyShadowingNames
    # 
    #     # noinspection PyShadowingNames
    #     def status_for_id(order_id=None):
    #         status = {d.get('orderId', 0): d.get('status', 'UNKNOWN') for d in orders}
    #         if order_id is None:
    #             return status
    #         else:
    #             return status.get(order_id)
    # 
    #     for order in tqdm(orders, desc='placing orders'):
    #         if order['side'] == 'BUY' and len(orders) > 0:
    #             while True:
    #                 pending_orders = any([status_for_id(o['OrderId']) in (
    #                     ORDER_STATUS_NEW, ORDER_STATUS_PARTIALLY_FILLED, ORDER_STATUS_PENDING_CANCEL) for o in orders
    #                                          if o.get('OrderId', -1) > 0 and o.get('side') == SIDE_SELL])
    #                 if pending_orders:
    #                     time.sleep(30)
    #                     self.refresh_order_status(orders)
    #                 else:
    #                     break
    #             unfilled_orders = [o for o in orders if status_for_id(o.get('orderId', -1)) in (
    #                 ORDER_STATUS_CANCELED, ORDER_STATUS_EXPIRED, ORDER_STATUS_REJECTED, 'UNKNOWN')]
    #             if len(unfilled_orders) > 1:
    #                 self.mark_down_orders(unfilled_orders)
    #                 return self.place_orders(unfilled_orders)
    #         # noinspection PyPep8Naming
    #         status = self.place_order(order)
    #         order.update(status)
    #     else:
    #         self.refresh_order_status(orders)
    #         return all([s == ORDER_STATUS_FILLED for s in status_for_id().values()])

    def price_for_amount(self, symbol, amount=None, side=SIDE_BUY, maker=False):
        try:
            order_book = self.account.get_order_book(symbol=symbol)
            if maker:
                maker_price_index = iter((i for i in (-1, 0, 1, 2, 5, 10)
                                          if i > self.maker_order_book_position.get(symbol, -2)))
                while len(order_book) > 0:
                    i = next(maker_price_index)
                    if i < 0:
                        price = np.average([float(order_book['bids'][0][0]), float(order_book['bids'][0][0])])
                    else:
                        price = float(order_book['asks' if side == SIDE_SELL else 'bids'][i][0])
                    self.maker_order_book_position[symbol] = i
                    order_book = self.account.get_order_book(symbol=symbol)
                    best_bid = float(order_book['bids'][0][0])
                    best_ask = float(order_book['asks'][0][0])
                    if (side == SIDE_BUY and price < best_ask) or (side == SIDE_SELL and price > best_bid):
                        return price
            else:
                if amount is None:
                    amount = self.account.min_amount(symbol)
                orders = order_book['asks' if side == SIDE_BUY else 'bids']
                amount_to_fill = amount
                for order in orders:
                    amount_to_fill -= float(order[-1])
                    if amount_to_fill <= 0:
                        return float(order[0])
        except ValueError:
            log_error(f' No orders on order book for {symbol}.')
        except Exception as e:
            log_error(e)
        return 0.0

    def rebalance(self, orders=None, target=None):
        if target is None:
            target = []
        if orders is None:
            orders = self.create_orders(target)
        if len(orders) > 0:
            try:
                orders = sorted(orders, key=lambda row: (row['side'], float(row['quantity'])), reverse=True)
            except Exception as e:
                log_error(e)
            retries = 5
            while not self.place_orders(orders):
                self.refresh_order_status(orders)
                orders = [o for o in orders if o.get('status') != ORDER_STATUS_FILLED]
                self.recycle_orders(orders)
                if len(orders) < 1 or retries < 1:
                    break
                else:
                    retries -= 1

    def recycle_orders(self, orders):

        # noinspection PyShadowingNames
        def unknown_action(order):
            if order.get('status', 'UNKNOWN') != ORDER_STATUS_NEW:
                log_error(f'Not sure what to do with this order: {order}.')
                orders.pop(order)

        for index, order in enumerate(orders):
            try:
                if order['orderId'] < 0:
                    if order['status'] == 'Account has insufficient balance for requested action.':
                        order['quantity'] = str(self.account.round_amount(
                            self.account.balance.loc[QUOTE_ASSET, 'Amount'] / float(order['price']), order['symbol']))
                        continue
                if order['type'] == ORDER_TYPE_LIMIT_MAKER and 'take' in order['status']:
                    order['price'] = str(self.price_for_amount(order['symbol'], side=order['side'], maker=True))
                elif order['status'] in (ORDER_STATUS_EXPIRED, ORDER_STATUS_CANCELED):
                    order['price'] = str(self.price_for_amount(order['symbol'], amount=float(order['quantity']),
                                         side=order['side'], maker=False))
                else:
                    unknown_action(order)
                order.pop('status')
                order.pop('orderId')
                orders[index] = order
            except KeyError:
                order['orderId'] = -1
                order['status'] = 'UNKNOWN'
            except Exception as e:
                log_error(e)
        return orders

    def refresh_order_status(self, orders):
        for order in orders:
            try:
                status = order.get("status", 'UNKNOWN')
                order_id = order.get("orderId", -1)
                if order_id > 0 and status != ORDER_STATUS_FILLED:
                    try:
                        status = self.account.get_order(symbol=order.get("symbol"),
                                                        orderId=order.get("orderId", {status: 'UNKNOWN'}))["status"]
                    except BinanceAPIException as e:
                        if str(e).find('Order does not exist') > -1:
                            status = 'FILLED'
                        else:
                            log_error(e)
                    order['status'] = status
                logging.debug(f' Order # {order_id}: {status}')
            except Exception as e:
                log_error(e)
        return orders

    def set_attributes_from_config(self, **kwargs):
        attributes = self.account.config
        attributes.update(kwargs)
        for key, value in attributes.items():
            setattr(self, key, value)

    def trim_target(self, target):
        # TODO adjust target to simple list
        if isinstance(target, dict):
            if all([isinstance(k, int) for k in target.keys()]):
                # noinspection PyArgumentList
                target = pd.DataFrame({index: {coin: percentage / self.top_n
                                       for coin in self.nm_data.get(index).index[:self.top_n]}
                                       for index, percentage in target.items()}).T.sum().to_dict()
            else:
                total = sum(target.values())
                target = {k: v / total for k, v in target.items()}
        else:
            target = {k: 1/len(target) for k in target}
        return target


class Statement:
    def __init__(self, load=True, datafile=None):
        if load and datafile is None:
            try:
                # noinspection PyPep8Naming,PyShadowingNames
                from config import statement_file as STATEMENT_FILE
            except (ImportError, ModuleNotFoundError):
                # noinspection PyGlobalUndefined
                global STATEMENT_FILE
            datafile = STATEMENT_FILE
        self.daily_yield = self.load(datafile)
        self.filename = datafile
        self._binance_api = BinanceAccount()
        self.transactions = self.load(datafile)

    @property
    def binance_api(self):
        if self._binance_api is None:
            self._binance_api = BinanceAccount(connect=True)
        return self._binance_api

    # noinspection PyShadowingNames
    def load(self, datafile=None):
        if datafile is None:
            try:
                # noinspection PyPep8Naming,PyShadowingNames
                from config import statement_file as STATEMENT_FILE
            except (ImportError, ModuleNotFoundError):
                # noinspection PyGlobalUndefined
                global STATEMENT_FILE
                datafile = STATEMENT_FILE
        try:
            return pd.read_pickle(datafile)
        except FileNotFoundError:
            return pd.DataFrame()
        except ValueError:
            downgrade_pickle(datafile)
            return self.load(datafile)
        except Exception as e:
            log_error(e)

    # noinspection PyShadowingNames
    def statement(self, since=None):
        trades = []
        last_transaction = None
        symbols = [i.get('symbol') for i in self.binance_api.get_all_tickers()]
        for symbol in tqdm(symbols):
            try:
                if since is not None:
                    since = pd.Timestamp(since).value // 10 ** 6
                    trades += self._binance_api.get_my_trades(symbol=symbol, startTime=since)
                else:
                    try:
                        last_transaction = self.transactions[self.transactions.symbol == symbol].time.max(
                                ).value // 10 ** 6
                        trades += self.binance_api.get_my_trades(symbol=symbol, startTime=last_transaction)
                    except Exception as e:
                        logging.warning(e)
                        trades += self.binance_api.get_my_trades(symbol=symbol)
            except BinanceAPIException as e:
                log_error(e)
        # noinspection PyTypeChecker
        transactions = pd.DataFrame.from_dict(trades)
        for timestamp in ['time', 'updateTime']:
            if timestamp in transactions.columns:
                transactions[timestamp] = (transactions[timestamp] * 10 ** 6).apply(pd.Timestamp)
        transactions = transactions.applymap(partial(pd.to_numeric, errors='ignore'))
        dusts = pd.DataFrame()
        if since is None:
            since = last_transaction
        try:
            dusts = pd.DataFrame([i.get('logs')[0] for i in self.binance_api.get_dust_log(startTime=since)
                                 ['results']['rows'] if i.get('logs') is not None])
        except KeyError:
            # noinspection PyShadowingNames
            try:
                dusts = pd.DataFrame([i.get('logs')[0] for i in self.binance_api.get_dust_log()['results']
                                     ['rows'] if i.get('logs') is not None])
            except Exception as e:
                log_error(e)

        dusts = dusts.applymap(partial(pd.to_numeric, errors='ignore'))
        # noinspection PyTypeChecker
        dusts.operateTime = pd.to_datetime(dusts.operateTime).apply(pd.Timestamp)
        dusts['symbol'] = pd.Series([f'BNB{s}' if f'BNB{s}' in symbols else f'{s}BNB'
                                    for i, s in enumerate(dusts.fromAsset.values)])
        dusts = dusts.rename({'tranId': 'orderId', 'serviceChargeAmount': 'commission', 'amount': 'qty', 'operateTime':
                             'time', 'uid': 'id'}, axis='columns')
        dusts['isBuyer'] = dusts.apply(lambda row: False if row['fromAsset'] == row['symbol'][:len(row['fromAsset'])]
                                       else True, axis=1)
        dusts['isMaker'] = False
        dusts['isBestMatch'] = True
        dusts.loc[dusts.isBuyer, 'quoteQty'] = dusts.loc[dusts.isBuyer, 'qty']
        dusts.loc[dusts.isBuyer, 'qty'] = dusts.loc[dusts.isBuyer, 'transferredAmount'] + dusts.loc[
            dusts.isBuyer, 'commission']
        dusts.loc[dusts.isBuyer, 'price'] = dusts.loc[dusts.isBuyer, 'quoteQty'] / dusts.loc[dusts.isBuyer, 'qty']
        dusts.loc[~dusts.isBuyer, 'quoteQty'] = dusts.loc[~dusts.isBuyer, 'qty'] ** 2 / dusts.loc[
            ~dusts.isBuyer, 'transferredAmount']
        dusts.loc[~dusts.isBuyer, 'price'] = dusts.loc[~dusts.isBuyer, 'qty'] / dusts.loc[
            ~dusts.isBuyer, 'transferredAmount']
        dusts['commissionAsset'] = 'BNB'
        dusts['orderListId'] = -1
        transactions = pd.concat([transactions, dusts[transactions.columns]]).set_index('time').sort_index()
        self.transactions = self.transactions.append(transactions)


# noinspection PyShadowingNames
class TAData:
    def __init__(self, datafile=None, load=False):
        self._coins = None
        self._filename = datafile
        self._df = None
        if load:
            self._df = self.load(datafile)

    def __repr__(self):
        return self.df.__repr__()

    def __str__(self):
        return self.df.__str__()

    def __getattr__(self, attr):
        if attr[0] != '_' and not hasattr(super(), attr) and hasattr(self.df, attr):
            setattr(self, attr, getattr(self.df, attr))
            return getattr(self, attr)
        else:
            super().__getattribute__(attr)

    def __setitem__(self, key, value):
        if not hasattr(super(), key) and hasattr(self.df, key):
            setattr(self._df, key, value)
        else:
            super().__setattr__(key, value)

    def __getitem__(self, item):
        if item in self.df.keys():
            return self.df[item]
        else:
            raise KeyError

    @property
    def coins(self):
        if self._coins is None:
            self._coins = CoinData()
        return self._coins

    @property
    def filename(self):
        if self._filename is None:
            try:
                # noinspection PyPep8Naming,PyShadowingNames
                from config import ta_file as TA_DATA_FILE
            except (ImportError, ModuleNotFoundError):
                # noinspection PyGlobalUndefined
                global TA_DATA_FILE
            self._filename = TA_DATA_FILE
        return self._filename

    @filename.setter
    def filename(self, value):
        self._filename = value

    # noinspection PyShadowingNames
    @property
    def df(self):
        if self._df is None:
            self._df = self.load(self.filename)
        try:
            if (len(self._df) < 1 or
                    self._df.reset_index(level=0, drop=True).index[-1] < self.coins.history.index.unique()[-2]):
                self._df = self.add_tech_analysis(add_next_day_results=True)
                self.save()
        except Exception as e:
            log_error(e)
        return self._df

    @df.setter
    def df(self, value):
        self._df = value

    # noinspection PyUnboundLocalVariable,PyShadowingNames
    def load(self, datafile=None):
        if datafile is None:
            datafile = self.filename
        try:
            return pd.read_pickle(datafile)
        except FileNotFoundError:
            return pd.DataFrame()
        except ValueError:
            downgrade_pickle(datafile)
            return self.load(datafile)
        except Exception as e:
            log_error(e)

    def save(self):
        if self.filename is None:
            self.filename = NMDATA_FILE
        safe_save(self.df, self.filename)

    def daily_returns(self, coin):
        if hasattr(self._df, 'others_dr'):
            return self._df.loc[coin].others_dr
        else:
            return self.coins.history_for(coin)['Close'] / self.coins.history_for(coin)['Open'] - 1

    def sharpe(self, coin, days_range=30, from_date=None, risk_free_ratio=6 / 100, fillna=True):
        df = pd.DataFrame(self.daily_returns(coin)[from_date:] * 100, columns=['returns'])
        annualized = np.sqrt(365)
        df['sharpe'] = df.returns.rolling(days_range).apply(lambda x: (x.mean() - risk_free_ratio) / x.std() *
                                                            annualized, raw=True)
        if fillna:
            df.fillna(0, inplace=True)
        return df['sharpe']

    @staticmethod
    def add_ta(df, from_date=None, risk_free_ratio=6 / 100, days_range=10, fillna=True,
               add_next_day_results=True, pump_percentage=1.5, dump_percentage=-1.5):
        if not isinstance(df.index, pd.DatetimeIndex):
            df = df.set_index(OPEN_TIME)
        df = df[from_date:]
        if SYMBOL in df.columns:
            df = df.drop(SYMBOL, axis=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = ta.add_all_ta_features(df, open="Open", high="High", low="Low", close="Close",
                                        volume='Taker buy base asset volume', fillna=True)
            annualized = np.sqrt(365)
            df[f'sharpe_{days_range}days'] = df.others_dr.rolling(days_range).apply(
                    lambda x: (x.mean() - risk_free_ratio) / x.std() * annualized,
                    raw=True)
            if add_next_day_results:
                df['next_dr'] = df['others_dr'].shift(-1)
                df['next_dlr'] = df['others_dlr'].shift(-1)
                df['next_pump'] = df['next_dlr'] >= pump_percentage
                df['next_dump'] = df['next_dlr'] <= dump_percentage
        if fillna:
            df = df.fillna(0)
        return df

    def add_daily_yield_for_date(self, date, expected_yield_for_date=0):
        coin_yields_for_date = self.coins.history[next_date(date, -1):date]
        coin_yields_for_date = coin_yields_for_date.reset_index().sort_values([SYMBOL, OPEN_TIME])
        coin_yields_for_date[OPEN] = coin_yields_for_date[CLOSE].shift(1)
        coin_yields_for_date = coin_yields_for_date[coin_yields_for_date.set_index(OPEN_TIME).index == date]
        coin_yields_for_date[YIELD] = (coin_yields_for_date[CLOSE] / coin_yields_for_date[OPEN] - 1) * 100
        coin_yields_for_date[DIFF] = abs(coin_yields_for_date[YIELD] - expected_yield_for_date)
        coin_yields_for_date = coin_yields_for_date.sort_values(DIFF)
        coin_yields_for_date = coin_yields_for_date.set_index(SYMBOL)[YIELD]
        return coin_yields_for_date

    def add_tech_analysis(self, df=None, **kwargs):
        if df is None:
            df = self.coins.history
        return df.groupby(SYMBOL).progress_apply(partial(self.add_ta, **kwargs))

    def reset(self):
        self._df = None

    def ta_for_date(self, coin, from_date, to_date=None):
        return self.df.loc[coin][from_date:to_date]

    def yield_for_date(self, date=None, yield_indicator='others_dr'):
        df = self.df.reset_index(0, drop=True)[[SYMBOL, yield_indicator]].sort_index()[date:date]
        if date is None:
            return df
        else:
            return df.set_index(SYMBOL)
