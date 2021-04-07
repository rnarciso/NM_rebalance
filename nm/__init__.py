import ta
import json
import logging
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from functools import partial
# noinspection PyPackageRequirements
from binance.client import Client
from collections.abc import Collection
from itertools import permutations, combinations
# noinspection PyPackageRequirements
from binance.exceptions import BinanceAPIException
from nm.util import math, downgrade_pickle, next_date, readable_kline, safe_save, sum_dict_values, truncate, \
    tz_remove_and_normalize

AT_SIGN = ' às '
AVG_SLIPPAGE = 0.0045755
COIN_MARKET_COLUMNS = ['volume_24h', 'percent_change_1h', 'percent_change_24h', 'percent_change_7d', 'market_cap']
COIN_HISTORY_FILE = 'history.dat'
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
ORDER_AMOUNT_REDUCING_FACTOR = 5 / 100
QUOTE_ASSET = 'USDT'
RISK_FREE_DAILY_IRATE = 0.0001596535874
SINCE = '20191231'
SYMBOL = 'symbol'
STATEMENT_FILE = 'statement.dat'
UPDATED = 'atualizado'
UPDATED_ON: str = f'{UPDATED} em'
TOP_N_MAX = 4
YIELD_FILE = 'yield.dat'
# Following constants are imported from Client later on
SIDE_SELL, SIDE_BUY, TIME_IN_FORCE_GTC, ORDER_STATUS_FILLED, ORDER_TYPE_LIMIT, ORDER_TYPE_LIMIT_MAKER, \
    ORDER_TYPE_MARKET = [None]*7
# import constants from Client
for const in globals().copy().keys():
    if globals()[const] is None and Client.__dict__.get(const) is not None:
        globals()[const] = Client.__dict__[const]
tqdm.pandas()


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
    if tz_remove_and_normalize(from_date) < tz_remove_and_normalize(pd.Timestamp.now('utc')):
        if to_date is None:
            to_date = tz_remove_and_normalize(pd.Timestamp.now('utc'))
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


def convert(nmdf):
    nmdf[nmdf.columns[0]] = pd.to_datetime(nmdf[nmdf.columns[0]], dayfirst=True)
    nmdf[nmdf.columns[1]] = pd.to_numeric(nmdf[nmdf.columns[1]].str.replace('%', '').str.replace(',', '.'))
    return nmdf.rename({nmdf.columns[0]: 'date', nmdf.columns[1]: 'yield'}, axis='columns').set_index('date')


class Portfolio:

    _time_offset: int

    def __init__(self, keyname: str = None, connect=False, include_locked=False, config=None):
        if isinstance(keyname, dict):
            self._config = config
            keyname = config.get('account_name')
        elif isinstance(config, dict):
            self._config = config
        else:
            try:
                from config import accounts
                if keyname is not None:
                    for account in accounts:
                        if keyname == account.get('account_name'):
                            self._config = account
                            break
                else:
                    self._config = accounts[0]
            except Exception as e:
                logging.error(e)
                self._config = None
        self._balance = {}
        self._client = None
        self._fees = None
        self._include_locked_asset_in_balance = include_locked
        self._info = {}
        self._time_offset = 0
        self.min_notational = {}
        self.lot_size = {}
        self.connected = False
        if connect:
            self.connect(keyname)

    @property
    def time_offset(self):
        if self._time_offset == 0:
            time_offset = pd.Timestamp.now('utc') - pd.Timestamp(
                    self._client.get_server_time().get("serverTime") * 10 ** 6).tz_localize('utc')
            self._time_offset = math.ceil(abs(time_offset.value) / 10 ** 6) + MINIMUM_TIME_OFFSET
        return self._time_offset

    def __getattr__(self, attr):
        if attr[0] != '_' and not hasattr(super(), attr):
            if self._client is None:
                self.connect()
            if attr[0] != '_' and not hasattr(super(), attr) and hasattr(self._client, attr):
                if attr in ['get_trade_fee', 'get_account', 'create_order', 'get_open_orders',
                            'create_test_order', 'get_asset_balance', 'get_order', 'get_all_orders',
                            'get_my_trades', 'get_sub_account_list', 'transfer_dust', ]:
                    setattr(self, attr, partial(getattr(self._client, attr), recvWindow=self.time_offset))
                else:
                    setattr(self, attr, getattr(self._client, attr))
                try:
                    return getattr(self, attr)
                except BinanceAPIException as e:
                    if 'recvWindow' in e:
                        self._time_offset = 0
                        setattr(self, attr, partial(getattr(self._client, attr), recvWindow=self.time_offset))
                        try:
                            return getattr(self, attr)
                        except Exception as e:
                            logging.error(e)
                    else:
                        logging.error(e)
                        return

        super().__getattribute__(attr)

    @property
    def fees(self):
        if self._fees is None:
            try:
                self._fees = pd.DataFrame.from_dict(self.get_trade_fee()['tradeFee']).set_index('symbol')
            except Exception as e:
                logging.error(e)
                return pd.DataFrame()
        return self._fees

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
            logging.error(e)
        return df

    def connect(self, keyname: str = None):

        if self._config is None or keyname is not None:
            if keyname is None:
                keyname = 'binance'
            try:
                with open(KEYFILE, 'r') as file:
                    __keys__ = file.read()
                api_key = json.loads(__keys__)[f'{keyname}_key']
                api_secret = json.loads(__keys__)[f'{keyname}_api_secret']
            except FileNotFoundError:
                logging.error(' Key file not found!')
            except json.JSONDecodeError:
                logging.error(' Invalid Key file format!')
            except Exception as e:
                logging.error(e)
        else:
            api_key = self._config.get('api_key')
            api_secret = self._config.get('api_secret')
        try:
            logging.info('Connecting ')
            # noinspection PyUnboundLocalVariable
            self._client = Client(api_key, api_secret)
            self.connected = True
            return self._client
        except Exception as e:
            logging.error(e)

    def info(self, pair):
        if pair not in self._info.keys():
            self._info[pair] = self.get_symbol_info(pair)
        return self._info[pair]

    def minimal_order(self, pair):
        try:
            return self.min_notational[pair]
        except KeyError:
            try:
                self.min_notational[pair] = float(
                        [f['minNotional'] for f in
                            self.info(pair)['filters']
                            if f['filterType'] == 'MIN_NOTIONAL'][0])
                return self.min_notational[pair]
            except Exception as e:
                logging.debug(e)
                default_min = dict(BNB=0.1, BTC=1e-4, ETH=0.01, TRX=100)
                return default_min.get(pair[-4:], 10)

    def step_size(self, pair):
        try:
            return self.lot_size[pair]['stepSize']
        except KeyError:
            try:
                self.lot_size[pair] = {
                    k: v for f in self.info(pair)['filters'] if f['filterType'] == 'LOT_SIZE'
                    for k, v in f.items() if k != 'filterType'}
                return self.lot_size[pair]['stepSize']
            except Exception as e:
                logging.debug(e)
                return '0.00100000'

    def refresh_balance(self, client=None, keyname: str = None, include_locked: bool=None):
        if client is None:
            if self._client is None:
                client = self.connect(keyname)
            else:
                client = self._client
        try:
            assets = client.get_account()
        except BinanceAPIException:
            self._time_offset = 0
            _ = self.time_offset
            try:
                assets = self.get_account()
            except BinanceAPIException:
                logging.error(' Unable to retrieve balances from Binance!')
                return pd.Series(self._balance).sort_values()
        if include_locked is None:
            include_locked = self._include_locked_asset_in_balance
        if include_locked:
            self._balance = {a['asset']: float(a['locked']) + float(a['free']) for a in assets['balances'] if
                             float(a['locked']) > 0.0 or float(a['free']) > 0.0}
        else:
            self._balance = {a['asset']: float(a['free']) for a in assets['balances'] if float(a['free']) > 0.0}
        return pd.Series(self._balance).sort_values()

    def avg_price(self, amount, market, side=SIDE_BUY, add_fee=True):
        try:
            book_orders = self.get_order_book(symbol=market).get('asks' if side == SIDE_BUY else 'bids')
        except Exception as e:
            logging.error(e)
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
                fee = self.fees.loc[market]['taker']
                cost *= 1 + fee

            return dict(price=avg_price, quote_amount=cost)

    def convert_small_balances(self, base_asset='BNB'):
        balance = self.balance
        small_balances = [asset for asset in balance.index if balance.loc[asset, f'{QUOTE_ASSET} Value'] <
                          self.minimal_order(f'{asset}{QUOTE_ASSET}')]
        try:
            bnb_index = small_balances.index(base_asset)
            small_balances.pop(bnb_index)
        except ValueError:
            pass
        if len(small_balances) > 0:
            try:
                self.transfer_dust(asset=','.join(small_balances))
            except BinanceAPIException as e:
                logging.error(f'{e}. Assets: {small_balances}.')

    def fit_market_order(self, market=f'BTC{QUOTE_ASSET}', quote_amount=None, side=SIDE_BUY, add_fee=True):
        # noinspection PyShadowingNames
        def return_dict(base_amount, avg_price):
            return dict(amount=base_amount, price=avg_price)

        if quote_amount is None:
            try:
                quote_amount = self.balance.get('Amount', {}).get(self.get_symbol_info(market).get('quoteAsset'), 0.0)
            except AttributeError:
                quote_amount = 0.0

        if quote_amount > 0:
            try:
                book_orders = self.get_order_book(symbol=market).get('asks' if side == SIDE_BUY else 'bids')
            except Exception as e:
                logging.error(e)
                return return_dict(0.0, 0.0)

            if len(book_orders) > 0:
                quote_amount_left = quote_amount
                base_amount = 0.0
                avg_price = 0.0
                fee = 0.0
                if add_fee:
                    fee = self.fees.loc[market]['taker']
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

    def place_order(self, order, return_number_only=True):
        try:
            order.pop('validated')
        except KeyError:
            pass

        try:
            status = self.create_order(**order)
        except BinanceAPIException as e:
            status = dict(orderId=e.code, status=e.message, symbol=order.get('symbol'))

        if return_number_only:
            return status.get('orderId')
        else:
            return status

    def order_status(self, orderId):
        order = {}
        while True:
            try:
                order = {'status': o.get('status') for o in self.get_open_orders() if o.get('orderId', -1) == orderId}
                break
            except BinanceAPIException:
                self._time_offset = 0
                if self.time_offset < 1:
                    break
        return order.get('status', 'CLOSED')

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

        try:
            precision: float = float(self.step_size(pair))
        except ValueError:
            precision = 10 ** -5
        if not precision > 0:
            precision = 10 ** -5
        amt_str = truncate(amount, precision)
        price_str = truncate(price, precision)
        order = dict(symbol=pair, side=side, type=order_type, quantity=amt_str)

        if order_type not in [ORDER_TYPE_MARKET, ORDER_TYPE_LIMIT_MAKER]:
            order['timeInForce'] = order_time
            order['price'] = price_str

        if validate:
            # noinspection PyBroadException
            try:
                self.create_test_order(**order)
                order['validated'] = True
            except Exception as e:
                logging.info(f' Invalid order: {order}, error: {e}')
                order['validated'] = False

        return order

    def rebalanced_portfolio_proposal(self, target,
                                      quote_asset: str = QUOTE_ASSET,
                                      threshold: float = 2.0,
                                      fee_type: str = 'taker',
                                      market_order: bool = True,
                                      print_final_balance: bool = False,
                                      print_orders: bool = False):
        """
        :type print_orders: bool
        :param print_final_balance: bool
        :param print_orders: bool
        :param market_order: bool
        :param fee_type: str
        :type threshold: float
        :param quote_asset:
        :rtype: (list, dict)
        :type target: list of assets or dict with asset: percentage
        """
        if len(self.refresh_balance()) < 1:
            logging.error(' Nothing in your portfolio to rebalance!')
            return []

        if not isinstance(quote_asset, str) and isinstance(quote_asset, Collection):
            orders = {}
            for asset in quote_asset:
                orders[asset] = self.rebalanced_portfolio_proposal(target, asset, threshold, fee_type)
            return orders

        target = self.target_portfolio(quote_asset, target)

        assets = set(target.keys()).union(self._balance.keys()).union({quote_asset})

        pairs = ["".join(a) for a in permutations(assets, 2)]

        prices = pd.DataFrame.from_dict(self.get_orderbook_tickers())
        prices = prices[prices.symbol.isin(pairs)]
        prices = prices.set_index('symbol').astype('float', errors='ignore')

        pairs_for_each_asset: dict = {coin: {
            pair: 'bid' if pair.find(coin) == 0 else 'ask'
            for pair in prices.index.values if pair.find(coin) >= 0}
            for coin in assets}

        try:
            # noinspection PyUnresolvedReferences
            portfolio_value = self.balance[f'{quote_asset} Value'].sum()
        except (KeyError, IndexError):
            logging.error(f' Some assets cannot be traded directly to {quote_asset}.')
            return []

        if portfolio_value <= 0:
            logging.error(' Your current portfolio is worthless!')
            return []

        target = {k: v * portfolio_value / sum(
                target.values()) if k == quote_asset else v * portfolio_value / sum(
                target.values()) / prices.loc[f'{k}{quote_asset}', 'bidPrice']
            for k, v in target.items()
            }

        fees = self.fees
        fees = fees[fees.index.isin(prices.index)]

        orders = []

        previous_diff = np.inf

        balance = self._balance.copy()

        to_sell = pd.DataFrame()
        to_buy = pd.DataFrame()

        while True:
            diff = pd.DataFrame(pd.Series(
                    {c: target.get(c, 0) - balance.get(c, 0) for c in assets}
                    )).reset_index()
            diff.columns = ['coin', 'delta']

            # noinspection
            diff['quote_value'] = abs(diff.apply(lambda row: prices.loc[
                                    f"{row['coin']}{quote_asset}", 'bidPrice'
                                    if row['delta'] < 0 else
                                    'askPrice']
                                    * row['delta']
                                    if f"{row['coin']}{quote_asset}" in prices.index
                                    else (row['delta'] if quote_asset == row['coin'] else 0), axis=1))

            new_diff = diff['quote_value'].sum()

            if new_diff >= previous_diff or new_diff < portfolio_value * threshold / 100:
                break
            else:
                previous_diff = new_diff

            # noinspection PyUnresolvedReferences
            try:
                diff['fee'] = diff.apply(
                    lambda row: fees.loc[f"{row['coin']}{quote_asset}", fee_type]
                    * row['quote_value'] if f"{row['coin']}{quote_asset}" in prices.index else
                    0, axis=1)
            except KeyError:
                diff['fee'] = diff.apply(lambda row: 0.001 if row['coin'] != quote_asset else
                                         0, axis=1)

            diff['actual_value'] = diff['quote_value'] - diff['fee']

            diff['pair'] = (diff['coin'] + quote_asset).values

            valid = diff.apply(lambda row: row['pair'] in pairs_for_each_asset[row['coin']], axis=1)
            valid &= diff[valid].apply(lambda row: row['actual_value'] > self.minimal_order(row['pair']), axis=1)

            to_sell = diff[(diff['delta'] < 0) & valid].sort_values('quote_value', ascending=False)
            to_buy = diff[(diff['delta'] > 0) & valid].sort_values('quote_value', ascending=False)

            if not to_sell.empty:
                to_sell['delta'] *= -1
                if market_order:
                    sell_orders = pd.Series(
                        to_sell.apply(lambda row: self.order_trim(row['pair'], row['delta']), axis=1))
                    for index, order in sell_orders.items():
                        bids = self.get_order_book(symbol=order['symbol'])['bids']
                        amount_to_sell: float = float(order['quantity'])
                        amount_left = amount_to_sell
                        amount_sold = amount_to_sell
                        result = 0.0
                        for bid, amount in bids:
                            amount = float(amount)
                            result += float(bid) * (amount
                                                    if amount < amount_left else
                                                    amount_left)
                            if amount_left > amount:
                                amount_left -= amount
                            else:
                                break
                        else:
                            amount_sold = amount_to_sell - amount_left
                        to_sell.loc[index, 'predicted_result'] = result
                        to_sell.loc[index, 'sold'] = amount_sold
                else:
                    sell_orders = pd.Series(to_sell.apply(lambda row: self.order_trim(
                            row['pair'], row['delta'], price=prices.loc[
                            row['pair'], f"{pairs_for_each_asset[row['coin']][row['pair']]}Price"] *
                            (1 if fee_type == 'taker' else (1 - MAKER_PREMIUM)),
                            order_type=ORDER_TYPE_LIMIT if fee_type == 'taker'
                            else ORDER_TYPE_LIMIT_MAKER), axis=1))
                    to_sell['predicted_result'] = to_sell['actual_value']

                balance = sum_dict_values(balance, (to_sell.set_index('coin')['delta'] * -1).to_dict())
                orders += list(sell_orders.values)
                predicted_quote_balance = balance.get(quote_asset, 0.0) + to_sell['predicted_result'].sum()
            else:
                predicted_quote_balance = balance.get(quote_asset, 0.0)

            if not to_buy.empty:
                if predicted_quote_balance < to_buy.actual_value.sum():
                    amount_reduction = predicted_quote_balance / to_buy.actual_value.sum()
                    to_buy['delta'] *= amount_reduction
                    to_buy['quote_value'] = to_buy.apply(lambda row: row['delta'] * prices.loc[row['pair'],
                                                         f"{pairs_for_each_asset[row['coin']][row['pair']]}Price"],
                                                         axis=1)

                try:
                    to_buy['fee'] = to_buy.apply(lambda row: fees.loc[f"{row['coin']}{quote_asset}", fee_type] *
                                                 row['quote_value'] if f"{row['coin']}{quote_asset}" in prices.index
                                                 else 0, axis=1)
                except KeyError:
                    to_buy['fee'] = to_buy.apply(lambda row: 0.001 * row['quote_value']
                                                 if row['coin'] != quote_asset else 0, axis=1)

                to_buy['actual_value'] = to_buy['quote_value'] - to_buy['fee']
                to_buy['delta'] *= to_buy['actual_value'] / to_buy['quote_value']

                if market_order:
                    # noinspection
                    buy_orders = pd.Series(to_buy.apply(lambda row: self.order_trim(row['pair'], row['delta'],
                                           side=SIDE_BUY, order_type=ORDER_TYPE_MARKET), axis=1))
                else:
                    # noinspection
                    buy_orders = pd.Series(to_buy.apply(lambda row: self.order_trim(
                            row['pair'], row['delta'], price=prices.loc[row['pair'], f"askPrice"] *
                            (1 if fee_type == 'taker' else (1 - MAKER_PREMIUM)),
                            side=SIDE_BUY, order_type=ORDER_TYPE_LIMIT if fee_type == 'taker'
                            else ORDER_TYPE_LIMIT_MAKER), axis=1))
                    to_buy['predicted_result'] = to_buy['actual_value']
                balance[quote_asset] = predicted_quote_balance
                balance = sum_dict_values(balance, to_buy.set_index('coin')['delta'].to_dict())
                orders += list(buy_orders.values)
            elif to_sell.empty:
                break

        if print_orders:
            print(f'Sell orders:\n{to_sell}\n\nBuy orders:\n{to_buy}')

        if print_final_balance:
            balance = pd.Series(balance, name='amount').sort_values()
            print(balance[balance > 0])
        return orders

    @staticmethod
    def target_portfolio(quote_asset, target):
        if type(target) is dict:
            target_total = sum([v for v in target.values() if v > 0])
            if target_total > 0:
                for coin in target.keys():
                    if target[coin] > 0 and target != 100:
                        target[coin] = target[coin] / target_total * 100
            else:
                target = dict(zip(target.keys(), [1 / len(target) * 100] * len(target)))
        elif isinstance(target, Collection):
            target = dict(zip(target, [1 / len(target) * 100] * len(target)))
            target = dict(zip((lambda ps: [p[0] for p in ps if len(p) > 0])(
                              [[q for q in k.split(quote_asset) if len(q) > 0] for k in target.keys()]),
                              target.values()))
        return target

    def rebalance(self, orders_list=None, **kwargs):
        if orders_list is None:
            orders_list = self.rebalanced_portfolio_proposal(
                    kwargs.pop('target', []), **kwargs)
        if len(orders_list) > 0:
            order_numbers = []
            for order in orders_list:
                # noinspection PyPep8Naming
                orderId = self.place_order(order)
                if orderId < 0:
                    market = order.get('symbol')
                    amount = self.fit_market_order(market).get('amount', 0.0)
                    response = {}
                    while True:
                        try:
                            response = self.place_order(self.order_trim(market, amount,
                                                        side=order.get('side', SIDE_BUY),
                                                        order_type=ORDER_TYPE_MARKET),
                                                        False)
                            break
                        except BinanceAPIException:
                            self._time_offset = 0
                            if self.time_offset < 1:
                                break
                    # noinspection PyPep8Naming
                    orderId = response.get('orderId', -1)
                    if orderId < 0:
                        logging.error(f' Invalid order # {orderId} for pair {market}: {response.get("status")}')
                        continue
                order_numbers += [orderId]
            for orderId in order_numbers:
                logging.info(f'Order # {orderId}: {self.order_status(orderId)}')


class Deposits:
    def __init__(self):
        self._data = pd.DataFrame()
        self._symbols = None
        self._binance_api = None

    def __repr__(self):
        return self._data.__repr__()

    def __str__(self):
        return self._data.__str__()

    @property
    def binance_api(self):
        if self._binance_api is None:
            self._binance_api = Portfolio(connect=True)
        return self._binance_api

    @property
    def df(self):
        return self._data

    @property
    def symbols(self):
        if self._symbols is None:
            self._symbols = [i.get('symbol') for i in self.binance_api.get_all_tickers()]
        return self._symbols

    def add(self, date, value, currency=QUOTE_ASSET, nm_index=1):
        date = pd.Timestamp(date)
        self._data.loc[date, 'Amount'] = value
        self._data.loc[date, 'NM'] = str(nm_index)
        self._data.loc[date, 'Currency'] = currency
        if currency == QUOTE_ASSET:
            self._data.loc[date, 'Quote Value'] = value
        elif f'{QUOTE_ASSET}{currency}' in self.symbols:
            market_data = readable_kline(self.binance_api.get_historical_klines(f'{QUOTE_ASSET}{currency}',
                                                                                Client.KLINE_INTERVAL_1DAY,
                                                                                *[date.strftime('%Y-%m-%d')]*2))
            self._data.loc[date, 'Quote Value'] = value / market_data['High'].iloc[0]
        else:
            try:
                market_data = self.binance_api.get_historical_klines(f'{currency}{QUOTE_ASSET}',
                                                                     Client.KLINE_INTERVAL_1DAY,
                                                                     *[date.strftime('%Y-%m-%d')] * 2)
                self._data.loc[date, 'Quote Value'] = value * market_data['Low'].iloc[0]
            except BinanceAPIException as e:
                logging.error(e)
        return self

    def deposit(self, *args, **kwargs):
        return self.add(*args, **kwargs)

    def reset(self):
        self._data = pd.DataFrame()


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
            self._binance_api = Portfolio(connect=True)
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
                global COIN_HISTORY_FILE
            self._filename = COIN_HISTORY_FILE
        return self._filename

    def get(self, coin, from_date=None, to_date=None):
        try:
            history = self.history[self.history['Asset'] == coin]
        except KeyError:
            return pd.DataFrame()
        return history[from_date:to_date]

    def history_for(self, coin):
        return self.history[self.history['Asset'] == coin]

    @property
    def history(self):
        if self._history is None:
            self._history = self.load(self._filename)
        return self._history

    @history.setter
    def history(self, value):
        self._history = value

    # noinspection PyUnboundLocalVariable
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
            logging.error(e)

    def reset(self, confirm=False):
        self.history = pd.DataFrame()
        if confirm:
            self.save()

    def update(self, assets: list=None, from_date=None, to_date=None):
        if assets is None:
            assets = NMData().assets
        from_date, to_date = adjust(from_date, to_date, pd.Timestamp(EXCHANGE_OPENING_DATE))
        if from_date == to_date:
            to_date = tz_remove_and_normalize(pd.Timestamp.now('utc'))
        symbols = [i.get('symbol') for i in self.binance_api.get_all_tickers()]
        for asset in tqdm(assets):
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
                                          Client.KLINE_INTERVAL_1DAY, last_date.strftime('%Y-%m-%d'),
                                          to_date.strftime('%Y-%m-%d'))).set_index('Open time')
                new_data['Asset'] = asset
                new_data['Close time'] = pd.to_datetime(new_data['Close time'] * 10 ** 6)
                merged_data = old_data[~(old_data.index >= old_data.index.max())].append(new_data)
                try:
                    self.history = self.history[self.history['Asset'] != asset].append(merged_data)
                except KeyError:
                    self.history = self.history.append(merged_data)
            except ValueError:
                logging.info(f'No data for {symbol} from {from_date} to {to_date}.')
            except Exception as e:
                logging.error(e)
        self.history = self.history.sort_values('Asset').sort_index()
        self.save()
        return self.history

    def update_single_date(self, assets: list=None, date=None):
        if isinstance(assets, str):
            try:
                date = pd.Timestamp(assets)
                assets = None
            except ValueError:
                assets = [assets]
        if date is None:
            date = tz_remove_and_normalize(pd.Timestamp.now('utc'))
            if assets is None:
                assets = self.assets
        else:
            if assets is None and pd.Timestamp(date).normalize() != pd.Timestamp(self.history.index.max()).normalize():
                assets = [a for a in set(self.assets).difference(self.history[date: date]['Asset'].unique())
                          if date > self.history_for(a).index.min()]
        if len(assets) > 0:
            try:
                return self.update(assets, from_date=date, to_date=date)
            except Exception as e:
                logging.error(e)
        return pd.DataFrame()

    def save(self):
        safe_save(self.history, self.filename)

    def yield_for_coin(self, coin, from_date=None, to_date=None):
        if to_date is not None and pd.Timestamp(to_date) > self.history.index.max():
            self.update([coin])
        df = self.get(coin, from_date, to_date)
        return df.iloc[-1]['Close']/df.iloc[0]['Close'] - 1

    def yield_for_coins(self, coins, from_date=None, to_date=None, return_df=False):
        if to_date is None:
            to_date = from_date
        if to_date is not None and pd.Timestamp(to_date) > self.history.index.max():
            self.update(coins)
        df = pd.DataFrame()
        for coin in coins:
            df.loc[coin, 'yield'] = self.yield_for_coin(coin, next_date(from_date, -1), to_date)
            df.loc[coin, 'from'] = pd.Timestamp(from_date)
            df.loc[coin, 'to'] = pd.Timestamp(to_date) if to_date is not None else to_date
        if return_df:
            return df
        else:
            return df['yield'].mean()

    def daily_returns(self, coin):
        return self.history_for(coin)['Close'] / self.history_for(coin)['Open'] - 1

    def sharpe(self, coin, days_range=30, from_date=None, risk_free_ratio=6/100):
        df = pd.DataFrame(self.daily_returns(coin) * 100, columns=['returns'])[from_date:]
        annualized = np.sqrt(365)
        df['sharpe'] = df.returns.rolling(days_range).apply(lambda x: (x.mean() - risk_free_ratio) / x.std() *
                                                            annualized, raw=True)
        df.fillna(0, inplace=True)
        return df['sharpe']

    def ta(self, coin, from_date=None, risk_free_ratio=6/100, days_range=10):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = ta.add_all_ta_features(
                    self.history_for(coin)[from_date:], open="Open", high="High", low="Low", close="Close",
                    volume='Taker buy base asset volume', fillna=True)
            annualized = np.sqrt(365)
            df['sharpe'] = df.others_dr.rolling(days_range).apply(lambda x: (x.mean() - risk_free_ratio) /
                                                x.std() * annualized, raw=True).fillna(0)
        return df

    def yield_for_date(self, date, yield_for_date=0):
        coin_yields_for_date = self.history[next_date(date, -1):date]
        coin_yields_for_date = coin_yields_for_date.reset_index().sort_values(['Asset', 'Open time'])
        coin_yields_for_date['Open'] = coin_yields_for_date['Close'].shift(1)
        coin_yields_for_date = coin_yields_for_date[coin_yields_for_date.set_index('Open time').index == date]
        coin_yields_for_date['yield'] = (coin_yields_for_date['Close'] / coin_yields_for_date['Open'] - 1) * 100
        coin_yields_for_date['diff'] = abs(coin_yields_for_date['yield'] - yield_for_date)
        coin_yields_for_date = coin_yields_for_date.sort_values('diff')
        coin_yields_for_date = coin_yields_for_date.set_index('Asset')['yield']
        return coin_yields_for_date


class Backtest:
    def __init__(self, advisor=None):
        if advisor is None:
            advisor = NMData()
        self.advisor = advisor
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
            self._binance_api = Portfolio(connect=True)
        return self._binance_api

    @property
    def coin_data(self):
        if self._coin_data is None:
            self._coin_data = CoinData()
        return self._coin_data

    def index_yield_for_date(self, nm_index, date, top_n=4):
        try:
            return self.coin_data.yield_for_coins(self.advisor.get(nm_index, date)[:top_n].index,
                                                  next_date(date, -1), date)['yield'].mean()
        except KeyError:
            return 0

    def nm_index_yield_for_period(self, nm_index, from_date, to_date=None, top_n=4, fees=False, slippage=None,
                                  interval=1, return_df=False):
        fee_database = pd.DataFrame()
        df = pd.DataFrame()
        last_coins = []
        from_date = tz_remove_and_normalize(from_date)
        if to_date is None:
            to_date = tz_remove_and_normalize(pd.Timestamp.now('utc'))
        elif isinstance(to_date, int):
            to_date = next_date(from_date, to_date)
        accrued_yield = 0
        if fees:
            fee_database = self.binance_api.fees.reset_index()
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
                    fee_for_date = fee_database[
                        fee_database[SYMBOL].isin([f'{c}{QUOTE_ASSET}' for c in set(coins).difference(last_coins)])]
                except NameError:
                    fee_for_date = fee_database[fee_database.symbol.isin([f'{c}{QUOTE_ASSET}' for c in coins])]
                fee_for_date = np.average(fee_database.taker) * 2 if date > from_date else 1
                if slippage is not None:
                    fee_for_date *= (1 + slippage)
                yield_for_period *= (1 - fee_for_date)
                last_coins = coins

            accrued_yield += yield_for_period
            if return_df:
                df.loc[date, f'NM index {nm_index} /{interval}D'] = yield_for_period
                df.loc[date, f'NM index {nm_index} ACUM.'] = accrued_yield

        if return_df:
            return df
        else:
            return accrued_yield

    def account_yield_for_period(self, accounts, from_date, *kwargs):
        if isinstance(accounts, dict):
            accounts = [accounts]
        return np.average([self.nm_index_yield_for_period(account.get('index'),
                          from_date, *kwargs) for account in accounts])

    def yield_simulation(self, contributions: Deposits, to_date=None, fees=True, slippage=AVG_SLIPPAGE, top_n=4,
                         fee_type='taker'):
        # noinspection PyShadowingNames
        def coin_yield(row):
            coin_name = row.name
            date = row.date
            try:
                coin_data = readable_kline(self.binance_api.get_historical_klines(f'{coin_name}{QUOTE_ASSET}',
                                           Client.KLINE_INTERVAL_1DAY, date.strftime('%Y-%m-%d'),
                                           date.strftime('%Y-%m-%d'))).iloc[0]
                return coin_data['Close'] / coin_data['Open'] - 1
            except Exception as e:
                logging.error(e)
        dfs = {}
        for nm in sorted(contributions.df.NM.unique()):
            dfs[nm] = pd.DataFrame()
            min_date = contributions.df[contributions.df.NM == nm].index.min()
            if to_date is None:
                to_date = pd.Timestamp.now('utc').normalize().tz_localize(None)
            else:
                to_date = pd.Timestamp(to_date)
            value = 0
            last_set_of_coins = set()
            nmdf = dfs[nm]
            coin_data = CoinData()
            for date in tqdm(pd.date_range(min_date, to_date)):
                nmdf.loc[date, 'open'] = value
                if date in contributions.df[contributions.df['NM'] == nm].index:
                    nmdf.loc[date, 'contribution'] = contributions.df.loc[date, 'Quote Value']
                    value += contributions.df.loc[date, 'Quote Value']
                else:
                    nmdf.loc[date, 'contribution'] = 0
                # noinspection PyBroadException
                try:
                    coins_for_date = self.advisor.get(nm, date)[:top_n]
                    coins_for_date['date'] = date
                    try:
                        nmdf.loc[date, f'NM{nm} yield'] = coin_data.yield_for_coins(coins_for_date.index,
                                                                                    next_date(date, -1),
                                                                                    date)['yield'].mean()
                    except KeyError:
                        coins_for_date['yield'] = coins_for_date.apply(coin_yield, axis=1)
                        nmdf.loc[date, f'NM{nm} yield'] = coins_for_date['yield'].mean()

                    nmdf.loc[date, f'NM{nm} coins'] = ','.join(coins_for_date.index)
                    value = value * (1 + nmdf.loc[date, f'NM{nm} yield'])
                    nmdf.loc[date, 'close'] = value
                    new_coins = set(coins_for_date.index).difference(last_set_of_coins)
                    if fees and len(new_coins) > 0:
                        value *= (1 - self.binance_api.fees[self.binance_api.fees.index.isin(
                                  [f'{coin_name}{QUOTE_ASSET}' for coin_name in new_coins])][fee_type].mean()
                                  * (2 if date > min_date else 1) * (len(new_coins)/top_n))
                    if slippage is not None and len(new_coins) > 0:
                        value *= (1 - slippage * (2 if date > min_date else 1) * (len(new_coins)/top_n))
                    last_set_of_coins = set(coins_for_date.index)
                    nmdf.loc[date, 'adjusted close'] = value
                except Exception as e:
                    logging.error(f'\n{e}, while processing data for {date.date()}.')

        consolidated_df = pd.concat(dfs.values()).reset_index().groupby('index').sum()
        consolidated_df.index.name = None
        return consolidated_df


class NMData:

    _nm_url: str

    def __init__(self, nm_url=None, load=True, datafile=None):
        if nm_url is None:
            try:
                from config import nm_url
            except (ImportError, ModuleNotFoundError):
                logging.error("Invalid config.py file, 'nm_url' not specified!")
                self._nm_url = NM_REPORT_DEFAULT_URL
        self._nm_url = nm_url
        self._history = None
        self._coin_data = None
        if datafile is None:
            try:
                from config import nm_data_file as datafile
            except (ImportError, ModuleNotFoundError):
                datafile = NMDATA_FILE
        if datafile is not None and len(datafile) < 1:
            datafile = None
        self.filename = datafile
        self.subset = ['price']+[f'NM{i}' for i in range(1, 5)]
        if load and datafile is not None:
            self.history = self.load(datafile)

    def __repr__(self):
        return f'<NMData container class at {hex(id(self))}:\n{self.history.__repr__()}' \
               f'\n\nLast update on {self.last_update}>'

    def __str__(self):
        return self.history.__str__()

    @property
    def assets(self):
        if self.history is None or len(self.history) > 0:
            return self.history.symbol.unique()
        else:
            return list()

    @property
    def coins(self):
        if self._coin_data is None:
            self._coin_data = CoinData()
        return self._coin_data

    @property
    def history(self):
        if self._history is None:
            self._history = self.load()
        return self._history

    @history.setter
    def history(self, value):
        self._history = value

    @property
    def last_update(self):
        if self.history is not None and 'date' in self.history.columns:
            return self.history.date.max().tz_localize(NM_TIME_ZONE)
        else:
            return pd.Timestamp(EXCHANGE_OPENING_DATE)

    def load(self, datafile=None):
        if datafile is None:
            if self.filename is None:
                self.filename = NMDATA_FILE
            datafile = self.filename
        try:
            self._history = pd.read_pickle(datafile)
        except FileNotFoundError:
            pass
        except ValueError:
            downgrade_pickle(datafile)
            return self.load(datafile)
        except Exception as e:
            logging.error(e)
        return self._history

    def save(self):
        if self.filename is None:
            self.filename = NMDATA_FILE
        safe_save(self.history, self.filename)

    def sort(self, by=None):
        if by is None:
            by = ['date', 'symbol']
            self.history = self.history.sort_values(by)
            return self.history

    def to_numeric(self):
        self.history = self.history.applymap(partial(pd.to_numeric, errors='ignore'))

    def get(self, index=1, date='now', include_price=True):
        columns = ['symbol', f'NM{index}', 'price']
        df: pd.DataFrame = self.history
        if df is not None and 'date' in df.columns:
            df.index = pd.DatetimeIndex(pd.to_datetime(df.date)).tz_localize(NM_TIME_ZONE
                                                                             ).tz_convert('UTC').to_series()
            df = df.loc[pd.Timestamp(date).tz_localize(NM_TIME_ZONE).tz_convert('UTC').normalize():pd.Timestamp(date)
                                                                    .tz_localize(NM_TIME_ZONE).tz_convert('UTC')]
            df = df.drop_duplicates(subset=['symbol'], keep='last')
        else:
            df = pd.DataFrame(columns=columns, index=[None]*TOP_N_MAX)
        if include_price:
            df = df[['symbol', f'NM{index}', 'price']].sort_values(f'NM{index}', ascending=False
                                                                   ).set_index('symbol')
        else:
            df = df[['symbol', f'NM{index}']].sort_values(f'NM{index}', ascending=False).set_index('symbol')

        return df

    def get_nm_data(self, url=None):
        if url is None:
            url = self._nm_url
        if self.history is None:
            df = pd.DataFrame()
        else:
            df = self.history

        for i in range(1, 5):
            mndf = pd.read_html(url+str(i), decimal=',', thousands='.')[0]
            mndf['date'] = pd.to_datetime(mndf.iloc[-1][0].replace(UPDATED_ON, '').replace(AT_SIGN, ' '), dayfirst=True)
            mndf.columns = NM_COLUMNS
            mndf = mndf.drop(index=[0, 1, mndf.index.max()])
            df = df.append(mndf)
        df.index = pd.RangeIndex(len(df))
        self.history = df
        self.to_numeric()
        self.drop_duplicates()
        if self.filename is not None and len(self.history) > 0:
            self.save()
        return self.history

    def drop_duplicates(self):
        self.history = self.history.drop_duplicates(subset=self.subset)

    def find_nm(self, nm_to_find: pd.DataFrame(), nm_index=1, top_n=4, acceptable_error=0.5/100):

        def find_nm(row):
            date = row.name
            yield_for_date = row['yield']
            coin_yields_for_date = self.coins.yield_for_date(date, yield_for_date)
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
                    min_yield = (required_coins_yield*len(required_coins) + min_yield) / top_n
                    max_yield = (top_n - list_n) * np.max([y for c, y in coin_yields_for_date.items()
                                                          if c in other_coins])
                    max_yield = (required_coins_yield * len(required_coins) + max_yield) / top_n
                    if max_yield >= yield_for_date >= min_yield:
                        coin_yields_for_date = self.coins.yield_for_date(date, (yield_for_date * top_n -
                                                                         required_coins_yield * list_n) /
                                                                         (top_n - list_n))
                        # min_yield = ((yield_for_date*top_n - required_coins_yield * list_n ) * (1 + acceptable_error)
                        #              ) / (top_n - list_n)
                        # max_yield = ((yield_for_date * top_n - required_coins_yield * list_n) * (1 + acceptable_error)
                        #              ) * (top_n - list_n)
                        # if min_yield > max_yield:
                        #     tmp_yield = max_yield
                        #     max_yield = min_yield
                        #     min_yield = tmp_yield
                        # other_coins = coin_yields_for_date[coin_yields_for_date.index.isin(other_coins) & (
                        #               coin_yields_for_date >= min_yield) & (coin_yields_for_date <= max_yield)].index
                        for coins in set(combinations(other_coins, top_n - list_n)):
                            coins = set(required_coins).union(coins)
                            combined_yield = np.average([y for c, y in coin_yields_for_date.items() if c in coins])
                            if abs(combined_yield / yield_for_date - 1) <= acceptable_error:
                                logging.info(
                            f'{coins}, match: {np.average([y for c, y in coin_yields_for_date.items() if c in coins])}')
                                discovered_nm[date] = coins
                                valid_combinations.append(coins)
                        else:
                            if len(valid_combinations) > 0:
                                list_n = 0
                                break
                list_n -= 1

            return valid_combinations

        nm_to_find = convert(nm_to_find).sort_index(ascending=False)
        discovered_nm = {}
        nm_to_find[f'suggested NM{nm_index}'] = nm_to_find.progress_apply(find_nm, axis=1)
        return nm_to_find

    def find_best_sharpe_range(self, nm_index, date, max_days=21):
        coin_list = self.get(nm_index, date).index
        nm_index_for_date = self.get(nm_index, date)[f'NM{nm_index}'].to_dict()
        test = pd.DataFrame()
        for day in tqdm(range(2, max_days)):
            for coin in coin_list:
                test.loc[coin, f'sharpe {day} days'] = self.coins.sharpe(coin, days_range=day,
                                                                         from_date=pd.Timestamp(date) -
                                                                         pd.Timedelta(day + 1, 'days'))[:date].iloc[-1]
        for column in test.columns:
            test.loc['matchs', column] = sum([1 if test.index.values[i] == coin else 0 for i, coin in
                                             enumerate(test[column].sort_values(ascending=False).index)])
            test.loc['rms', column] = np.average([abs(test[column].get(coin, 0) - nm_index_for_date.get(coin, 0)
                                                      ) for coin in test.index.values if coin in coin_list])

        # if len(test) > 1:
        #     print(f'Best match: {test.T.matchs.sort_values(ascending=False).index[0]}.')

        return test.T.rms.sort_values()

    def tech_data(self, nm_index, n=None):
        return self.history[['date', 'symbol', f'NM{nm_index}']].iloc[:n].join(
            self.history[['date', 'symbol']].iloc[:n].progress_apply(
            lambda row: self.coins.ta(row['symbol'], from_date=row['date'] - pd.Timedelta(100, 'days')
                                      ).asof(next_date(row['date'])), axis=1)).drop(
                                      ['Asset', 'Close time'], axis='columns')


class Statement:
    def __init__(self, load=True, datafile=None):
        if load and datafile is None:
            try:
                # noinspection PyPep8Naming,PyShadowingNames
                from config import statement_file as STATEMENT_FILE
            except (ImportError, ModuleNotFoundError):
                global STATEMENT_FILE
            datafile = STATEMENT_FILE
        self.daily_yield = self.load(datafile)
        self.filename = datafile
        self._binance_api = Portfolio()
        self.transactions = self.load(datafile)

    @property
    def binance_api(self):
        if self._binance_api is None:
            self._binance_api = Portfolio(connect=True)
        return self._binance_api

    def load(self, datafile=None):
        if datafile is None:
            try:
                # noinspection PyPep8Naming,PyShadowingNames
                from config import statement_file as STATEMENT_FILE
            except (ImportError, ModuleNotFoundError):
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
            logging.error(e)

    def statement(self, since=None):
        trades = []
        last_transaction = None
        symbols = [i.get('symbol') for i in self.binance_api.get_all_tickers()]
        for symbol in tqdm(symbols):
            try:
                if since is not None:
                    since = pd.Timestamp(since).value // 10**6
                    trades += self._binance_api.get_my_trades(symbol=symbol, startTime=since)
                else:
                    try:
                        last_transaction = self.transactions[self.transactions.symbol == symbol].time.max(
                                ).value // 10**6
                        trades += self.binance_api.get_my_trades(symbol=symbol, startTime=last_transaction)
                    except Exception as e:
                        logging.warning(e)
                        trades += self.binance_api.get_my_trades(symbol=symbol)
            except BinanceAPIException as e:
                logging.error(e)
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
            try:
                dusts = pd.DataFrame([i.get('logs')[0] for i in self.binance_api.get_dust_log()['results']
                                     ['rows'] if i.get('logs') is not None])
            except Exception as e:
                logging.error(e)

        dusts = dusts.applymap(partial(pd.to_numeric, errors='ignore'))
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
        dusts.loc[dusts.isBuyer, 'qty'] = dusts.loc[dusts.isBuyer, 'transferedAmount'] + dusts.loc[
                                                    dusts.isBuyer, 'commission']
        dusts.loc[dusts.isBuyer, 'price'] = dusts.loc[dusts.isBuyer, 'quoteQty'] / dusts.loc[dusts.isBuyer, 'qty']
        dusts.loc[~dusts.isBuyer, 'quoteQty'] = dusts.loc[~dusts.isBuyer, 'qty']**2 / dusts.loc[
                                                          ~dusts.isBuyer, 'transferedAmount']
        dusts.loc[~dusts.isBuyer, 'price'] = dusts.loc[~dusts.isBuyer, 'qty'] / dusts.loc[
                                                       ~dusts.isBuyer, 'transferedAmount']
        dusts['commissionAsset'] = 'BNB'
        dusts['orderListId'] = -1
        transactions = pd.concat([transactions, dusts[transactions.columns]]).set_index('time').sort_index()
        self.transactions = self.transactions.append(transactions)
