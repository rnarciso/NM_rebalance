import ta
import json
import warnings
import numpy as np
from nm.util import *
from tqdm import tqdm
from functools import partial
# noinspection PyPackageRequirements
from binance.client import Client
from collections.abc import Collection, Iterable
from itertools import permutations, combinations
# noinspection PyPackageRequirements
from binance.exceptions import BinanceAPIException

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
            time_offset = tz_remove_and_normalize('utc') - tz_remove_and_normalize(
                    pd.Timestamp(self._client.get_server_time().get("serverTime") * 10 ** 6))
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
        if hasattr(super(), attr):
            super().__getattribute__(attr)
        else:
            logging.error(f'Binance API object has no attribute {attr}, '
                          f'or is not initialized.\nCheck configuration file!')

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

    # noinspection PyUnboundLocalVariable
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
            try:
                # noinspection PyUnboundLocalVariable
                self._client = Client(api_key, api_secret)
                self.connected = True
            except NameError:
                try:
                    self._client = Client()
                    self.connected = True
                except BinanceAPIException:
                    self.connected = False
            except Exception as e:
                logging.error(e)
                self.connected = False

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

    def refresh_balance(self, client=None, keyname: str = None, include_locked: bool = None):
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
            diff['quote_value'] = abs(diff.apply(lambda row: prices.loc[f"{row['coin']}{quote_asset}", 'bidPrice'
                                                 if row['delta'] < 0 else 'askPrice'] * row['delta']
                                                 if f"{row['coin']}{quote_asset}" in prices.index
                                                 else (row['delta'] if quote_asset == row['coin'] else 0), axis=1))

            new_diff = diff['quote_value'].sum()

            if new_diff >= previous_diff or new_diff < portfolio_value * threshold / 100:
                break
            else:
                previous_diff = new_diff

            # noinspection PyUnresolvedReferences
            try:
                diff['fee'] = diff.apply(lambda row: fees.loc[f"{row['coin']}{quote_asset}", fee_type] *
                                         row['quote_value'] if f"{row['coin']}{quote_asset}" in prices.index else 0,
                                         axis=1)
            except KeyError:
                diff['fee'] = diff.apply(lambda row: 0.001 if row['coin'] != quote_asset else 0, axis=1)

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
                                                 if row['coin'] != quote_asset
                                                 else 0, axis=1)

                to_buy['actual_value'] = to_buy['quote_value'] - to_buy['fee']
                to_buy['delta'] *= to_buy['actual_value'] / to_buy['quote_value']

                if market_order:
                    # noinspection
                    buy_orders = pd.Series(to_buy.apply(lambda row: self.order_trim(row['pair'], row['delta'],
                                                                                    side=SIDE_BUY,
                                                                                    order_type=ORDER_TYPE_MARKET),
                                                        axis=1))
                else:
                    # noinspection
                    buy_orders = pd.Series(to_buy.apply(lambda row: self.order_trim(row['pair'], row['delta'],
                                                        price=prices.loc[row['pair'], f"askPrice"] *
                                                        (1 if fee_type == 'taker' else (1 - MAKER_PREMIUM)),
                                                        side=SIDE_BUY, order_type=ORDER_TYPE_LIMIT
                                                        if fee_type == 'taker' else ORDER_TYPE_LIMIT_MAKER), axis=1))
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
                                                                                *[date.strftime('%Y-%m-%d')] * 2))
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
            logging.error(e)
        return self._df

    @df.setter
    def df(self, value):
        self._df = value

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
               add_next_day_results=False, pump_percentage=1.5, dump_percentage=-1.5):
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
                logging.info(f'No data for {symbol} from {from_date} to {to_date}.')
            except Exception as e:
                logging.error(e)
        if SYMBOL in self.history.columns:
            self.history = self.history.sort_values(SYMBOL).sort_index()
            self.save()
        return self.history

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
                logging.error(e)
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
            if (len(df_for_coin) > 0 and tz_remove_and_normalize('utc') > tz_remove_and_normalize(to_date) >
                df_for_coin[CLOSE_TIME].max() + pd.Timedelta(1, 'day')):
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
            to_date = tz_remove_and_normalize('utc')
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
                    fee_for_date = fee_database[fee_database[SYMBOL].isin([f'{c}{QUOTE_ASSET}'
                                                for c in set(coins).difference(last_coins)])]
                except NameError:
                    fee_for_date = fee_database[fee_database.symbol.isin([f'{c}{QUOTE_ASSET}' for c in coins])]
                fee_for_date = np.average(fee_for_date.taker) * 2 if date > from_date else 1
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
            return df
        else:
            return accrued_yield

    def account_yield_for_period(self, accounts, from_date, *kwargs):
        if isinstance(accounts, dict):
            accounts = [accounts]
        return np.average([self.nm_index_yield_for_period(account.get('index'),
                                                          from_date, *kwargs) for account in accounts])

    def yield_simulation(self, contributions: Deposits, to_date=None, fees=True, slippage=AVG_SLIPPAGE, top_n=4,
                         fee_type='taker', interval=1):
        dfs = {}
        for nm in sorted(contributions.df.NM.unique()):
            dfs[nm] = pd.DataFrame()
            min_date = next_date(contributions.df[contributions.df.NM == nm].index.min(), -1)
            to_date = self.default_to_date(to_date)
            value = 0
            last_set_of_coins = set()
            nmdf = dfs[nm]
            for date in tqdm(pd.date_range(min_date, to_date, freq=f'{interval}D')):
                if to_date <= tz_remove_and_normalize('utc'):
                    nmdf.loc[date, OPEN] = value
                    if next_date(date) in contributions.df[contributions.df['NM'] == nm].index:
                        nmdf.loc[date, DEPOSIT] = contributions.df.loc[next_date(date), QUOTE_VALUE]
                        value += contributions.df.loc[next_date(date), QUOTE_VALUE]
                    else:
                        nmdf.loc[date, DEPOSIT] = 0
                    # noinspection PyBroadException
                    try:
                        coins_for_date = self.advisor.get(nm, date)[:top_n]
                        if len(coins_for_date) > 0:
                            coins_for_date[DATE] = date
                            nmdf.loc[date, f'NM{nm} {YIELD}'] = self.coin_data.yield_for_coins(
                                    coins_for_date.index, from_date=date, to_date=next_date(date, interval - 1))
                            nmdf.loc[date, f'NM{nm} coins'] = ','.join(coins_for_date.index)
                            value = value * (1 + nmdf.loc[date, f'NM{nm} {YIELD}'])
                            nmdf.loc[date, CLOSE] = value
                            new_coins = set(coins_for_date.index).difference(last_set_of_coins)
                            if fees and len(new_coins) > 0:
                                value *= (1 - self.binance_api.fees[self.binance_api.fees.index.isin(
                                        [f'{coin_name}{QUOTE_ASSET}' for coin_name in new_coins])][fee_type].mean()
                                          * (2 if date > min_date else 1) * (len(new_coins) / top_n))
                            if slippage is not None and len(new_coins) > 0:
                                value *= (1 - slippage * (2 if date > min_date else 1) * (len(new_coins) / top_n))
                            last_set_of_coins = set(coins_for_date.index)
                            nmdf.loc[date, ADJUSTED_CLOSE] = value
                    except Exception as e:
                        logging.error(f'\n{e}, while processing data for {date.date()}.')

        consolidated_df = pd.concat(dfs.values()).reset_index().groupby('index').sum()
        consolidated_df.index = consolidated_df.index.shift(1, freq='D')
        consolidated_df.index.name = None
        return consolidated_df

    @staticmethod
    def default_to_date(to_date):
        if to_date is None:
            to_date = tz_remove_and_normalize('utc')
        else:
            to_date = pd.Timestamp(to_date)
        return to_date

    def yield_simulation2(self, deposits: Deposits, to_date=None, fees=True, slippage=AVG_SLIPPAGE, top_n=4,
                         fee_type='taker', interval=1):

        def add_deposits(row):
            dfs.setdefault(row['NM'], pd.DataFrame()).loc[row.name, DEPOSIT] = row[QUOTE_VALUE]

        def add_yield(row):
            date = row.name
            date = next_date(date, - 1)
            to_date = row.to_date
            coins = tuple(self.advisor.get(nm, date).index[:top_n])
            row[f'NM{nm} {YIELD}'] = self.coin_data.yield_for_coins(coins, date, next_date(to_date, interval - 1))
            row[f'NM{nm} coins'] = ','.join(coins)
            return row

        dfs = {}
        to_date = self.default_to_date(to_date)
        deposits.apply(add_deposits, axis=1)
        for nm in dfs.keys():
            dfs[nm] = pd.concat([dfs[nm], pd.DataFrame(index=pd.date_range(start=dfs[nm].index.min(),
                                end=to_date, freq=f'{interval}D', closed='right'))]).fillna(0)
            dfs[nm]['to_date'] = dfs[nm].index
            dfs[nm]['to_date'] = dfs[nm]['to_date'].shift(-1)
            dfs[nm] = dfs[nm].progress_apply(add_yield, axis=1)
            dfs[nm][OPEN] = 0


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
        self._df = None
        self._coin_data = None
        self._ta = None
        if datafile is None:
            try:
                from config import nm_data_file as datafile
            except (ImportError, ModuleNotFoundError):
                datafile = NMDATA_FILE
        if datafile is not None and len(datafile) < 1:
            datafile = None
        self.filename = datafile
        self.subset = ['price'] + [f'NM{i}' for i in range(1, 5)]
        if load and datafile is not None:
            self.df = self.load(datafile)

    def __repr__(self):
        return f'<NMData container class at {hex(id(self))}:\n{self.df.__repr__()}' \
               f'\n\nLast update on {self.last_update}>'

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
        return self._df

    @df.setter
    def df(self, value):
        self._df = value

    def import_from_excel(self, filename='~/Downloads/NM_Guathan.xlsx'):
        imported_nm = pd.concat([df[['Data', 'NM1', 'NM2', 'NM3', 'NM4', 'Moeda']]
                       for df in (pd.read_excel(filename, f'NM{i + 1}') for i in tqdm(range(4)))]).rename(
                {'Data': 'date', 'Moeda': 'symbol'}, axis='columns').drop_duplicates().reset_index(drop=True)
        imported_nm.symbol = imported_nm.symbol.str.replace('USDT', '')
        imported_nm.date = pd.to_datetime(imported_nm.date, errors='coerce')
        imported_nm = imported_nm[~imported_nm.date.isna()]
        imported_nm.index = pd.DatetimeIndex(
                imported_nm.date).tz_localize('utc').tz_convert('Brazil/East').tz_localize(None)
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

    @property
    def last_update(self):
        if self.df is not None and 'date' in self.df.columns:
            return self.df.date.max().tz_localize(NM_TIME_ZONE)
        else:
            return pd.Timestamp(EXCHANGE_OPENING_DATE)

    def load(self, datafile=None):
        if datafile is None:
            if self.filename is None:
                self.filename = NMDATA_FILE
            datafile = self.filename
        try:
            self._df = pd.read_pickle(datafile)
        except FileNotFoundError:
            pass
        except ValueError:
            downgrade_pickle(datafile)
            return self.load(datafile)
        except Exception as e:
            logging.error(e)
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

    def get(self, index=1, date='utc'):
        date = next_date(date, -1)
        columns = [SYMBOL, f'NM{index}']
        if 'date' in self.df.columns:
            self.df = self.df.set_index('date')
        retries = 3
        while retries > 0:
            retries -= 1
            try:
                nm_for_date = self.df.sort_index().loc[date.strftime('%Y%m%d')]
            except KeyError:
                nm_for_date = pd.DataFrame()
            if len(nm_for_date) < 1:
                if len(self.get_nm_data()) > 0:
                    continue
                else:
                    break
            else:
                return nm_for_date[columns].set_index(SYMBOL).sort_values(f'NM{index}', ascending=False)
        raise IndexError

    def get_nm_data(self, url=None):
        if url is None:
            url = self._nm_url
        if self.df is not None:
            max_date = self.df.index.max()
        else:
            max_date = tz_remove_and_normalize(EXCHANGE_OPENING_DATE)
        df = pd.DataFrame()
        for i in range(1, NM_MAX+1):
            try:
                mndf = pd.read_html(url + str(i), decimal=',', thousands='.')[0]
                date = pd.to_datetime(mndf.iloc[-1][0].replace(UPDATED_ON, '').replace(AT_SIGN, ' '), dayfirst=True)
                if date > max_date:
                    mndf['date'] = date
                    mndf.columns = NM_COLUMNS
                    mndf = mndf.drop(index=[0, 1, mndf.index.max()], columns=['price'])
                    mndf = mndf.applymap(partial(pd.to_numeric, errors='ignore'))
                    df = df.append(mndf)
                else:
                    return pd.DataFrame()
            except Exception as e:
                logging.error(e)
        df = df.drop_duplicates()
        if len(df) > 0:
            self._df = pd.concat([self.df, df.set_index('date')])
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
                                            f'{coins}, match: '
                                            f'{np.average([y for c, y in coin_yields_for_date.items() if c in coins])}'
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
            from_date = row[DATE] - pd.Timedelta(100, 'days')
            to_date = next_date(row[DATE])
            try:
                return self.coins.history.asof(to_date).add_ta(coin, from_date=from_date).drop(SYMBOL)
            except IndexError:
                return pd.DataFrame()

        if date is None:
            return self.coins.history[[DATE, SYMBOL, f'NM{nm_index}']].iloc[:n].join(
                    self.coins.history[[DATE, SYMBOL]].iloc[:n].progress_apply(add_ta, axis=1))
        else:
            history = self.coins.history.set_index('date')[date:date]
            if len(history) > 0:
                history = history.reset_index()

                return history[[SYMBOL, f'NM{nm_index}']].iloc[:n].join(
                        history[[DATE, SYMBOL]].iloc[:n].progress_apply(add_ta,
                                                                        axis=1)).set_index(SYMBOL)
            else:
                return pd.DataFrame()

    def yield_for_date(self, nm_index, date, top_n=4):
        date = next_date(date, -1)
        coins = self.get(nm_index, date).index[:top_n]
        return self.coins.yield_for_coins(coins, from_date=date)


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
        dusts.loc[~dusts.isBuyer, 'quoteQty'] = dusts.loc[~dusts.isBuyer, 'qty'] ** 2 / dusts.loc[
            ~dusts.isBuyer, 'transferedAmount']
        dusts.loc[~dusts.isBuyer, 'price'] = dusts.loc[~dusts.isBuyer, 'qty'] / dusts.loc[
            ~dusts.isBuyer, 'transferedAmount']
        dusts['commissionAsset'] = 'BNB'
        dusts['orderListId'] = -1
        transactions = pd.concat([transactions, dusts[transactions.columns]]).set_index('time').sort_index()
        self.transactions = self.transactions.append(transactions)
