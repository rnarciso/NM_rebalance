import json
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from functools import partial
# noinspection PyPackageRequirements
from binance.client import Client
from itertools import permutations
from collections.abc import Collection
# noinspection PyPackageRequirements
from binance.exceptions import BinanceAPIException
from nm.util import sum_dict_values, truncate, math, safe_save


AT_SIGN = ' às '
COIN_MARKET_COLUMNS = ['volume_24h', 'percent_change_1h', 'percent_change_24h', 'percent_change_7d', 'market_cap']
KEYFILE = '.keys'
MAKER_PREMIUM = 0.1 / 100
MINIMUM_TIME_OFFSET = 2000
NM_COLUMNS = ['symbol', 'price', 'NM1', 'NM2', 'NM3', 'NM4', 'date']
NM_MAX = 4
NM_TIME_ZONE = 'Brazil/East'
NMDATA_FILE = 'nm_index.dat'
ORDER_AMOUNT_REDUCING_FACTOR = 5 / 100
SINCE = '20191231'
SYMBOL = 'symbol'
UPDATED = 'atualizado'
UPDATED_ON: str = f'{UPDATED} em'
YIELD_FILE = 'yield.dat'
# Following constants are imported from Client later on
SIDE_SELL, SIDE_BUY, TIME_IN_FORCE_GTC, ORDER_STATUS_FILLED, ORDER_TYPE_LIMIT, ORDER_TYPE_LIMIT_MAKER, \
    ORDER_TYPE_MARKET = [None]*7


class Backtest:
    def __init__(self, advisor=None, datafile=None, load=True):
        if advisor is None:
            advisor = NMData()
        self.advisor = advisor
        if load and datafile is None:
            try:
                # noinspection PyPep8Naming,PyShadowingNames
                from config import yield_file as YIELD_FILE
            except ImportError:
                global YIELD_FILE
            datafile = YIELD_FILE
        self.daily_yield = self.load(datafile)
        self.filename = datafile
        self.yield_df_columns = [f'NM{i}' for i in range(1, NM_MAX + 1)]
        self.binance_api = None
        for column in self.yield_df_columns.copy():
            for suffix in ['', ' time']:
                for feature in ['bottom', 'top']:
                    self.yield_df_columns.append(f'{column} {feature}{suffix}')

    # noinspection PyUnboundLocalVariable
    @staticmethod
    def load(datafile=None):
        if datafile is None:
            try:
                # noinspection PyPep8Naming,PyShadowingNames
                from config import yield_file as YIELD_FILE
            except ImportError:
                datafile = YIELD_FILE
        try:
            return pd.read_pickle(datafile)
        except (FileNotFoundError, ValueError):
            return pd.DataFrame()

    def save(self):
        if self.filename is None:
            try:
                # noinspection PyPep8Naming,PyShadowingNames
                from config import yield_file as YIELD_FILE
            except ImportError:
                global YIELD_FILE
            # noinspection PyUnboundLocalVariable
            self.filename = YIELD_FILE
        safe_save(self.daily_yield, self.filename)

    def yield_report(self, from_date=None, to_date=None, accounts=None, candle=Client.KLINE_INTERVAL_1MINUTE):
        if from_date is None:
            from_date = (pd.Timestamp('now')).date().strftime('%Y-%m-%d')
        if to_date is None:
            to_date = (pd.Timestamp(from_date) + pd.Timedelta(1, 'day')).date().strftime('%Y-%m-%d')
        if self.binance_api is None:
            self.binance_api = Portfolio(connect=True)
        binance_api = self.binance_api
        if accounts is None:
            portfolios = [1, 2, 4]
            top_n = {1: 4, 2: 4, 4: 4}
        else:
            portfolios = [v for d in accounts for k, v in d.items() if k == 'index']
            top_n = {portfolios[i]: d.get('top_n') for i, d in enumerate(accounts)}
        date_range = pd.date_range(from_date, to_date)
        nm_yield = pd.DataFrame()
        for i, from_date in enumerate(tqdm(date_range, desc='Retrieving NM yield for specified dates')):
            if from_date == date_range[-1]:
                continue
            nm_for_date = pd.DataFrame()
            coin_data = pd.DataFrame()
            for nm_index in portfolios:
                nm_coins = self.advisor.get(nm_index, from_date)[:top_n.get(nm_index)]
                prices = nm_coins.to_dict().get('price')
                nm_coins = nm_coins.index.values
                for coin in nm_coins:
                    if coin not in nm_for_date.columns:
                        symbol = f'{coin}USDT'
                        coin_data = pd.DataFrame()
                        retries = 3
                        while len(coin_data) < 1440 and retries > 0:
                            try:
                                coin_data = pd.DataFrame(binance_api.get_historical_klines(symbol, candle,
                                                         from_date.strftime('%Y-%m-%d'), date_range[i + 1].strftime
                                                         ('%Y-%m-%d'))).apply(pd.to_numeric)
                            except Exception as e:
                                logging.error(e)
                                pass
                            retries -= 1
                        coin_data.columns = ['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time',
                                             'Quote asset volume', 'Number of trades', 'Taker buy base asset volume',
                                             'Taker buy quote asset volume', 'Ignore']
                        nm_for_date[coin] = (coin_data['Close'] / prices[coin] - 1)*100

                nm_for_date[f'NM{nm_index}'] = nm_for_date.apply(lambda row: np.average([v for k, v in row.items()
                                                                 if k in nm_coins]), axis=1)
            try:
                nm_for_date['time'] = (coin_data['Open time'] * 10**6).apply(pd.Timestamp).values
            except KeyError:
                continue
            features_for_date = nm_for_date[nm_for_date['time'] < date_range[i+1]].iloc[-1]
            for nm_index in portfolios:
                features_for_date[f'NM{nm_index} top'] = nm_for_date[f'NM{nm_index}'].max()
                features_for_date[f'NM{nm_index} bottom'] = nm_for_date[f'NM{nm_index}'].min()
                features_for_date[f'NM{nm_index} top time'] = nm_for_date[
                                  nm_for_date[f'NM{nm_index}'] == nm_for_date[f'NM{nm_index}'].max()]['time'].values[0]
                features_for_date[f'NM{nm_index} bottom time'] = nm_for_date[
                                  nm_for_date[f'NM{nm_index}'] == nm_for_date[f'NM{nm_index}'].min()]['time'].values[0]
            nm_yield = nm_yield.append(features_for_date, ignore_index=True)
        these_first = [column for column in self.yield_df_columns if column in nm_yield.columns]
        try:
            nm_yield = nm_yield[these_first + list(nm_yield.columns.difference(these_first))].set_index('time')
            nm_yield.index = nm_yield.index.normalize()
            nm_yield.index.name = None
            return nm_yield
        except Exception as e:
            logging.error(e)

    def update_yield_data(self, to_date=None, from_date=None, top_n=4):
        if from_date is None:
            try:
                from_date = self.daily_yield.index.max()
                if pd.Timestamp(from_date) is pd.NaT or pd.Timestamp(from_date) < pd.Timestamp(SINCE):
                    raise ValueError
            except ValueError:
                from_date = SINCE
            from_date = pd.Timestamp(from_date) + pd.Timedelta(1, 'day')
        else:
            from_date = pd.Timestamp(from_date)
        if to_date is None:
            to_date = pd.Timestamp('now').date()
        elif isinstance(to_date, int):
            to_date = from_date + pd.Timedelta(to_date, 'days')
        else:
            to_date = pd.Timestamp(to_date) + pd.Timedelta(1, 'days')
        if to_date < from_date:
            date = to_date
            to_date = from_date
            from_date = date
        from_date = pd.Timestamp(from_date.date())
        to_date = pd.Timestamp(to_date.date())
        accounts = [{'index': i, 'top_n': top_n} for i in range(1, NM_MAX + 1)]
        new_data = self.yield_report(from_date, to_date, accounts=accounts)[self.yield_df_columns]
        self.daily_yield = pd.concat([self.daily_yield, new_data])
        self.save()
        return self.daily_yield


class NMData:
    _nm_url: str

    def __init__(self, nm_url=None, load=True, datafile=None):
        if nm_url is None:
            from config import nm_url
        self._nm_url = nm_url
        self._nm_data = None
        if datafile is None:
            try:
                from config import nm_data_file as datafile
            except ImportError:
                datafile = NMDATA_FILE
        if datafile is not None and len(datafile) < 1:
            datafile = None
        self.filename = datafile
        self.subset = ['price']+[f'NM{i}' for i in range(1, 5)]
        if load and datafile is not None:
            self._nm_data = self.load(datafile)

    @property
    def last_update(self):
        return self._nm_data.date.max().tz_localize(NM_TIME_ZONE)

    def __repr__(self):
        return f'<NMData container class at {hex(id(self))}:\n{self._nm_data.__repr__()}>'

    def __str__(self):
        return self._nm_data.__str__()

    def load(self, datafile=None):
        if datafile is None:
            if self.filename is None:
                self.filename = NMDATA_FILE
            datafile = self.filename
        try:
            self._nm_data = pd.read_pickle(datafile)
        except FileNotFoundError:
            pass
        return self._nm_data

    def save(self):
        if self.filename is None:
            self.filename = NMDATA_FILE
        safe_save(self._nm_data, self.filename)

    def sort(self, by=None):
        if by is None:
            by = ['date', 'symbol']
            self._nm_data = self._nm_data.sort_values(by)
            return self._nm_data

    def to_numeric(self):
        self._nm_data = self._nm_data.applymap(partial(pd.to_numeric, errors='ignore'))

    def get(self, index=1, date='now', include_price=True):
        df: pd.DataFrame = self._nm_data
        df.index = pd.DatetimeIndex(pd.to_datetime(df.date)).tz_localize(NM_TIME_ZONE
                                                                         ).tz_convert('UTC').to_series()
        df = df.loc[pd.Timestamp(date).tz_localize(NM_TIME_ZONE).tz_convert('UTC').normalize():pd.Timestamp(date)
                                                                .tz_localize(NM_TIME_ZONE).tz_convert('UTC')]
        df = df.drop_duplicates(subset=['symbol'], keep='last')
        if include_price:
            df = df[['symbol', f'NM{index}', 'price']].sort_values(f'NM{index}', ascending=False).set_index('symbol')
        else:
            df = df[['symbol', f'NM{index}']].sort_values(f'NM{index}', ascending=False).set_index('symbol')

        return df

    def get_nm_data(self, url=None):
        if url is None:
            url = self._nm_url
        if self._nm_data is None:
            df = pd.DataFrame()
        else:
            df = self._nm_data

        for i in range(1, 5):
            mndf = pd.read_html(url+str(i), decimal=',', thousands='.')[0]
            mndf['date'] = pd.to_datetime(mndf.iloc[-1][0].replace(UPDATED_ON, '').replace(AT_SIGN, ' '), dayfirst=True)
            mndf.columns = NM_COLUMNS
            mndf = mndf.drop(index=[0, 1, mndf.index.max()])
            df = df.append(mndf)
        df.index = pd.RangeIndex(len(df))
        self._nm_data = df
        self.to_numeric()
        self.drop_duplicates()
        if self.filename is not None and len(self._nm_data) > 0:
            self.save()
        return self._nm_data

    def drop_duplicates(self):
        self._nm_data = self._nm_data.drop_duplicates(subset=self.subset)


class Portfolio:

    _time_offset: int

    def __init__(self, keyname: str = None, connect=False, include_locked=False, config=None):
        if isinstance(config, dict):
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
        self.transactions = pd.DataFrame()
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

            df['USDT Value'] = df.apply(lambda row: float(prices.get(f'{row.name}USDT', '0')) *
                                        row['Amount'] if row.name != 'USDT' else row['Amount'], axis=1)
            df['%'] = df['USDT Value'] / df['USDT Value'].sum() * 100
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
        small_balances = [asset for asset in balance.index if balance.loc[asset, 'USDT Value'] <
                          self.minimal_order(f'{asset}USDT')]
        try:
            bnb_index = small_balances.index(base_asset)
            small_balances.pop(bnb_index)
        except ValueError:
            pass
        for asset in small_balances:
            try:
                self.transfer_dust(asset=asset)
            except BinanceAPIException as e:
                logging.error(f'{e}. Asset: {asset}.')

    def fit_market_order(self, market='BTCUSDT',
                         quote_amount=None, side=SIDE_BUY, add_fee=True):
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
                                      quote_asset: str = 'USDT',
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

    def statement(self, since=None):
        trades = []
        last_transaction = None
        symbols = [i.get('symbol') for i in self.get_all_tickers()]
        for symbol in tqdm(symbols):
            try:
                if since is not None:
                    since = pd.Timestamp(since).value // 10**6
                    trades += self.get_my_trades(symbol=symbol, startTime=since)
                else:
                    try:
                        last_transaction = self.transactions[self.transactions.symbol == symbol].time.max(
                                ).value // 10**6
                        trades += self.get_my_trades(symbol=symbol, startTime=last_transaction)
                    except Exception as e:
                        logging.warning(e)
                        trades += self.get_my_trades(symbol=symbol)
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
                dusts = pd.DataFrame([i.get('logs')[0] for i in self.get_dust_log(startTime=since)
                                     ['results']['rows'] if i.get('logs') is not None])
        except KeyError:
            try:
                dusts = pd.DataFrame([i.get('logs')[0] for i in self.get_dust_log()['results']
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


# import constants from Client
for const in globals().copy().keys():
    if globals()[const] is None and Client.__dict__.get(const) is not None:
        globals()[const] = Client.__dict__[const]
