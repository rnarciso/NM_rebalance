from websocket import create_connection
import pandas as pd
import requests
import string
import random
import json
import re


def prepend_header(st):
    return "~m~" + str(len(st)) + "~m~" + st


def construct_message(func, param_list):
    return json.dumps({"m": func, "p": param_list}, separators=(',', ':'))


def create_message(func, param_list):
    return prepend_header(construct_message(func, param_list))


class TradingView:

    def __init__(self, username='username', password='password'):
        self.username = username
        self.password = password
        self.ws = None
        self.auth_token = None
        self.default_exchange = 'BINANCE'

    def get_auth_token(self):
        sign_in_url = 'https://www.tradingview.com/accounts/signin/'
        username = self.username
        password = self.password
        data = {"username": username, "password": password, "remember": "on"}
        headers = {'Referer': 'https://www.tradingview.com'}
        response = requests.post(url=sign_in_url, data=data, headers=headers)
        self.auth_token = response.json()['user']['auth_token']
        return self.auth_token

    @property
    def session(self):
        string_length = 12
        letters = string.ascii_lowercase
        random_string = ''.join(random.choice(letters) for i in range(string_length))
        return "qs_" + random_string

    @property
    def chart_session(self):
        string_length = 12
        letters = string.ascii_lowercase
        random_string = ''.join(random.choice(letters) for i in range(string_length))
        return "cs_" + random_string

    def send_raw_message(self, message):
        self.ws.send(prepend_header(message))

    def send_message(self, func, args):
        self.ws.send(create_message(func, args))

    def get_candles(self, ticker=None, from_date=None, to_date=None, period='1D'):
        if ticker is None:
            ticker = 'CRYPTOCAP:TOTAL'
        if ticker.find(':') < 0:
            ticker = f'{self.default_exchange}:{ticker}'

        # Initialize the headers needed for the websocket connection
        headers = json.dumps({'Origin': 'https://data.tradingview.com'})

        # Then create a connection to the tunnel
        self.ws = create_connection('wss://data.tradingview.com/socket.io/websocket', headers=headers)

        session = self.session

        chart_session = self.chart_session

        # Then send a message through the tunnel
        self.send_message("set_auth_token", ["unauthorized_user_token"])
        self.send_message("chart_create_session", [chart_session, ""])
        self.send_message("quote_create_session", [session])
        self.send_message("quote_set_fields",
                          [session, "ch", "chp", "current_session", "description", "local_description", "language",
                           "exchange", "fractional", "is_tradable", "lp", "lp_time", "minmov", "minmove2",
                           "original_name", "pricescale", "pro_name", "short_name", "type", "update_mode",
                           "volume", "currency_code", "rchp", "rtc"])
        self.send_message("quote_add_symbols", [session, ticker, {"flags": ['force_permission']}])
        self.send_message("quote_fast_symbols", [session, ticker])
        self.send_message("resolve_symbol", [chart_session, "symbol_1",
            "={\"symbol\":\"%s\",\"adjustment\":\"splits\",\"session\":\"extended\"}".replace('%s', ticker)])
        self.send_message("create_series", [chart_session, "s1", "s1", "symbol_1", period, (
                pd.Timestamp(to_date if to_date is not None else 'now') -
                pd.Timestamp(from_date if from_date is not None else '20140301')).days])

        data = [['index', 'date', 'open', 'high', 'low', 'close', 'volume']]
        data += [[int(xi) if i == 1 else pd.Timestamp.fromtimestamp(float(xi)) if i == 4 else float(xi) for i, xi in
                 enumerate(re.split('\[|:|,|\]', x0)) if i in list(range(4, 10)) + [1]]  for x0 in
                 re.search('"s":\[(.+?)\}\]', '\n'.join([result for result in [self.ws.recv() for _ in range(3)]])
                           ).group(1).split(',{\"')]
        data = pd.DataFrame(data[1:], columns=data[0]).set_index('index')
        data.index.name = None
        return data

