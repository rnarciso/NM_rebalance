accounts = [
    # One dict for each account you want to rebalance
    dict(
    account_name='nm1',            # Account name (
    api_key='J1m80FSIIvI6pvSAwv0qL7DeXyQq8Gv3pG27oFF7dZry057FRtWpDXpNJ58eZV51',                 # Binance API key
    api_secret='oplGt8yoJcimbZVLmjQFihJVovAeYQ18ZjnakUMD07ZUETlU4BXxeysPH5AqJNT7',              # Binance API secret
    index=1,                    # NM index to follow
    top_n=4,                    # Use Top N coins
    rebalance_interval=24*60,   # Interval between rebalance attempts in minutes
    convert_small_balances_before_rebalance=True,
        ),
    dict(
    account_name='nm2',            # Account name (
    api_key='6mBCaWXb4b29GWH5K7N5fOeY0mAu5VvJcIURq7eaoB6FhoXBOOa6XFHOlgQQ692F',                 # Binance API key
    api_secret='46kJ7RSCOyWMc1v9JXmqnHoDDlRsoY3urpFvcVDEhL4FHEwRvIUwqrYeV2Fio4xY',              # Binance API secret
    index=2,                    # NM index to follow
    top_n=4,                    # Use Top N coins
    rebalance_interval=24*60,   # Interval between rebalance attempts in minutes
    convert_small_balances_before_rebalance=True,
        ),
    dict(
    account_name='nm4',            # Account name (
    api_key='2IJ8SMoSsO54114SIQcWwiQPcWb2nvusRD4O28iqCM7g7L07pLV2Wqx39AWenNYj',                 # Binance API key
    api_secret='wnnrOGHivUDSZQvXWE9HZ2pO2tLd4rQe7dWWdiX9c2eaBIHOJgVaMcRda2Ht5Oon',              # Binance API secret
    index=4,                    # NM index to follow
    top_n=4,                    # Use Top N coins
    rebalance_interval=24*60,   # Interval between rebalance attempts in minutes
    convert_small_balances_before_rebalance=True,
        ),
    dict(
    account_name='AMNS nm1',            # Account name (
    api_key='Hoy8sJAcuI8cK5NXrI9rOq5puWRxsGssSvJqrp2Xd5iw2aJEWCDFmCqsrZA58Tqg',                 # Binance API key
    api_secret='agr6OqCqrInNJYpmNlHwtFxeBylp8xwJ65pQGYyTnEgXIfYoNRE188CA8nZ3BRdG',              # Binance API secret
    index=1,                    # NM index to follow
    top_n=4,                    # Use Top N coins
    rebalance_interval=24*60,   # Interval between rebalance attempts in minutes
    convert_small_balances_before_rebalance=True,
        ),
    dict(
    account_name='AMNS nm2',            # Account name (
    api_key='zY676gUk2gAyuzSEP5gIvDRcdwdjrxwxQInchDiLCv9FCIRHEjxyEX0OCi4p7DBM',                 # Binance API key
    api_secret='NFHX4pLPfP5JOGlbfnlaUg0eM5PJ8BWi5zN1gx7vsJhnZK4L1OUgUr1CqWYHI1kZ',              # Binance API secret
    index=2,                    # NM index to follow
    top_n=4,                    # Use Top N coins
    rebalance_interval=24*60,    # Interval between rebalance attempts in minutes
    convert_small_balances_before_rebalance=True,
        ),
    ]

nm_url = "http://anovamoeda.oinvestidordesucesso.com/IS/nmREPORT.asp?NM="                 # Full NM index URL without the trailing index (i.e. http://cooldude.com/nmREPORT.asp?NM=')

RETRIES = 5                     # number of rebalance attempts before giving up

save_nm_data = True

nm_data_file = 'data/nm_table.dat'

yield_file = 'data/yield.dat'
