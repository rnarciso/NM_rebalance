#!/usr/bin/python3
import sys
import time
import getopt
import logging
import pandas as pd
from nm import NMData, BinanceAccount, Rebalance
from nm.util import tz_remove_and_normalize, QUOTE_ASSET, log_error
from config import RETRIES as DEFAULT_RETRIES
from config import accounts as account_config


def rebalance(argv):
    force_first_rebalance = False
    dry_run = False
    opts, args = getopt.getopt(argv, "dhf", ["dry_run",  "help", "force"])
    for opt, arg in opts:
        if opt == '-h':
            print('Usage: rebalance [--force] --dry_run')
            return
        elif opt in ('-f', '--force'):
            force_first_rebalance = True
        elif opt in ('-d', '--dry_run'):
            dry_run = True
            force_first_rebalance = True
    nm_data = NMData()
    retries = DEFAULT_RETRIES
    if (tz_remove_and_normalize('utc') - tz_remove_and_normalize(nm_data.last_update)).seconds // 60 > min(
            [a.get('rebalance_interval', 24 * 60) for a in account_config]):
        while retries > 0:
            try:
                logging.info('Retrieving NM index data table...')
                nm_data.get_nm_data()
                break
            except Exception as e:
                log_error(e)
                retries -= 1
        else:
            logging.error('Unable to read NM index table. Aborting...')
            exit(-1)
    logging.info('Connecting to configured Binance account(s)...')
    for account in account_config:
        account['portfolio'] = BinanceAccount(config=account)
        account['last_update'] = tz_remove_and_normalize(nm_data.last_update)
        account['force_first_rebalance'] = force_first_rebalance
    rebalancing_completed = False
    first_run = True
    while True:
        for account in account_config:
            try:
                if tz_remove_and_normalize('utc') - account['last_update'] > pd.Timedelta(
                        account['rebalance_interval'], 'minutes') or account['force_first_rebalance']:
                    logging.info(f"Rebalancing account: {account['account_name']}")
                    first_run = False
                    rebalancing_completed = False
                    account['force_first_rebalance'] = False
                    if account.get('convert_small_balances_before_rebalance', False) and not dry_run:
                        logging.info('Converting small balances...')
                        account['portfolio'].convert_small_balances()
                    logging.info('Getting new NM data...')
                    nm_retries = DEFAULT_RETRIES
                    while nm_retries > 0:
                        try:
                            nm_data.get_nm_data()
                            break
                        except Exception as e:
                            log_error(e)
                            nm_retries -= 1
                    if 'distribution' in account.keys():
                        target = account['distribution']
                        logging.info(f"Target distribution {target}.")
                        target = Rebalance(account['portfolio']).trim_target(target)
                        logging.info(f'{target} based on NM index for today.')
                    else:
                        target = nm_data.get(account['index'])[:account['top_n']].index.values
                        logging.info(f"Target NM data for NM{account['index']}: {target}.")
                    retries = DEFAULT_RETRIES
                    while retries > 0:
                        if account['portfolio'].balance[f'{QUOTE_ASSET} Value'].sum() >= 10:
                            print(f"\nCurrent balance:\n{account['portfolio'].balance}\n")
                            logging.info(f'Setting up orders for rebalancing. Attempt # '
                                         f'{DEFAULT_RETRIES - retries + 1}')
                            account_rebalancer = Rebalance(account['portfolio'])
                            orders = account_rebalancer.create_orders(target)
                            if len(orders) < 1:
                                logging.info(f"No more orders, NM{account['index']} portfolio already rebalanced!!!")
                                rebalancing_completed = True
                                break
                            else:
                                if not dry_run:
                                    logging.info('Rebalancing...')
                                    account_rebalancer.rebalance(orders)
                                    account['portfolio'].refresh_balance()
                                    account['last_update'] = tz_remove_and_normalize(nm_data.last_update)
                                    retries -= 1
                                else:
                                    logging.info(f"\nProposed orders:\n{orders}")
                                    rebalancing_completed = True
                                    break
                        else:
                            print('Nothing to rebalance on this account.')
                            rebalancing_completed = True
                            break
                elif first_run:
                    logging.info(
                            'Waiting next rebalance for account "{0}" in {1} minutes.'.format(
                                    account['account_name'], ((account['last_update'] +
                                                               pd.Timedelta(account['rebalance_interval'],
                                                                            'minutes')) - pd.Timestamp('now')
                                                              ).seconds//60))
            except Exception as e:
                log_error(e)
        else:
            if rebalancing_completed:
                print("Rebalancing complete for all configured portfolios!")
                if dry_run:
                    return
                try:
                    # noinspection PyUnboundLocalVariable
                    print(f"Next rebalance in {account['rebalance_interval']} minutes.")
                except NameError:
                    pass
                print("Press Ctrl-C to exit!")
                rebalancing_completed = False
        time.sleep(60)


if __name__ == '__main__':
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    rebalance(sys.argv[1:])
