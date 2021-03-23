#!/usr/bin/python3
import sys
import time
import getopt
import logging
import pandas as pd
from nm import NMData, Portfolio
from config import nm_url
from config import RETRIES as DEFAULT_RETRIES
from config import accounts as account_config


def rebalance(argv):
    force_first_rebalance = False
    dry_run = False
    opts, args = getopt.getopt(argv, "dhf", ["dry_run",  "help", "force"])
    for opt, arg in opts:
        if opt == '-h':
            print('Usage: rebalance [--force] --dry_run')
            quit()
        elif opt in ('-f', '--force'):
            force_first_rebalance = True
        elif opt in ('-d', '--dry_run'):
            dry_run = True
            force_first_rebalance = True

    nm_data = NMData(nm_url)
    retries = DEFAULT_RETRIES
    while retries > 0:
        try:
            nm_data.get_nm_data()
            break
        except Exception as e:
            logging.error(e)
            retries -= 1
    else:
        logging.error('Unable to read NM index table. Aborting...')
        quit(-1)
    logging.info('Connecting to configured Binance account(s)...')
    for account in account_config:
        account['portfolio'] = Portfolio(config=account)
        account['last_update'] = nm_data.last_update.tz_convert('utc')
        account['force_first_rebalance'] = force_first_rebalance
    rebalancing_completed = False
    first_run = True
    while True:
        for account in account_config:
            try:
                if pd.Timestamp.now('utc') - account['last_update'] > pd.Timedelta(
                        account['rebalance_interval'], 'minutes') or account['force_first_rebalance']:
                    logging.info(f"Rebalancing account: {account['account_name']}")
                    first_run = False
                    rebalancing_completed = False
                    account['force_first_rebalance'] = False
                    logging.info('Getting new NM data...')
                    nm_retries = DEFAULT_RETRIES
                    while nm_retries > 0:
                        try:
                            nm_data.get_nm_data()
                            break
                        except Exception as e:
                            logging.error(e)
                            nm_retries -= 1
                    target = nm_data.get(account['index'])[:account['top_n']].index.values
                    logging.info(f"Target NM data for NM{account['index']}: {target}.")
                    retries = DEFAULT_RETRIES
                    while retries > 0:
                        logging.info(f'Setting up orders for rebalancing. Attempt # {DEFAULT_RETRIES - retries + 1}')
                        orders = account['portfolio'].rebalanced_portfolio_proposal(target)
                        if len(orders) < 1:
                            logging.info(f"No more orders, NM{account['index']} portfolio already rebalanced!!!")
                            print(f"\nCurrent balance:\n{account['portfolio'].balance}\n")
                            rebalancing_completed = True
                            break
                        else:
                            if not dry_run:
                                logging.info('Rebalancing...')
                                account['portfolio'].rebalance(orders)
                                account['portfolio'].refresh_balance()
                                account['last_update'] = nm_data.last_update.tz_convert('utc')
                                retries -= 1
                            else:
                                logging.info(f"\nProposed orders:\n{orders}")
                                rebalancing_completed = True
                                break
                elif first_run:
                    logging.info('Waiting next rebalance in {0} minutes.'.format(
                                 ((account['last_update'] + pd.Timedelta(account['rebalance_interval'], 'minutes'))
                                  - pd.Timestamp.now('utc')).seconds//60))
            except Exception as e:
                logging.error(e)
        else:
            if rebalancing_completed:
                print("Rebalancing complete for all configured portfolios!")
                if dry_run:
                    quit(0)
                print(f"Next rebalance in {account['rebalance_interval']} minutes.")
                print("Press Ctrl-C to exit!")
                rebalancing_completed = False
        time.sleep(60)


if __name__ == '__main__':
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    rebalance(sys.argv[1:])
