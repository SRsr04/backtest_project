import pandas as pd
from pybit.unified_trading import HTTP
from constants import API_KEY, API_SECRET
import time

def get_historical_ohlc(session, symbol, interval, total_batches=10):

    all_data = []
    end_time = int(time.time() * 1000)

    for _ in range(total_batches):
        candels = session.get_kline(
            category = 'linear',
            symbol = symbol,
            interval = interval,
            limit = 500, 
            end = end_time
        )['result']['list']

        if not candels:
            break

        all_data.extend(candels)

        candels = sorted(candels, key=lambda x: int(x[0]))
        oldest = int(candels[0][0])
        end_time = oldest - 1
        time.sleep(0.2)

    df = pd.DataFrame([row[:6] for row in all_data], columns=['time', 'open', 'high', 'low', 'close', 'volume'])
    df['time'] = pd.to_datetime(df['time'], errors = 'coerce', unit='ms')

    df = df.astype({
        'open': 'float',
        'high': 'float',
        'low': 'float',
        'close': 'float',
        'volume': 'float'})

    df = df.sort_values(by='time')
    df = df.reset_index(drop=True)

    return df