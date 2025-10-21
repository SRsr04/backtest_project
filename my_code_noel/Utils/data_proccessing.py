from pybit.unified_trading import HTTP
import time
import pandas as pd
from datetime import datetime
def get_historical_ohlc(session, symbol, interval, total_batches=5):

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
    df['time'] = pd.to_datetime(
        pd.to_numeric(df['time'], errors='coerce'),
        errors='coerce',
        unit='ms'
    )

    df = df.astype({
        'open': 'float',
        'high': 'float',
        'low': 'float',
        'close': 'float',
        'volume': 'float'})

    df = df.sort_values(by='time')
    df = df.reset_index(drop=True)

    return df

def get_historical_ohlc_by(session, symbol, interval, start_date_str, max_batches=250):
    """
    Завантажує історичні дані до вказаної дати.

    :param start_date_str: Дата у форматі "РРРР-ММ-ДД", до якої завантажувати дані.
    :param max_batches: Захист від нескінченного циклу, максимальна кількість запитів.
    """
    # --- НОВА ЛОГІКА: Конвертуємо дату в Unix timestamp (мілісекунди) ---
    start_date_ms = int(datetime.strptime(start_date_str, "%Y-%m-%d").timestamp() * 1000)

    all_data = []
    end_time = int(time.time() * 1000)

    # --- НОВА ЛОГІКА: Використовуємо цикл з лічильником для безпеки ---
    for i in range(max_batches):
        # print(f"Завантажуємо {symbol} ({interval}), запит #{i+1}...")
        candels = session.get_kline(
            category='linear',
            symbol=symbol,
            interval=interval,
            limit=500, 
            end=end_time
        )['result']['list']

        if not candels:
            print("Більше немає даних від біржі.")
            break

        all_data.extend(candels)

        candels_sorted = sorted(candels, key=lambda x: int(x[0]))
        oldest_ts_ms = int(candels_sorted[0][0])
        
        # --- НОВА ЛОГІКА: Перевіряємо, чи ми досягли потрібної дати ---
        if oldest_ts_ms <= start_date_ms:
            print("Досягнуто вказаної дати. Завершуємо завантаження.")
            break

        end_time = oldest_ts_ms - 1
        time.sleep(0.2)
    else:
        # Цей блок виконається, якщо цикл завершився через max_batches
        print(f"Попередження: Досягнуто ліміту у {max_batches} запитів.")

    # Решта коду залишається без змін
    df = pd.DataFrame([row[:6] for row in all_data], columns=['time', 'open', 'high', 'low', 'close', 'volume'])
    df['time'] = pd.to_datetime(pd.to_numeric(df['time']), unit='ms')
    df = df.astype({'open': 'float', 'high': 'float', 'low': 'float', 'close': 'float', 'volume': 'float'})
    df = df.sort_values(by='time').reset_index(drop=True)
    
    # Видаляємо дані, старші за потрібну дату, щоб не було зайвого
    df = df[df['time'] >= pd.to_datetime(start_date_str)]
    
    print(f"Завантаження ({symbol}, {interval}) завершено. Отримано {len(df)} свічок.")
    return df