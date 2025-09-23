# detect_h1_fvg.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import random
from math import ceil
from datetime import datetime, timedelta
from time import sleep

import pytz  # залишаємо pytz
import pandas as pd
import numpy as np
from okx.MarketData import MarketAPI

# --- Global Timezone Definition ---
try:
    from zoneinfo import ZoneInfo
    kyiv_tz = ZoneInfo('Europe/Kyiv')
except ImportError:
    kyiv_tz = pytz.timezone('Europe/Kyiv')

# --- Configuration ---
INST_ID = "BTC-USDT-SWAP"
TF = '15m'                 # 1-годинні свічки для FVG
LIMIT = 300               # максимум у OKX
DAYS_TO_FETCH = 1826      # рівно 5 календарних років (із 2024)

# Output file names reflecting H1 timeframe
CANDLES_OUTPUT_FILE = "m15_fvg_candels.csv"             # залишаю як у твоєму коді
FVG_OUTPUT_FILE     = "btc/fvg_m15.csv"

# Рейт-ліміт/стабільність
SAFE_SLEEP_EVERY = 50     # коротка пауза кожні N запитів
SAFE_SLEEP_SEC   = 2
REFRESH_CLIENT_EVERY = 1000
RETRY_MAX  = 6
RETRY_BASE = 2
RETRY_CAP  = 30

# --- Helper Function ---
def ensure_dir(path: str):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)

def from_str_ms(ms_str: str) -> datetime:
    """Конвертує рядок мітки часу в мс → datetime за UTC."""
    return datetime.fromtimestamp(int(ms_str) / 1000, tz=pytz.UTC)

# --- Candlestick Download Logic (FAST) ---
def fetch_candles_from_okx(inst_id: str, tf: str, limit: int, days_to_fetch: int) -> pd.DataFrame:
    """
    Швидке завантаження історичних свічок з OKX із ретрі, курсором і паузами.
    Повертає DataFrame з колонками: timestamp, datetime_utc, open, high, low, close, volume.
    """
    print(f"*** OKX {tf} Candles – Last {days_to_fetch // 365} Years for {inst_id} ***")

    # межа у часі: йдемо у минуле, поки не перейдемо цю дату
    min_dt = datetime.now(tz=pytz.UTC) - timedelta(days=days_to_fetch)

    # барів на добу для конкретного TF (для верхньої межі сторінок)
    bars_per_day = {
        '1s': 86400, '1m': 1440, '3m': 480, '5m': 288, '15m': 96, '30m': 48,
        '1H': 24, '2H': 12, '4H': 6, '6H': 4, '12H': 2, '1D': 1
    }.get(tf, 24)

    pages_needed = ceil(days_to_fetch * bars_per_day / limit) + 50  # невеликий буфер

    def fetch_with_retry(cl, **kwargs):
        for attempt in range(1, RETRY_MAX + 1):
            try:
                r = cl.get_history_candlesticks(**kwargs)
                return r.get("data", [])
            except Exception as e:
                delay = min(RETRY_BASE * (2 ** (attempt - 1)), RETRY_CAP)
                delay += random.uniform(0, 0.5 * delay)  # джиттер
                print(f"[WARN] {e}. retry {attempt}/{RETRY_MAX} in {delay:.1f}s...")
                sleep(delay)
                # На парних спробах — пересоздаємо клієнт
                if attempt % 2 == 0:
                    try:
                        cl = MarketAPI(flag="0", debug=False)
                        print("[INFO] Клієнт пересоздано.")
                    except Exception as e2:
                        print(f"[WARN] Не вдалося пересоздати клієнт: {e2}")
        raise RuntimeError("Вичерпано ретрі для get_history_candlesticks")

    records = []
    seen_ts = set()
    cursor = None  # рух у минуле через after=min_ts з попередньої пачки

    cl = MarketAPI(flag="0", debug=False)

    for page in range(1, pages_needed + 1):
        if page % REFRESH_CLIENT_EVERY == 0:
            try:
                cl = MarketAPI(flag="0", debug=False)
                print(f"[INFO] REFRESH_CLIENT_EVERY: пересоздано клієнт на сторінці {page}.")
            except Exception as e:
                print(f"[WARN] Не вдалося пересоздати клієнт: {e}")

        kwargs = dict(instId=inst_id, bar=tf, limit=limit)
        if cursor is not None:
            kwargs["after"] = str(cursor)  # дає старіші свічки

        data = fetch_with_retry(cl, **kwargs)
        if not data:
            print("No more data received. Stopping.")
            break

        min_ts_in_batch = None
        hit_time_boundary = False

        for c in data:
            ts = int(c[0])
            dt = from_str_ms(c[0])

            if min_ts_in_batch is None or ts < min_ts_in_batch:
                min_ts_in_batch = ts

            if dt < min_dt:
                hit_time_boundary = True
                continue

            if ts in seen_ts:
                continue
            seen_ts.add(ts)

            records.append({
                'timestamp': ts,          # мс
                'datetime_utc': dt,
                'open':  float(c[1]),
                'high':  float(c[2]),
                'low':   float(c[3]),
                'close': float(c[4]),
                'volume': float(c[7]),
            })

        if min_ts_in_batch is None or (cursor is not None and min_ts_in_batch >= cursor):
            print("Пагінація не рухається в минуле. Зупинка.")
            break
        cursor = min_ts_in_batch

        if hit_time_boundary:
            print(f"Досягнуто межі {min_dt.date()}. Зупинка.")
            break

        if len(data) < limit:
            print(f"Finished fetching data. Received {len(data)} (< {limit}) у останній пачці.")
            break

        if page % SAFE_SLEEP_EVERY == 0:
            print(f"Page {page} processed, short pause...")
            sleep(SAFE_SLEEP_SEC)

    if not records:
        print("Fetched 0 candles.")
        return pd.DataFrame()

    df = pd.DataFrame(records)
    print(f"\nFetched {len(df)} candles.")
    return df

# --- FVG Detection Logic (як у тебе) ---
def detect_fvg(df_candles_slice: pd.DataFrame) -> dict | None:
    if len(df_candles_slice) < 3:
        return None

    c0 = df_candles_slice.iloc[0]
    c1 = df_candles_slice.iloc[1]
    c2 = df_candles_slice.iloc[2]

    bullish = (c2['low']  > c0['high'])
    bearish = (c2['high'] < c0['low'])

    if bullish:
        return {
            'type': 'bullish',
            'min': c0['high'],
            'max': c2['low'],
            'time': c2['timestamp']
        }
    if bearish:
        return {
            'type': 'bearish',
            'min': c2['high'],
            'max': c0['low'],
            'time': c2['timestamp']
        }
    return None

def detect_fvgs_from_dataframe(df: pd.DataFrame) -> list[dict]:
    df = df.sort_values(by='timestamp', ascending=True)
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df = df.set_index('datetime')

    detected = []
    for i in range(len(df) - 2):
        sl = df.iloc[i:i+3]
        fvg = detect_fvg(sl)
        if fvg:
            detected.append({
                'time':   pd.to_datetime(fvg['time'], unit='ms', utc=True),
                'type':   fvg['type'],
                'min':    float(fvg['min']),
                'max':    float(fvg['max']),
                'middle': (float(fvg['min']) + float(fvg['max'])) / 2.0,
            })
    return detected

# --- Main execution flow ---
def main_fvg_script():
    print(f"Running FVG detection script for {INST_ID} on {TF} over {DAYS_TO_FETCH // 365} years.")

    # 1) Fetch H1 candles fast
    df_candles = fetch_candles_from_okx(INST_ID, TF, LIMIT, DAYS_TO_FETCH)
    if df_candles.empty:
        print("No candlestick data to process. Exiting.")
        return

    # Дедуп + впорядкування, збереження сирих свічок (за потреби)
    df_candles = df_candles.drop_duplicates(subset=['timestamp']).sort_values('timestamp')
    # (опц.) збереження сирих свічок:
    # ensure_dir(CANDLES_OUTPUT_FILE)
    # df_candles.to_csv(CANDLES_OUTPUT_FILE, index=False)
    # print(f"\nSaved {len(df_candles)} H1 candles to: {CANDLES_OUTPUT_FILE}")

    # 2) Detect FVGs
    print("\nDetecting FVGs...")
    fvgs_list = detect_fvgs_from_dataframe(df_candles)
    print(f"Знайдено {len(fvgs_list)} FVG")

    if not fvgs_list:
        print("No FVGs detected.")
        return

    df_fvgs = pd.DataFrame(fvgs_list)
    df_fvgs['time'] = df_fvgs['time'].dt.tz_convert(kyiv_tz)

    # Сортування: від нових до старих
    df_fvgs = df_fvgs.drop_duplicates(keep='first', subset=['time','type','min','max'])
    df_fvgs = df_fvgs.sort_values(by='time', ascending=False)

    ensure_dir(FVG_OUTPUT_FILE)
    df_fvgs.to_csv(FVG_OUTPUT_FILE, index=False)
    print(f"💾 Збережено {len(df_fvgs)} FVG у файл {FVG_OUTPUT_FILE}")

if __name__ == "__main__":
    main_fvg_script()
