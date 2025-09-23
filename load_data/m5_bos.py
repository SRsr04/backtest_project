# detect_m15_bos.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import random
from math import ceil
from datetime import datetime, timedelta
from time import sleep

import pytz
import pandas as pd
import numpy as np
from okx.MarketData import MarketAPI

# --- Timezone (Kyiv) ---
try:
    from zoneinfo import ZoneInfo
    kyiv_tz = ZoneInfo('Europe/Kyiv')
except ImportError:
    kyiv_tz = pytz.timezone('Europe/Kyiv')

# --- Config ---
INST_ID = "BTC-USDT-SWAP"
TF = '5m'                 # свічки для BOS
LIMIT = 300                # максимум у OKX
DAYS_TO_FETCH = 1826       # рівно 5 календарних років (включно з 2024)

# Вивід
CANDLES_OUTPUT_FILE = "btc/m5_candles.csv"
BOS_OUTPUT_FILE     = "btc/bos_m5.csv"

# Рейт-ліміт/стабільність
SAFE_SLEEP_EVERY = 50      # пауза кожні N запитів
SAFE_SLEEP_SEC   = 2
REFRESH_CLIENT_EVERY = 1000
RETRY_MAX  = 6
RETRY_BASE = 2
RETRY_CAP  = 30

# --- Utils ---
def ensure_dir(path: str):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)

def from_str_ms(ms_str: str) -> datetime:
    return datetime.fromtimestamp(int(ms_str) / 1000, tz=pytz.UTC)

def fetch_with_retry(cl, **kwargs):
    """OKX виклик із ретрі та експоненційним бекофом."""
    for attempt in range(1, RETRY_MAX + 1):
        try:
            r = cl.get_history_candlesticks(**kwargs)
            return r.get("data", [])
        except Exception as e:
            delay = min(RETRY_BASE * (2 ** (attempt - 1)), RETRY_CAP)
            delay += random.uniform(0, 0.5 * delay)
            print(f"[WARN] {e}. retry {attempt}/{RETRY_MAX} in {delay:.1f}s...")
            sleep(delay)
            if attempt % 2 == 0:
                try:
                    cl = MarketAPI(flag="0", debug=False)
                    print("[INFO] Клієнт пересоздано.")
                except Exception as e2:
                    print(f"[WARN] Не вдалося пересоздати клієнт: {e2}")
    raise RuntimeError("Вичерпано ретрі для get_history_candlesticks")

# --- Candlestick downloader (fast) ---
def fetch_candles_from_okx(inst_id: str, tf: str, limit: int, days_to_fetch: int) -> pd.DataFrame:
    print(f"*** OKX {tf} Candles – Last {days_to_fetch // 365} Years for {inst_id} ***")

    # Межа за часом (йдемо у минуле, поки не перейдемо цю дату)
    min_dt = datetime.now(tz=pytz.UTC) - timedelta(days=days_to_fetch)

    # Для оцінки верхньої межі сторінок: барів/день для TF
    bars_per_day = {
        '1m': 1440, '3m': 480, '5m': 288, '15m': 96, '30m': 48,
        '1H': 24, '2H': 12, '4H': 6, '6H': 4, '12H': 2, '1D': 1
    }.get(tf, 96)  # дефолт 96 (15m)

    pages_needed = ceil(days_to_fetch * bars_per_day / limit) + 50  # невеликий буфер

    records = []
    seen_ts = set()
    cursor = None  # рух у минуле через after=min_ts попередньої пачки

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

            # Стоп по часовій межі
            if dt < min_dt:
                hit_time_boundary = True
                continue

            if ts in seen_ts:
                continue
            seen_ts.add(ts)

            records.append({
                'timestamp_utc': ts,
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

# --- BOS detection ---
def detect_bos_based_on_fractals(df: pd.DataFrame) -> list[dict]:
    """
    Вхід: df із колонками timestamp_utc, high, low, close.
    Алгоритм: стандартні 3-свічкові фрактали; BOS при пробитті рівня фрактала,
    перевірка і high/low, і close.
    """
    # Сортуємо з минулого в майбутнє
    df = df.sort_values(by='timestamp_utc', ascending=True)
    highs = df['high'].to_numpy(dtype=float)
    lows  = df['low'].to_numpy(dtype=float)
    closes = df['close'].to_numpy(dtype=float)

    timestamps = pd.to_datetime(df['timestamp_utc'], unit='ms', utc=True).to_numpy()
    bos_list = []
    n = len(df)
    if n < 3:
        return bos_list

    for i in range(1, n - 1):
        # Bullish fractal
        if highs[i] > highs[i - 1] and highs[i] > highs[i + 1]:
            level = highs[i]
            mask = (highs[(i+2):] > level) & (closes[(i+2):] > level)
            idx = np.where(mask)[0]
            if idx.size > 0:
                j = idx[0] + i + 2
                bos_list.append({
                    'type': 'bullish',
                    'fract_time': timestamps[i],
                    'bos_time': timestamps[j],
                    'level': float(level),
                    'close': float(closes[j])
                })

        # Bearish fractal
        if lows[i] < lows[i - 1] and lows[i] < lows[i + 1]:
            level = lows[i]
            mask = (lows[(i+2):] < level) & (closes[(i+2):] < level)
            idx = np.where(mask)[0]
            if idx.size > 0:
                j = idx[0] + i + 2
                bos_list.append({
                    'type': 'bearish',
                    'fract_time': timestamps[i],
                    'bos_time': timestamps[j],
                    'level': float(level),
                    'close': float(closes[j])
                })

    return bos_list

# --- Main ---
def main_bos_script():
    print(f"Running BOS detection for {INST_ID} on {TF} over ~{DAYS_TO_FETCH // 365} years.")

    # 1) Download M15 candles fast
    df_candles = fetch_candles_from_okx(INST_ID, TF, LIMIT, DAYS_TO_FETCH)
    if df_candles.empty:
        print("No candlestick data to process. Exiting.")
        return

    # Конвертація часу в Київ + збереження сирих свічок (опційно/швидко)
    df_candles["datetime"] = pd.to_datetime(df_candles["datetime_utc"], utc=True).dt.tz_convert(kyiv_tz)
    df_candles = df_candles.drop(columns=["datetime_utc"])
    df_candles = df_candles.sort_values(by="timestamp_utc", ascending=True)
    ensure_dir(CANDLES_OUTPUT_FILE)
    df_candles.to_csv(CANDLES_OUTPUT_FILE, index=False)
    print(f"💾 Saved {len(df_candles)} M15 candles to: {CANDLES_OUTPUT_FILE}")

    # 2) Detect BOS
    print("\nDetecting BOS...")
    bos_list = detect_bos_based_on_fractals(df_candles)
    print(f"Знайдено {len(bos_list)} BOS за період.")

    if not bos_list:
        print("⚠️ BOS не знайдено")
        return

    bos_df = pd.DataFrame(bos_list)
    bos_df['bos_time']   = pd.to_datetime(bos_df['bos_time'],   utc=True, errors='coerce')
    bos_df['fract_time'] = pd.to_datetime(bos_df['fract_time'], utc=True, errors='coerce')

    bos_df['bos_time_kiev']   = bos_df['bos_time'].dt.tz_convert(kyiv_tz)
    bos_df['fract_time_kiev'] = bos_df['fract_time'].dt.tz_convert(kyiv_tz)

    # Унікальність і порядок
    bos_df = bos_df.drop_duplicates(subset=['bos_time','fract_time','type','level'], keep='first')
    bos_df = bos_df.sort_values(by='bos_time', ascending=False)

    # Форматований прів'ю
    def format_bos_row(row):
        return (f"{row['bos_time_kiev'].strftime('%d.%m %H:%M')} | {row['type']} BOS | "
                f"fractal @ {row['fract_time_kiev'].strftime('%H:%M')} | "
                f"level: {row['level']:.2f}, close: {row['close']:.2f}")

    bos_df['formatted'] = bos_df.apply(format_bos_row, axis=1)

    output_columns = [
        'bos_time', 'bos_time_kiev', 'type',
        'fract_time', 'fract_time_kiev',
        'level', 'close'
    ]
    ensure_dir(BOS_OUTPUT_FILE)
    bos_df[output_columns].to_csv(BOS_OUTPUT_FILE, index=False)

    print(f"💾 Збережено {len(bos_df)} BOS у файл {BOS_OUTPUT_FILE}.\n")
    print("Приклад останніх 10 BOS:")
    for line in bos_df['formatted'].head(10):
        print(line)

if __name__ == "__main__":
    main_bos_script()