import pandas as pd
import numpy as np

def find_mitigation_v2(fvg_time, fvg_max, fvg_min, fvg_type, m15_df, *, eps=0.0, m15_index_is_close=True):
    """
    Шукає першу M15-свічку, що повертає ціну в межі FVG, та повертає часові позначки
    її відкриття й закриття разом із ціною закриття.

    Повертає словник із ключами "open", "close", "price" або None, якщо мітигації не знайдено.
    """
    z_low, z_high = (min(fvg_min, fvg_max), max(fvg_min, fvg_max))
    start = fvg_time + pd.Timedelta(minutes=15)  # якщо fvg_time – close FVG-бару
    candles = m15_df.loc[start:]
    if candles.empty:
        return None  # no mitigation

    if fvg_type == 'bullish':
        mask = (candles['close'] <= z_low + eps)
    else:
        mask = (candles['close'] >= z_high - eps)

    if not mask.any():
        return None

    pos = int(np.argmax(mask.values))
    ts_close = candles.index[pos]
    price    = float(candles['close'].iat[pos])

    ts_open  = (pd.to_datetime(ts_close) - pd.Timedelta(minutes=15)) if m15_index_is_close \
               else pd.to_datetime(ts_close).floor("15min")
    # нормалізуємо до naive
    try: ts_open  = ts_open.tz_localize(None)
    except: pass
    try: ts_close = ts_close.tz_localize(None)
    except: pass

    return {"open": ts_open, "close": ts_close, "price": price}
