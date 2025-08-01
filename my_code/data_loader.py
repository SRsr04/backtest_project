import pandas as pd
import numpy as np

RED = "\033[91m"
GREEN = "\033[92m"
RESET = "\033[0m"

def parse_kiev(x):
    """Parse date strings (or arrays) into naive Kyiv-localized Timestamps."""
    dt = pd.to_datetime(x, utc=True)
    return dt.tz_convert("Europe/Kiev").tz_localize(None)

def load_data():
    """Load and prepare all 4 CSVs."""
    m5_df = pd.read_csv(
        'm5_candels.csv',
        parse_dates=['datetime'],
        date_parser=parse_kiev
    )
    m15_df = pd.read_csv(
        'm15_candels.csv',
        parse_dates=['datetime'],
        date_parser=parse_kiev
    )
    fvg_df = pd.read_csv(
        'fvg_m15.csv',
        parse_dates=['time'],
        date_parser=parse_kiev
    )
    bos_df = pd.read_csv(
        'bos_m5.csv',
        parse_dates=['bos_time', 'bos_time_kiev'],
        date_parser=parse_kiev
    )

    # 3. Index & sort candle DataFrames
    m5_df = m5_df.set_index('datetime').sort_index()
    m15_df = m15_df.set_index('datetime').sort_index()
    # 4. Drop unused
    m15_df.drop(columns=['timestamp_utc'], inplace=True, errors='ignore')

    return m5_df, m15_df, fvg_df, bos_df

def find_fractals(df):
    """Vectorized 3-bar fractal detection."""
    highs = df['high'].to_numpy()
    lows  = df['low'].to_numpy()
    times = df.index.to_numpy()

    mask_high = (highs[1:-1] > highs[:-2]) & (highs[1:-1] > highs[2:])
    mask_low  = (lows[1:-1]  < lows[:-2]) & (lows[1:-1]  < lows[2:])

    high_idxs = np.nonzero(mask_high)[0] + 1
    low_idxs  = np.nonzero(mask_low)[0]  + 1

    all_times  = np.concatenate([times[high_idxs],  times[low_idxs]])
    all_prices = np.concatenate([highs[high_idxs], lows[low_idxs]])
    all_types  = np.concatenate([
        np.full(high_idxs.shape, 'high', dtype=object),
        np.full(low_idxs.shape,  'low',  dtype=object)
    ])

    fr = pd.DataFrame({
        'time':  all_times,
        'price': all_prices,
        'type':  all_types
    })
    return fr.sort_values('time').reset_index(drop=True)

def find_mitigation(fvg_time, fvg_max, fvg_min, fvg_type, m15_df):
    """Return (timestamp, close) of first M15 candle that mitigates the FVG."""
    start = fvg_time + pd.Timedelta(minutes=15)
    candles = m15_df.loc[start:]
    if candles.empty:
        return None

    if fvg_type == 'bullish':
        mask = candles['close'] < fvg_min
    else:
        mask = candles['close'] > fvg_max

    if not mask.any():
        return None

    # first True position
    pos = np.argmax(mask.values)
    ts  = candles.index[pos]
    price = float(candles['close'].iat[pos])
    return ts, price

def fractals_after_fvg(fvg_df, fractals_df):
    """
    Для кожного FVG знаходить перший коректний F1 та оптимальний F2 з урахуванням появи нових F1.
    Повертає список словників результатів.
    """
    results = []

    # Перетворюємо колонки на NumPy-масиви
    times = fractals_df['time'].to_numpy()
    prices = fractals_df['price'].to_numpy()
    types_ = fractals_df['type'].to_numpy()

    for _, fvg in fvg_df.iterrows():
        # Визначаємо параметри zoni
        fvg_time = np.datetime64(fvg['time']) + np.timedelta64(15, 'm')
        fvg_min = fvg['min']
        fvg_max = fvg['max']
        fvg_type = fvg['type']

        # Позиція першого фрактала після FVG + 15 хв
        start = np.searchsorted(times, fvg_time)
        if start >= len(times) - 1:
            continue

        # Зріз цін та типів від start
        slice_prices = prices[start:]
        slice_types = types_[start:]

        # Маска для кандидатів на F1
        if fvg_type == 'bullish':
            mask1 = (slice_types == 'high') & (slice_prices > fvg_max)
        else:
            mask1 = (slice_types == 'low') & (slice_prices < fvg_min)
        rels1 = np.nonzero(mask1)[0]
        if rels1.size == 0:
            continue

        # Перебір усіх rels1 у хронологічному порядку
        found = False
        for i, rel1 in enumerate(rels1):
            f1_idx = start + rel1

            # Обмежуємо діапазон пошуку F2 до наступного rel1 або кінця
            next_rel1 = rels1[i+1] if i+1 < len(rels1) else len(slice_prices)
            tail_start = f1_idx + 1
            tail_end = start + next_rel1
            tail_prices = prices[tail_start:tail_end]
            tail_times = times[tail_start:tail_end]

            if tail_prices.size == 0:
                continue

            # Маска для F2 у межах FVG
            mask2 = (tail_prices >= fvg_min) & (tail_prices <= fvg_max)
            if not mask2.any():
                continue

            # Вибір оптимального F2
            if fvg_type == 'bullish':
                rel2 = np.argmin(np.where(mask2, tail_prices, np.inf))
            else:
                rel2 = np.argmax(np.where(mask2, tail_prices, -np.inf))
            f2_idx = tail_start + rel2

            # Додаємо результат
            results.append({
                'fvg_time': pd.Timestamp(fvg_time),
                'fvg_type': fvg_type,
                'fvg_min': fvg_min,
                'fvg_max': fvg_max,
                'f1_time': pd.Timestamp(times[f1_idx]),
                'f1_price': float(prices[f1_idx]),
                'f1_type': types_[f1_idx],
                'f2_time': pd.Timestamp(times[f2_idx]),
                'f2_price': float(prices[f2_idx]),
                'f2_type': types_[f2_idx],
            })
            found = True
            break

        # якщо знайшли — переходимо до наступного FVG
        if found:
            continue

    return pd.DataFrame(results)

def add_bos_after_f2(fvg_list, bos_df, debug=False):
    """Attach BOS after F2 using NumPy masks for speed."""
    results = []
    bos_sorted = bos_df.sort_values('bos_time_kiev')
    times   = bos_sorted['bos_time_kiev'].to_numpy()
    closes  = bos_sorted['close'].to_numpy()
    types   = bos_sorted['type'].to_numpy()
    levels  = bos_sorted['level'].to_numpy()

    for item in fvg_list:
        f2ts, f1p, fvg_min, fvg_max, f1t = (
            np.datetime64(item['f2_time']),
            item['f1_price'],
            item['fvg_min'],
            item['fvg_max'],
            item['f1_type'],
        )
        start = np.searchsorted(times, f2ts)
        if start >= times.size:
            continue

        deadline = np.datetime64(item['fvg_time']) + np.timedelta64(2, 'h')
        t_sub = times[start:]
        c_sub = closes[start:]

        mask = (t_sub <= deadline) & (
            (c_sub > f1p) if f1t == 'high' else (c_sub < f1p)
        )
        if not mask.any():
            continue
        rel = np.argmax(mask)
        gi = start + rel

        item.update({
            'bos_time': times[gi],
            'bos_type': types[gi],
            'bos_price': float(levels[gi])
        })
        results.append(item)

    return results

def format_results(results):
    formatted = []
    for r in results:
        formatted.append({
            "fvg_time": pd.Timestamp(r['fvg_time']).strftime("%Y-%m-%d %H:%M"),
            "f1_time": pd.Timestamp(r['f1_time']).strftime("%Y-%m-%d %H:%M"),
            "f2_time": pd.Timestamp(r['f2_time']).strftime("%Y-%m-%d %H:%M"),
            "bos_time": pd.Timestamp(r['bos_time']).strftime("%Y-%m-%d %H:%M"),
            "f1_price": float(r['f1_price']),
            "f2_price": float(r['f2_price']),
            "bos_price": float(r['bos_price']),
            "fvg_min": float(r['fvg_min']),
            "fvg_max": float(r['fvg_max']),
            "fvg_type": r['fvg_type'],
            "bos_type": r['bos_type'],
        })
    return formatted

if __name__ == "__main__":
    m5_df, m15_df, fvg_df, bos_df = load_data()
    m5_fractals = find_fractals(m5_df)
    fvg_res     = fractals_after_fvg(fvg_df, m5_fractals)

    # --- 1) будуємо список сетапів і додаємо BOS ---
    raw_setups = fvg_res.to_dict(orient="records")
    setups     = add_bos_after_f2(raw_setups, bos_df)

    if setups:
        first = setups[0]
        print("Перший сетап:")
        print(f"  FVG    : {first['fvg_time']}  ({first['fvg_type']})  [{first['fvg_min']}–{first['fvg_max']}]")
        print(f"  F1     : {first['f1_time']}  {first['f1_type']} @ {first['f1_price']}")
        print(f"  F2     : {first['f2_time']}  {first['f2_type']} @ {first['f2_price']}")
        print(f"  BOS    : {first['bos_time']}  {first['bos_type']} @ {first['bos_price']}")
    else:
        print("Немає жодного повного сетапу F1–F2–BOS.")

    # --- 2) перевіряємо мітигації від FVG ---
    def mit_row(r):
        start = pd.Timestamp(r['fvg_time']) + pd.Timedelta(minutes=15)
        mit = find_mitigation(start, r['fvg_max'], r['fvg_min'], r['fvg_type'], m15_df)
        return mit or (pd.NaT, np.nan)

    fvg_res[['mit_time','mit_close']] = fvg_res.apply(mit_row, axis=1, result_type='expand')

    print("Мітиговано:", fvg_res['mit_time'].notna().sum())
    if fvg_res['mit_time'].notna().any():
        first_mit = fvg_res.loc[fvg_res['mit_time'].notna()].iloc[0].to_dict()
        print("Перший мітигований FVG:", first_mit)