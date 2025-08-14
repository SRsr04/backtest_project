import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import logging
import itertools
import os
import sys
from sweep import sweep

logging.basicConfig(
    filename='trade_search.log',
    filemode='w',                # перезаписувати файл при кожному запуску
    level=logging.INFO,
    format='%(asctime)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def strip_offset_keep_wall_time(s: pd.Series) -> pd.Series:
    """
    Перетворює будь-який мікс дат (з офсетами типу +03:00/Z і без) у наївний час.
    Якщо елемент має tz — просто знімаємо tz, ЗБЕРІГАЮЧИ локальний wall time.
    Якщо елемент уже наївний — лишаємо як є.
    Непарсибельні значення -> NaT.
    """
    def _one(x):
        if pd.isna(x):
            return pd.NaT
        ts = pd.to_datetime(x, errors='coerce')  # парсимо один елемент
        if pd.isna(ts):
            return pd.NaT
        # ts — це pd.Timestamp
        if ts.tz is not None:
            # важливо: знімаємо tz БЕЗ конвертації, щоб зберегти wall time
            return ts.tz_localize(None)
        return ts

    # Якщо колонка вже чистий datetime64[ns] → просто повертаємо її
    if pd.api.types.is_datetime64_ns_dtype(s.dtype):
        return s

    # Якщо вся серія однаково tz-aware → знімаємо tz векторно
    if isinstance(getattr(s.dtype, "tz", None), (type,)) or str(s.dtype).startswith("datetime64[ns,"):
        return pd.to_datetime(s, errors='coerce').dt.tz_localize(None)

    # Інакше мікс → елементно (прибирає і FutureWarning, і .dt-issue)
    return s.apply(_one)

def load_data():

    m5_df  = pd.read_csv('m5_candles.csv')
    m15_df = pd.read_csv('m15_candels.csv')
    # m1_df  = pd.read_csv('m1_candels.csv')
    h1_df  = pd.read_csv('h1_candels.csv')
    h4_df  = pd.read_csv('h4_candels.csv')
    fvg_df = pd.read_csv('fvg_m15.csv')
    bos_df = pd.read_csv('bos_m5.csv')

    # Конвертуємо тільки datetime-колонки
    for df, cols in [
        (m5_df,  ['datetime']),
        (m15_df, ['datetime']),
        (h4_df,  ['datetime']),
        (h1_df,  ['datetime']),
        (fvg_df, ['time']),
        (bos_df, ['bos_time_kiev']),
    ]:
        for c in cols:
            df[c] = strip_offset_keep_wall_time(df[c])

    # bos_time_kiev -> bos_time
    bos_df.drop(columns=['bos_time'], errors='ignore', inplace=True)
    bos_df.rename(columns={'bos_time_kiev': 'bos_time'}, inplace=True)

    m5_df = m5_df.set_index('datetime');  m5_df.sort_index(inplace=True)
    m15_df = m15_df.set_index('datetime'); m15_df.sort_index(inplace=True)
    h1_df = h1_df.set_index('datetime');  h1_df.sort_index(inplace=True)
    h4_df = h4_df.set_index('datetime');  h4_df.sort_index(inplace=True)
    bos_df.sort_values('bos_time', inplace=True)
    fvg_df = fvg_df.sort_values('time')

    sweep_h1 = sweep(h1_df)
    sweep_h4 = sweep(h4_df)


    return m5_df, m15_df, fvg_df, bos_df, h1_df, h4_df, sweep_h1, sweep_h4
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
    EPS = 1e-9

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
            mask1 = (slice_types == 'high') & (slice_prices >= fvg_min - EPS)
        else:
            mask1 = (slice_types == 'low') & (slice_prices <= fvg_max + EPS)
        
        rels1 = np.nonzero(mask1)[0]
        if rels1.size == 0:
            continue

        # Перебір усіх rels1 у хронологічному порядку
        found = False
        for i, rel1 in enumerate(rels1):
            f1_idx = start + rel1
            f1_in_fvg = (fvg_min - EPS <= prices[f1_idx] <= fvg_max + EPS)

            # Обмежуємо діапазон пошуку F2 до наступного rel1 або кінця
            next_rel1 = rels1[i+1] if i+1 < len(rels1) else len(slice_prices)
            tail_start = f1_idx + 1
            tail_end = start + next_rel1
            tail_prices = prices[tail_start:tail_end]
            tail_types  = types_[tail_start:tail_end]

            if tail_prices.size == 0:
                continue

            mask2_base = (tail_types == ('low' if fvg_type == 'bullish' else 'high'))

            inside_fvg = (tail_prices >= fvg_min - EPS) & (tail_prices <= fvg_max + EPS)
            cand_in = mask2_base & inside_fvg

            if cand_in.any():
                if fvg_type == 'bullish':
                    rel2 = np.argmin(np.where(cand_in, tail_prices, np.inf))
                else:
                    rel2 = np.argmax(np.where(cand_in, tail_prices, -np.inf))
            else:
                if fvg_type == 'bullish':
                    allowed = mask2_base & (tail_prices <= fvg_max + EPS)
                    if not allowed.any():
                        continue
                    rel2 = np.argmin(np.where(allowed, tail_prices, np.inf))
                else:
                    allowed = mask2_base & (tail_prices >= fvg_min - EPS)
                    if not allowed.any():
                        continue
                    rel2 = np.argmax(np.where(allowed, tail_prices, -np.inf))

            # 4) Індекс у глобальному масиві
            f2_idx = tail_start + rel2

            f2_in_fvg = (fvg_min - EPS <= prices[f2_idx] <= fvg_max + EPS)

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
                'f1_in_fvg': bool(f1_in_fvg),
                'f2_in_fvg': bool(f2_in_fvg),
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
    bos_sorted = bos_df.sort_values('bos_time')
    times   = bos_sorted['bos_time'].to_numpy()
    closes  = bos_sorted['close'].to_numpy()
    types   = bos_sorted['type'].to_numpy()
    levels  = bos_sorted['level'].to_numpy()
    latest = {}

    for item in fvg_list:
        f1ts = np.datetime64(item['f1_time'])
        key = item['fvg_time']
        if (key not in latest) or (f1ts > latest[key]['f1ts']):
        # клон item + додаємо поле f1ts для майбутнього порівняння
            new = item.copy()
            new['f1ts'] = f1ts
            latest[key] = new
        
    for item in latest.values():
        f2ts, f1p,   f1t = (
            np.datetime64(item['f2_time']),
            item['f1_price'],
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

def simulate_entry(setups, m5_df, sweep_h1, sweep_h4, *, fib_level, stop_offset, rr, balance, be_multiplier=0, entry_timeout_candles=None,):
    results = []
    lose_streak = 0
    last_loss_date = None
    EPS = 1e-9
    entry_durations = []  # хвилини від початку пошуку (search_start) до фактичного філу на M1
    def finish(outcome_, t_, px_):
            results.append({
                'direction': direction,
                'outcome': outcome_,
                'fvg_time': setup.get('fvg_time'),
                'f1_price': f1p,
                'f2_price': f2p,
                'fib_level': fib_level,
                'limit_placed_time': bos_time,    
                'entry_time_m5': entry_time,                  
                'search_start_m5': search_start,      
                'entry_price': entry_price,
                'stop': initial_stop,
                'take': take_price,
                'bos_time': setup.get('bos_time'),
                'exit_time': t_,
                'exit_price': px_,
                'be_triggered': (outcome_ == 'be'),
                'be_price': be_stop,
                'after_h1_sweep': after_h1_sweep,
                'after_h4_sweep': after_h4_sweep,
            })

    for setup in setups:

        bos_time = pd.to_datetime(setup['bos_time'])

        last_h1_sweep = None
        last_h4_sweep = None

        for e in sweep_h1["low_to_high"]["raw_events"] + sweep_h1["high_to_low"]["raw_events"]:
            if e["end"] <= bos_time:
                if last_h1_sweep is None or e["end"] > last_h1_sweep["end"]:
                    last_h1_sweep = e

        for e in sweep_h4["low_to_high"]["raw_events"] + sweep_h4["high_to_low"]["raw_events"]:
            if e["end"] <= bos_time:
                if last_h4_sweep is None or e["end"] > last_h4_sweep["end"]:
                    last_h4_sweep = e

        after_h1_sweep = (last_h1_sweep is not None) and (bos_time > last_h1_sweep["end"])
        after_h4_sweep = (last_h4_sweep is not None) and (bos_time > last_h4_sweep["end"])
        bos_bar_start = bos_time.floor('5min')
        search_start  = bos_bar_start + pd.Timedelta(minutes=5)
        search_end = pd.to_datetime(setup.get('search_end', bos_time + pd.Timedelta(hours=8)))
        if search_start >= search_end:
            continue

        # compute prices
        f1p, f2p = float(setup['f1_price']), float(setup['f2_price'])
        if setup['f1_type'] == 'high':
            direction = 'long'
            entry_price = f2p + (f1p - f2p) * fib_level
            initial_stop = f2p - stop_offset
        else:
            direction = 'short'
            entry_price = f2p - (f2p - f1p) * fib_level
            initial_stop = f2p + stop_offset
        distance = abs(entry_price - initial_stop)
        take_price = entry_price + distance * rr if direction == 'long' else entry_price - distance * rr

        if distance <= EPS:
            continue

        risk_usd = balance * 0.01
        position_size = risk_usd / distance if distance > 0 else np.nan
        commission = position_size * entry_price * (0.000325)
        delta_p = commission / position_size if position_size else 0.0

        be_stop = None
        be_activation = None
        if be_multiplier > 0:
            be_activation = entry_price + distance * be_multiplier if direction == 'long' else entry_price - distance * be_multiplier
            be_stop = entry_price + delta_p if direction == 'long' else entry_price - delta_p

        entry_deadline = min(search_end, search_start + pd.Timedelta(minutes = 120))
        window = m5_df.loc[search_start:entry_deadline]
        if window.empty:
            continue

        if entry_timeout_candles is not None:
            window = window.iloc[: entry_timeout_candles + 1]

        lows = window['low'].to_numpy()
        highs = window['high'].to_numpy()
        times = window.index.to_numpy()

        entry_mask = (lows <= entry_price + EPS) if direction == 'long' else (highs >= entry_price - EPS)

        if not entry_mask.any(): continue

        idx_entry = int(np.argmax(entry_mask)) if entry_mask.any() else np.inf

        entry_time = pd.Timestamp(times[idx_entry])
        entry_hour = entry_time.hour
        if entry_hour < 7 or entry_hour > 23:
            continue

        pre_end = entry_time - pd.Timedelta(minutes=5)
        if pre_end > search_start:
            pre_window = m5_df.loc[search_start:pre_end]
            touched_take = (pre_window['high'] >= take_price - EPS).any() if direction == 'long' else (pre_window['low'] <= take_price + EPS).any()
            

            if touched_take:
                continue

        entry_durations.append(int((entry_time - search_start) / pd.Timedelta(minutes=1)))


        post = m5_df.loc[entry_time:m5_df.index.max()]
        if post.empty:
            continue

        entry_row = post.iloc[0]
        is_long = (direction == 'long')

        be_active_from_next_bar = False
        lows_p  = post['low'].to_numpy()
        highs_p = post['high'].to_numpy()
        closes_p= post['close'].to_numpy()
        times_p = post.index.to_numpy()


        hit_take_0 = (entry_row['high'] >= take_price - EPS) if direction == 'long' else (entry_row['low'] <= take_price + EPS)
        hit_stop_0 = (entry_row['low'] <= initial_stop + EPS) if direction == 'long' else (entry_row['high'] >= initial_stop - EPS)

        if hit_take_0 and hit_stop_0:
            finish('stop', entry_time, initial_stop); 
            continue
        elif hit_stop_0:
            finish('stop', entry_time, initial_stop); 
            continue
        elif hit_take_0:
            finish('take', entry_time, take_price); 
            continue
        if (not be_active_from_next_bar) and (be_activation is not None):
                be_hit_close = (entry_row['close'] >= be_activation - EPS) if is_long else (entry_row['close'] <= be_activation + EPS)
                if be_hit_close:
                    be_active_from_next_bar = True

        lows1, highs1, closes1, times1 = lows_p[1:], highs_p[1:], closes_p[1:], times_p[1:]
        if times1.size == 0:
            # не фіксуємо таймаут — просто пропускаємо сетап
            continue

        closed = False
        for i in range(len(times1)):
            current_stop = be_stop if be_active_from_next_bar else initial_stop
            low_i, high_i, close_i, t_i = lows1[i], highs1[i], closes1[i], times1[i]

            hit_stop_i = (low_i <= current_stop + EPS) if is_long else (high_i >= current_stop - EPS)
            hit_take_i = (high_i >= take_price - EPS)  if is_long else (low_i  <= take_price + EPS)

            if hit_stop_i and hit_take_i:
                finish('stop' if current_stop == initial_stop else 'be', pd.Timestamp(t_i), current_stop); closed = True; break
            elif hit_stop_i:
                finish('stop' if current_stop == initial_stop else 'be', pd.Timestamp(t_i), current_stop); closed = True; break
            elif hit_take_i:
                finish('take', pd.Timestamp(t_i), take_price); closed = True; break
            if (not be_active_from_next_bar) and (be_activation is not None):
                be_hit_close = (close_i >= be_activation - EPS) if is_long else (close_i <= be_activation + EPS)
                if be_hit_close:
                    be_active_from_next_bar = True
        
        if not closed:
            # не фіксуємо таймаут — просто пропускаємо сетап
            pass

    # Звіт щодо часу до входу (по тим сетапам, де вхід відбувся)
    if entry_durations:
        avg_min = float(np.mean(entry_durations))
        med_min = float(np.median(entry_durations))
        cnt     = len(entry_durations)
        msg = f"Середній час до відкриття: {avg_min:.2f} хв; медіана: {med_min:.2f} хв; n={cnt}"
        print(msg)
        try:
            logging.info(msg)
        except Exception:
            pass
    else:
        print("Жодна угода не відкрилась у вікні пошуку — немає що усереднювати")

    return results
    
if __name__ == "__main__":

    results_dir = "backtest_results"
    os.makedirs(results_dir, exist_ok=True)

    # 1) Дані + свіпи
    m5_df, m15_df, fvg_df, bos_df, h1_df, h4_df, sweep_h1, sweep_h4 = load_data()

    # Обрізаємо всі таймфрейми під M5-інтервал
    m5_min, m5_max = m5_df.index.min(), m5_df.index.max()
    start, end = m5_min, m5_max
    m5_df  = m5_df.loc[start:end]
    m15_df = m15_df.loc[start:end]
    h1_df  = h1_df.loc[start:end]
    h4_df  = h4_df.loc[start:end]

    # 2) Формуємо сетапи
    m5_fractals = find_fractals(m5_df)
    fvg_res     = fractals_after_fvg(fvg_df, m5_fractals)
    raw_setups  = fvg_res.to_dict(orient="records")
    setups      = add_bos_after_f2(raw_setups, bos_df)

    setups = [
        s for s in setups
        if (m5_min - pd.Timedelta(minutes=5)) <= pd.to_datetime(s["bos_time"]) <= (m5_max - pd.Timedelta(minutes=5))
    ]

    # 2.1) Фільтр мітигацій ДО BOS + дедлайн пошуку BOS
    H_MAX = 8
    clean_setups = []
    for s in setups:
        fvg_time = pd.to_datetime(s["fvg_time"])
        bos_time = pd.to_datetime(s["bos_time"])

        mit = find_mitigation(
            fvg_time, float(s["fvg_max"]), float(s["fvg_min"]), s["fvg_type"], m15_df
        )

        if mit is None:
            search_end = fvg_time + pd.Timedelta(hours=H_MAX)
            s["mitigated"] = False
            s["mit_ts"] = None
        else:
            mit_ts, _mit_price = mit
            if mit_ts <= bos_time:
                continue
            search_end = mit_ts
            s["mitigated"] = True
            s["mit_ts"] = mit_ts

        search_end = min(search_end, m5_max)
        if bos_time > search_end:
            continue
        if bos_time.floor('5min') + pd.Timedelta(minutes=5) >= search_end:
            continue

        s["search_end"] = search_end
        clean_setups.append(s)

    if not clean_setups:
        print("No setups in M5 coverage window")
        sys.exit(0)

    # 2.2) Маркуємо після якого свіпа був сетап
    def mark_sweep_relation(setups, sweep_data, tf_name):
        events = sweep_data["low_to_high"]["raw_events"] + sweep_data["high_to_low"]["raw_events"]
        for s in setups:
            bos_time = pd.to_datetime(s["bos_time"])
            last_sweep = max((e for e in events if e["end"] <= bos_time), key=lambda e: e["end"], default=None)
            s[f"after_{tf_name}_sweep"] = last_sweep is not None
        return setups

    clean_setups = mark_sweep_relation(clean_setups, sweep_h1, "h1")
    clean_setups = mark_sweep_relation(clean_setups, sweep_h4, "h4")

    # 3) Грід параметрів
    fib_levels         = [0.382, 0.5, 0.618, 0.705, 0.75, 1.0]
    stop_offsets       = [0.0, 5.0, 10.0]
    rrs                = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]
    h1_modes           = [True, False]
    be_multipliers     = [0, 1, 1.5, 2]
    max_trades_options = [1, 2, 3]
    balance            = 5000

    clean_setups.sort(key=lambda s: pd.to_datetime(s["bos_time"]))

    # 4) Перебір комбінацій
    for h1_filter in h1_modes:
        for be_mult in be_multipliers:
            be_folder  = f"BE_{be_mult}" if be_mult != 0 else "no_BE"
            h1_folder  = "h1_filter" if h1_filter else "no_h1_filter"
            base_dir   = os.path.join(results_dir, be_folder, h1_folder)
            os.makedirs(base_dir, exist_ok=True)

            for fib_level, stop_offset, rr, max_trades in itertools.product(
                fib_levels, stop_offsets, rrs, max_trades_options
            ):
                raw_results = simulate_entry(
                    clean_setups,
                    m5_df,
                    fib_level=fib_level,
                    stop_offset=stop_offset,
                    rr=rr,
                    balance=balance,
                    be_multiplier=be_mult,
                )

                df = pd.DataFrame(raw_results)

                tag   = "H1F" if h1_filter else "NOH1"
                fname = f"{be_mult}R_{tag}_f{fib_level}_s{stop_offset}_r{rr}_n{max_trades}.csv"
                path  = os.path.join(base_dir, fname)

                if df.empty:
                    df.to_csv(path, index=False)
                    print(f"→ Saved 0 trades to {path}")
                    continue

                time_col = next((c for c in ["entry_time", "entry_time_m1", "entry_time_m5"] if c in df.columns), None)
                if time_col is None:
                    print("No entry_time column in results. Columns:", df.columns.tolist())
                    continue

                df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
                try:
                    df["entry_time"] = df[time_col].dt.tz_localize(None)
                except Exception:
                    df["entry_time"] = df[time_col]

                df["entry_date"] = df["entry_time"].dt.date
                df = df.sort_values("entry_time", ascending=False)
                df = df.groupby("entry_date").head(max_trades)

                df.to_csv(path, index=False)
                print(f"→ Saved {len(df)} trades to {path}")
