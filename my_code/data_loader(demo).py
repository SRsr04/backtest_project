import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import logging
import os
from typing import Optional
import sys
import time
from itertools import product

logging.basicConfig(
    filename='trade_search.log',
    filemode='w',                # перезаписувати файл при кожному запуску
    level=logging.INFO,
    format='%(asctime)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def to_naive_kyiv(s: pd.Series, tz_name: str = "Europe/Kyiv"):
    """
    Конвертує числові, рядкові або timezone-aware позначки часу у наївний datetime
    у заданому часовому поясі (типово Europe/Kyiv).
    """
    s = s.copy()

    if pd.api.types.is_integer_dtype(s.dtype) or pd.api.types.is_float_dtype(s.dtype):
        return pd.to_datetime(s, unit='ms', utc=True, errors='coerce').dt.tz_convert(tz_name).dt.tz_localize(None)
    
    if isinstance(s.dtype, pd.DatetimeTZDtype):
        return s.dt.tz_convert(tz_name).dt.tz_localize(None)
    str_s = s.astype('string').str.strip()
    norm = str_s.str.replace(r'[\u200b\ufeff]', '', regex=True) \
             .str.strip() \
             .str.replace('T', ' ', regex=False) \
             .str.replace(r'Z$', '+00:00', regex=True) \
             .str.replace(r'([+-]\d{2})(\d{2})$', r'\1:\2', regex=True)
    tz_mask = norm.str.contains(r'(?:[+-]\d{2}:\d{2}|[+-]\d{4}|Z)$', regex=True, na=False)

    out = pd.Series(pd.NaT, index =s.index, dtype='datetime64[ns]')

    if tz_mask.any():
        parsed_tz = pd.to_datetime(norm[tz_mask], utc=True, errors='coerce') \
                        .dt.tz_convert(tz_name).dt.tz_localize(None)
        out.loc[tz_mask] = parsed_tz


    if (~tz_mask).any():
        parsed_naive = pd.to_datetime(norm[~tz_mask], errors='coerce')
        out.loc[~tz_mask] = parsed_naive

    return out
def load_data():
    """
    Завантажує CSV-файли зі свічками та BOS, уніфікує часові поля до наївного київського часу
    й повертає підготовлені DataFrame-и для подальшої побудови сетапів.
    """
    m5_df  = pd.read_csv('m5_candels.csv')
    m15_df = pd.read_csv('m15_candels.csv')
    m1_df  = pd.read_csv('m1_candels.csv')
    fvg_df = pd.read_csv('fvg_m15.csv')
    m5_bos_df = pd.read_csv('bos_m5.csv')
    h4_bos_df = pd.read_csv('bos_h4.csv')

    for df in (m5_df, m15_df, m1_df):
        if 'timestamp_utc' in df.columns:
            df.drop(columns=['timestamp_utc'], inplace=True)

    for name, bos_df in (("m5_bos_df", m5_bos_df), ("h4_bos_df", h4_bos_df)):
        # fract_time: може містити рядки з +/-offsets — парсимо з utc=True, конвертуємо в Kyiv, робимо naive
        if 'fract_time' in bos_df.columns:
            bos_df['fract_time'] = pd.to_datetime(bos_df['fract_time'], errors='coerce', utc=True) \
                                    .dt.tz_convert('Europe/Kyiv') \
                                    .dt.tz_localize(None)

        # fract_time_kiev: теж парсимо з utc=True на випадок змішаних форматів, конвертуємо, робимо naive
        if 'fract_time_kiev' in bos_df.columns:
            bos_df['fract_time_kiev'] = pd.to_datetime(bos_df['fract_time_kiev'], errors='coerce', utc=True) \
                                            .dt.tz_convert('Europe/Kyiv') \
                                            .dt.tz_localize(None)

        # Уніфікуємо канонічну колонку BOS-часу → 'bos_time' (Kyiv naive)
        if 'bos_time_kiev' in bos_df.columns:
            bos_df['bos_time'] = to_naive_kyiv(bos_df['bos_time_kiev'], tz_name='Europe/Kyiv')
        elif 'bos_time' in bos_df.columns:
            bos_df['bos_time'] = to_naive_kyiv(bos_df['bos_time'], tz_name='Europe/Kyiv')
        # якщо нічого з вищезгаданого немає — залишимо як є; нижче буде загальна обробка/перевірка

        # записати назад
        if name == "m5_bos_df":
            m5_bos_df = bos_df
        else:
            h4_bos_df = bos_df

    for df, cols in [
        (m5_df,  ['datetime']),
        (m15_df, ['datetime']),
        (m1_df,  ['datetime']),
        (fvg_df, ['time']),
        (m5_bos_df, ['bos_time']),
        (h4_bos_df, ['bos_time']),
    ]:
        for c in cols:
            # якщо колонки немає — пропускаємо
            if c not in df.columns:
                continue

            s = df[c]

            # Інша тактика: спочатку намагаємося розпарсити як UTC (щоб покрити випадки з офсетами).
            parsed = pd.to_datetime(s, utc=True, errors="coerce")

            # Якщо майже всі NaT після utc-парсингу — спробуємо без utc (naive місцевий формат)
            if parsed.isna().all():
                parsed = pd.to_datetime(s, errors="coerce")

            # Якщо щось вдалося розпарсити — привести до Kyiv-wall-time (наївний) і записати
            if not parsed.isna().all():
                try:
                    # якщо parsed має tz -> конвертуємо в Kyiv, потім зробимо naive (drop tz)
                    if getattr(parsed.dt, "tz", None) is not None:
                        df[c] = parsed.dt.tz_convert("Europe/Kyiv").dt.tz_localize(None)
                    else:
                        # parsed без tz — вважаємо що це вже локальний Kyiv time
                        df[c] = parsed.dt.tz_localize(None)
                except Exception:
                    # fallback на твою утиліту, якщо щось не так
                    df[c] = to_naive_kyiv(s, tz_name="Europe/Kyiv")
            else:
                # нічого не розпарсилось — покладаємось на to_naive_kyiv (вона робить більш тонкий підхід)
                df[c] = to_naive_kyiv(s, tz_name="Europe/Kyiv")

            # сортування і дедуп
            try:
                df.sort_values(c, inplace=True)
                df.drop_duplicates(subset=[c], keep='last', inplace=True)
                df.reset_index(drop=True, inplace=True)
            except Exception:
                pass

    # приберемо допоміжну колонку, якщо була
    for df in (m5_bos_df, h4_bos_df):
        if 'bos_time_kiev' in df.columns:
            df.drop(columns=['bos_time_kiev'], inplace=True)

    # Індексами робимо наївний datetime (київський wall-time), сортуємо
    m5_df  = m5_df.set_index('datetime').sort_index()
    m1_df  = m1_df.set_index('datetime').sort_index()
    m15_df = m15_df.set_index('datetime').sort_index()

    m5_bos_df = m5_bos_df.sort_values('bos_time')
    h4_bos_df = h4_bos_df.sort_values('bos_time')
    fvg_df = fvg_df.sort_values('time')

    return m5_df, m15_df, fvg_df, m5_bos_df, m1_df, h4_bos_df


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

def find_mitigation_v2(fvg_time, fvg_max, fvg_min, fvg_type, m15_df, *, eps=0.0, m15_index_is_close=True):
    """
    Шукає першу M15-свічку, що повертає ціну в межі FVG, та повертає часові позначки
    її відкриття й закриття разом із ціною закриття.

    Повертає словник з ключами "open", "close", "price" або None, якщо мітигацію не знайдено.
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

def fractals_after_fvg(fvg_df, fractals_df):
    """
    Для кожного FVG знаходить перший коректний F1 та оптимальний F2 з урахуванням появи нових F1.
    Повертає список словників результатів.
    """
    results = []

    # Перетворюємо колонки на NumPy-масиви
    times = fractals_df['time'].to_numpy()
    prices = fractals_df['price'].to_numpy()
    types = fractals_df['type'].to_numpy()
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
        slice_types = types[start:]

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
            tail_types  = types[tail_start:tail_end]

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
                'f1_type': types[f1_idx],
                'f2_time': pd.Timestamp(times[f2_idx]),
                'f2_price': float(prices[f2_idx]),
                'f2_type': types[f2_idx],
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
        f2ts, f1p, f1t = (
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

def simulate_entry(setups, m5_df, m1_df, mode=None, *, fib_level, stop_offset, rr, balance, be_multiplier=0, entry_timeout_candles=None, h4_required=None):
    """
    Логіка:
    - Шукаємо перший M5-бар, що торкає entry у вікні [search_start, search_end].
    - Перед входом перевіряємо "take-before-entry" від limit_placed_time (або search_start) ДОКУПИ з баром входу.
      Якщо тейк торкнуто раніше або в ту ж хвилину (на M1), сетап скасовується.
    - У першій M5 після входу порядок подій вирішує M1 (пріоритет STOP → TAKE при конфліктах у хвилині).
    - BE активується лише на закритті М5 і діє з наступного бару.
    """
    results = []
    EPS = 1e-9

    def finish(direction, setup, outcome_, exit_ts, exit_px, entry_time, entry_price, initial_stop, take_price, be_stop, search_start, limit_ts, f1p, f2p):
        # minutes від search_start до entry_time (усічено)
        minutes = None
        try:
            et = pd.to_datetime(entry_time)
            ss = pd.to_datetime(search_start)
            if et.tzinfo is None and getattr(ss, "tzinfo", None) is not None:
                et = et.tz_localize(ss.tzinfo)
            elif ss.tzinfo is None and getattr(et, "tzinfo", None) is not None:
                ss = ss.tz_localize(et.tzinfo)
            if not pd.isna(et) and not pd.isna(ss):
                minutes = int((et - ss).total_seconds() // 60)
                if minutes < 0:
                    minutes = 0
        except Exception:
            minutes = None

        row = {
            'direction': direction,
            'outcome': outcome_,
            'fvg_time': setup.get('fvg_time'),
            'f1_price': f1p,
            'f2_price': f2p,
            'fib_level': fib_level,
            'limit_placed_time': limit_ts,
            'entry_time_m5': entry_time,
            'search_start_m5': search_start,
            'entry_price': entry_price,
            'stop': initial_stop,
            'take': take_price,
            'bos_time': setup.get('bos_time'),
            'exit_time': exit_ts,
            'exit_price': exit_px,
            'be_triggered': (outcome_ == 'be'),
            'be_price': be_stop,
            'time_to_entry': minutes,
        }
        # стабільні ключі + H4/мітигація (якщо присутні в setup)
        if 'setup_id' in setup:       row['setup_id'] = setup['setup_id']
        if 'allowed_side' in setup:   row['allowed_side'] = setup['allowed_side']

        # --- НОВЕ: усе про мітки мітигації та політику ---
        for k in ("mitigated", "mit_ts", "mit_ts_raw", "mit_ts_open",
                  "mit_ts_close_m15", "mit_bar_open_m15",
                  "mit_cutoff", "mit_policy"):
            if k in setup:
                row[k] = setup[k]

        # (Опційно) кінцева межа вікна пошуку, якою реально користувалися при вході
        if '__entry_window_end__' in setup:
            row['entry_window_end'] = setup['__entry_window_end__']

        results.append(row)

    def _infer_mit_open_if_needed(mit_open, mit_close, *, m15_index_is_close=True):
        if pd.isna(mit_open) and pd.notna(mit_close):
            return (mit_close - pd.Timedelta(minutes=15)) if m15_index_is_close else mit_close.floor("15min")
        # випадок коли хтось поклав CLOSE у поле cutoff/open — виправляємо
        if pd.notna(mit_open) and pd.notna(mit_close) and mit_open == mit_close and m15_index_is_close:
            return mit_close - pd.Timedelta(minutes=15)
        return mit_open

    def take_hit_before_entry(pre_start, entry_time, entry_price, take_price, direction, m1_df, m5_df, EPS):
        """
        True → тейк був до/у ту ж хвилину, що і перше торкання entry → сетап скасовуємо.
        """
        # 1) Спроба визначити хвилину входу на M1 (більш точний порядок)
        if m1_df is not None and not m1_df.empty:
            m1_full = m1_df.loc[pre_start: entry_time]  # включно з хвилиною входу
            if not m1_full.empty:
                if direction == 'long':
                    touched_entry = (m1_full['low'] <= entry_price + EPS)
                    if touched_entry.any():
                        entry_min_ts = m1_full.index[int(np.argmax(touched_entry.values))]
                        # тейк доторкнуто до або в ту ж хвилину → скасувати
                        if (m1_full.loc[:entry_min_ts, 'high'] >= take_price - EPS).any():
                            return True
                else:  # short
                    touched_entry = (m1_full['high'] >= entry_price - EPS)
                    if touched_entry.any():
                        entry_min_ts = m1_full.index[int(np.argmax(touched_entry.values))]
                        if (m1_full.loc[:entry_min_ts, 'low'] <= take_price + EPS).any():
                            return True
        # 2) Консервативний фолбек на M5: включаємо бар входу
        m5_pre = m5_df.loc[pre_start: entry_time]
        if m5_pre.empty:
            return False
        if direction == 'long':
            return (m5_pre['high'] >= take_price - EPS).any()
        else:
            return (m5_pre['low']  <= take_price + EPS).any()

    for setup in setups:
        # межі пошуку
        bos_time = pd.to_datetime(setup.get('bos_time'))
        # search_start/search_end пришли з підготовчого етапу; якщо ні — фолбек на bos_time+5m / +8h
        search_start = pd.to_datetime(setup.get('search_start') or (bos_time + pd.Timedelta(minutes=5)))
        search_end   = pd.to_datetime(setup.get('search_end')   or (search_start + pd.Timedelta(hours=8)))
        if pd.isna(search_start) or pd.isna(search_end) or (search_start >= search_end):
            continue

        # сторони: поважаємо allowed_side, якщо задано
        allowed_side = setup.get('allowed_side', None)

        # ціни
        f1p, f2p = float(setup['f1_price']), float(setup['f2_price'])
        if setup['f1_type'] == 'high':
            direction = 'long'
            entry_price   = f2p + (f1p - f2p) * fib_level
            initial_stop  = f2p - stop_offset
        else:
            direction = 'short'
            entry_price   = f2p - (f2p - f1p) * fib_level
            initial_stop  = f2p + stop_offset

        if allowed_side and ((allowed_side == 'long' and direction != 'long') or (allowed_side == 'short' and direction != 'short')):
            continue

        distance = abs(entry_price - initial_stop)
        if distance <= EPS:
            continue

        take_price = entry_price + distance * rr if direction == 'long' else entry_price - distance * rr

        # ризик/комісія → BE-стоп
        risk_usd = balance * 0.01
        position_size = risk_usd / distance if distance > 0 else np.nan
        commission = (position_size * entry_price * 0.000325) if position_size and not np.isnan(position_size) else 0.0
        delta_p = commission / position_size if position_size and position_size != 0 else 0.0

        be_stop = None
        be_activation = None
        if be_multiplier > 0:
            be_activation = entry_price + distance * be_multiplier if direction == 'long' else entry_price - distance * be_multiplier
            be_stop = entry_price + delta_p if direction == 'long' else entry_price - delta_p

        mitigated = bool(setup.get("mitigated", False))
        mit_open  = pd.to_datetime(
            setup.get("mit_bar_open_m15") or setup.get("mit_ts_open") or setup.get("mit_cutoff") or setup.get("cutoff"),
            errors="coerce"
        )
        mit_close = pd.to_datetime(
            setup.get("mit_ts_close_m15") or setup.get("mit_ts_raw"),
            errors="coerce"
        )
        mit_policy = (setup.get("mit_policy") or "cutoff").lower()

        # інфер OPEN із CLOSE, якщо треба (сумісність зі старими/кривими сетапами)
        mit_open = _infer_mit_open_if_needed(mit_open, mit_close, m15_index_is_close=True)
        try:
            if pd.notna(mit_open):  mit_open  = mit_open.tz_localize(None)
            if pd.notna(mit_close): mit_close = mit_close.tz_localize(None)
        except: pass

        # --- формуємо вікно пошуку ---
        if mitigated and mit_policy in ("cutoff","open") and pd.notna(mit_open):
            hard_end = min(search_end, mit_open)
        else:
            hard_end = search_end

        window = m5_df[(m5_df.index >= search_start) & (m5_df.index < hard_end)]
        if window.empty:
            continue

        if entry_timeout_candles is not None:
            window = window.iloc[: entry_timeout_candles + 1]
            if window.empty:
                continue

        # --- шукаємо entry_time ---
        is_long = (direction == 'long')
        arr_low  = window['low' ].to_numpy()
        arr_high = window['high'].to_numpy()
        mask = (arr_low <= entry_price + EPS) if is_long else (arr_high >= entry_price - EPS)
        if not mask.any():
            continue

        idx_entry  = int(np.argmax(mask))
        entry_time = pd.Timestamp(window.index.to_numpy()[idx_entry])

        if mitigated and pd.notna(mit_open):
            # переконаємось, що mit_open без тайзони
            try:
                mit_open_n = pd.to_datetime(mit_open).tz_localize(None)
            except Exception:
                mit_open_n = pd.to_datetime(mit_open, errors="coerce")
            # перевіряємо чи була хоча б якась торкання entry ДО мит-бару (не включає мит-бар)
            pre_mit_start = search_start
            pre_mit_end = mit_open_n - pd.Timedelta(minutes=0)  # exclude mit_open itself

            touched_before_mit = False

            # точніша перевірка на M1, якщо є
            if (m1_df is not None) and (not m1_df.empty):
                try:
                    m1_slice_pre_mit = m1_df.loc[pre_mit_start: pre_mit_end]
                    if not m1_slice_pre_mit.empty:
                        if direction == 'long':
                            touched_before_mit = (m1_slice_pre_mit['low'] <= entry_price + EPS).any()
                        else:
                            touched_before_mit = (m1_slice_pre_mit['high'] >= entry_price - EPS).any()
                except Exception:
                    # у разі проблем зі зрізом — ігноруємо і продовжимо з M5
                    touched_before_mit = False

            # fallback на M5, якщо M1 не показав торкання
            if (not touched_before_mit) and (m5_df is not None) and (not m5_df.empty):
                try:
                    m5_slice_pre_mit = m5_df.loc[pre_mit_start: pre_mit_end]
                    if not m5_slice_pre_mit.empty:
                        if direction == 'long':
                            touched_before_mit = (m5_slice_pre_mit['low'] <= entry_price + EPS).any()
                        else:
                            touched_before_mit = (m5_slice_pre_mit['high'] >= entry_price - EPS).any()
                except Exception:
                    touched_before_mit = False

            if not touched_before_mit:
                print("MIT-KILL: mitigated but entry NOT touched before mitigation -> skip", setup.get("setup_id"), "| mit_open", mit_open_n, "| entry_price", entry_price)
                continue

        # законний assert
        assert search_start <= entry_time < hard_end

        # # --- страхувальна заборона входу після мітигації ---
        # print("MIT-DEBUG",
        #     "id", setup.get("setup_id"),
        #     "| s_start", search_start,
        #     "| s_end", search_end,
        #     "| mit_open", mit_open,
        #     "| policy", mit_policy)

        # # після обчислення entry_time:
        # if mitigated:
        #     if mit_policy in ("cutoff","open") and pd.notna(mit_open) and entry_time >= mit_open:
        #         print("MIT-KILL cutoff", setup.get("setup_id"),
        #             "entry", entry_time, ">= open", mit_open)
        #         continue
        #     if mit_policy in ("strict","close") and pd.notna(mit_close) and entry_time >= mit_close:
        #         print("MIT-KILL strict", setup.get("setup_id"),
        #             "entry", entry_time, ">= close", mit_close)
        #         continue

        # --- take-before-entry ---
        limit_ts  = pd.to_datetime(setup.get('limit_placed_time'), errors='coerce')
        limit_ts  = limit_ts if pd.notna(limit_ts) else search_start
        pre_start = min(max(search_start, limit_ts), entry_time)

        if take_hit_before_entry(pre_start, entry_time, entry_price, take_price, direction, m1_df, m5_df, EPS):
            continue

        # перша M5 після входу — розв'язуємо на M1
        # (вікно рівно 5 хвилин, включно з хвилиною entry_time)
        m1_slice = None
        resolved_on_m1 = False
        if (m1_df is not None) and (not m1_df.empty):
            # узгоджуємо tz, щоб зрізи не ламались
            et = pd.to_datetime(entry_time, errors="coerce")
            if pd.notna(et):
                try:
                    et = et.tz_localize(None)
                except Exception:
                    pass

                m1_slice = m1_df.loc[et : et + pd.Timedelta(minutes=4)]
                if not m1_slice.empty:
                    for ts, row1 in m1_slice.sort_index().iterrows():
                        if direction == "long":
                            # STOP → TAKE
                            if row1["low"]  <= initial_stop + EPS:
                                finish(direction, setup, 'stop', ts, initial_stop, entry_time, entry_price,
                                    initial_stop, take_price, be_stop, search_start, limit_ts, f1p, f2p)
                                resolved_on_m1 = True
                                break
                            if row1["high"] >= take_price  - EPS:
                                finish(direction, setup, 'take', ts, take_price, entry_time, entry_price,
                                    initial_stop, take_price, be_stop, search_start, limit_ts, f1p, f2p)
                                resolved_on_m1 = True
                                break
                        else:
                            # short: STOP → TAKE
                            if row1["high"] >= initial_stop - EPS:
                                finish(direction, setup, 'stop', ts, initial_stop, entry_time, entry_price,
                                    initial_stop, take_price, be_stop, search_start, limit_ts, f1p, f2p)
                                resolved_on_m1 = True
                                break
                            if row1["low"]  <= take_price  + EPS:
                                finish(direction, setup, 'take', ts, take_price, entry_time, entry_price,
                                    initial_stop, take_price, be_stop, search_start, limit_ts, f1p, f2p)
                                resolved_on_m1 = True
                                break

        if resolved_on_m1:
            # уже зафіксували результат на М1 у першій M5 → далі не перевіряємо M5-консолідацію
            continue

        # якщо M1 не визначив вихід у першій M5 — перевіряємо саму M5-консолідовано
        post = m5_df.loc[entry_time:m5_df.index.max()]
        if post.empty:
            continue
        entry_row = post.iloc[0]
        is_long = (direction == 'long')

        # Конфлікт у першій M5: STOP має пріоритет
        hit_take_0 = (entry_row['high'] >= take_price - EPS) if is_long else (entry_row['low']  <= take_price + EPS)
        hit_stop_0 = (entry_row['low']  <= initial_stop + EPS) if is_long else (entry_row['high'] >= initial_stop - EPS)
        if hit_stop_0:
            finish(direction, setup, 'stop', entry_time, initial_stop, entry_time, entry_price,
                   initial_stop, take_price, be_stop, search_start, limit_ts, f1p, f2p)
            continue
        if hit_take_0:
            finish(direction, setup, 'take', entry_time, take_price, entry_time, entry_price,
                   initial_stop, take_price, be_stop, search_start, limit_ts, f1p, f2p)
            continue

        # BE активується ТІЛЬКИ якщо бар закрився за рівнем be_activation; діє з наступного бару
        be_active_from_next_bar = False
        if be_activation is not None:
            be_hit_close = (entry_row['close'] >= be_activation - EPS) if is_long else (entry_row['close'] <= be_activation + EPS)
            if be_hit_close:
                be_active_from_next_bar = True

        # подальші бари
        lows_p, highs_p, closes_p, times_p = (
            post['low'].to_numpy(), post['high'].to_numpy(),
            post['close'].to_numpy(), post.index.to_numpy()
        )
        if len(times_p) <= 1:
            continue

        for i in range(1, len(times_p)):
            current_stop = be_stop if be_active_from_next_bar and (be_stop is not None) else initial_stop
            low_i, high_i, close_i, t_i = lows_p[i], highs_p[i], closes_p[i], times_p[i]

            hit_stop_i = (low_i  <= current_stop + EPS) if is_long else (high_i >= current_stop - EPS)
            hit_take_i = (high_i >= take_price  - EPS) if is_long else (low_i  <= take_price + EPS)

            if hit_stop_i:
                finish(direction, setup, 'be' if (be_active_from_next_bar and be_stop is not None and current_stop == be_stop) else 'stop',
                       pd.Timestamp(t_i), current_stop, entry_time, entry_price,
                       initial_stop, take_price, be_stop, search_start, limit_ts, f1p, f2p)
                break
            if hit_take_i:
                finish(direction, setup, 'take', pd.Timestamp(t_i), take_price, entry_time, entry_price,
                       initial_stop, take_price, be_stop, search_start, limit_ts, f1p, f2p)
                break

            # BE може активуватись на ЗАКРИТТІ цього бару і почати діяти з НАСТУПНОГО
            if (not be_active_from_next_bar) and (be_activation is not None):
                be_hit_close = (close_i >= be_activation - EPS) if is_long else (close_i <= be_activation + EPS)
                if be_hit_close:
                    be_active_from_next_bar = True

    return results  

def blank_h4_columns(df: pd.DataFrame, keep_k: int, max_keep: int):
    for j in range(keep_k+1, max_keep+1):
        for suffix in ("bos_time","confirm_time","dir","level","close","fract_time"):
            col = f"h4_{j}_{suffix}"
            if col in df.columns:
                if suffix in ("bos_time","confirm_time","fract_time"):
                    df[col] = pd.NaT
                elif suffix in ("level","close"):
                    df[col] = pd.NA
                else:
                    df[col] = None
    if keep_k == 0:
        for col in ("h4_bos_time","h4_confirm_time","h4_bos_dir","h4_level","h4_close","h4_fract_time"):
            if col in df.columns:
                if col.endswith("_time"):
                    df[col] = pd.NaT
                elif col in ("h4_level","h4_close"):
                    df[col] = pd.NA
                else:
                    df[col] = None

if __name__ == "__main__":
    results_dir = "backtest_results"
    os.makedirs(results_dir, exist_ok=True)

    # 1) Дані + свіпи
    m5_df, m15_df, fvg_df, m5_bos_df, m1_df, h4_bos_df = load_data()

    # Обрізаємо всі таймфрейми під M5-інтервал
    m5_min, m5_max = m5_df.index.min(), m5_df.index.max()
    start, end = m5_min, m5_max
    m5_df  = m5_df.loc[start:end]
    m15_df = m15_df.loc[start:end]
    m1_df  = m1_df.loc[start:end]
    M15_INDEX_IS_CLOSE = True

    if m1_df.empty:
        print("WARN: M1 empty → fallback-only M5")

    # --- ПАРАМЕТРИ ГРІДА (винесено наперед, бо MAX_H4_KEEP залежить) ---
    fib_levels           = [0.382, 0.5, 0.618, 0.705, 0.75, 1.0]
    stop_offsets         = [0.0, 5.0, 10.0]
    rrs                  = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]
    be_multipliers       = [0, 1, 1.5, 2]
    max_trades_options   = [1, 2, 3]
    h4_required_options  = [0, 1, 2, 3]

    # Скільки останніх H4-BOS лишати в колонках (напр., 3)
    MAX_H4_KEEP = max(h4_required_options)

    # --- Нормалізація напрямку в H4 BOS ---
    DIR_COL = 'direction' if 'direction' in h4_bos_df.columns else ('type' if 'type' in h4_bos_df.columns else None)
    if DIR_COL is None:
        raise ValueError("bos_h4.csv: немає колонки 'direction' або 'type'")

    def _norm_dir(x: str):
        x = str(x).strip().lower()
        if x in ('bull', 'bullish', 'up', 'long', 'buy'):
            return 'bullish'
        if x in ('bear', 'bearish', 'down', 'short', 'sell'):
            return 'bearish'
        return None

    h4_bos_df[DIR_COL] = h4_bos_df[DIR_COL].apply(_norm_dir)

    # Якщо в CSV 'bos_time' — це open бара, а треба підтвердження після закриття,
    # додаємо зсув у 4 години (для H4). Якщо в твоєму CSV вже стоїть саме close —
    # постав H4_BAR_HOURS = 0.
    H4_BAR_HOURS = 4
    h4_bos_df['confirm_time'] = pd.to_datetime(h4_bos_df['bos_time']) + pd.Timedelta(hours=H4_BAR_HOURS)
    h4_bos_df = h4_bos_df.sort_values('confirm_time').reset_index(drop=True)

    # Для швидкого бісекту (перевірка відсортованості)
    _h4_confirm_np = h4_bos_df['confirm_time'].values.astype('datetime64[ns]')
    assert (np.diff(_h4_confirm_np) >= np.timedelta64(0, 'ns')).all()

    # 2) Формуємо сетапи
    m5_fractals = find_fractals(m5_df)
    fvg_res     = fractals_after_fvg(fvg_df, m5_fractals)
    raw_setups  = fvg_res.to_dict(orient="records")

    # Повертаємо оригінальні межі FVG, якщо є
    fvg_map = {}
    if 'fvg_df' in globals():
        time_col = 'time' if 'time' in fvg_df.columns else ('fvg_time' if 'fvg_time' in fvg_df.columns else None)
        if time_col is not None:
            for _, r in fvg_df.iterrows():
                try:
                    k = pd.to_datetime(r[time_col]).floor('min')
                    fvg_map[k] = {'orig_fvg_min': float(r['min']), 'orig_fvg_max': float(r['max'])}
                except Exception:
                    continue

    for s in raw_setups:
        try:
            k = pd.to_datetime(s.get('fvg_time')).floor('min')
            if k in fvg_map:
                s['orig_fvg_min'] = fvg_map[k]['orig_fvg_min']
                s['orig_fvg_max'] = fvg_map[k]['orig_fvg_max']
            else:
                s['orig_fvg_min'] = s.get('fvg_min', None)
                s['orig_fvg_max'] = s.get('fvg_max', None)
        except Exception:
            s['orig_fvg_min'] = s.get('fvg_min', None)
            s['orig_fvg_max'] = s.get('fvg_max', None)

    setups = add_bos_after_f2(raw_setups, m5_bos_df)

    # Вікно покриття M5
    setups = [
        s for s in setups
        if (m5_min - pd.Timedelta(minutes=5)) <= pd.to_datetime(s["bos_time"]) <= (m5_max - pd.Timedelta(minutes=5))
    ]

    # 2.1) Фільтр мітигацій ДО BOS + дедлайн пошуку BOS
    H_MAX = 8  # годин пошуку після gate
    clean_setups = []

    def _norm_fvg(t):
        if t is None:
            return None
        t = str(t).strip().lower()
        return 'bullish' if t.startswith('bull') else ('bearish' if t.startswith('bear') else None)

    for s in setups:
        # Якщо немає збереженої мітигації — розрахувати
        if not s.get("mitigated", False) or pd.isna(s.get("mit_ts_open")):
            try:
                fvg_time = pd.to_datetime(s.get("fvg_time"))
                fmin     = float(s.get("fvg_min", s.get("f1_price")))
                fmax     = float(s.get("fvg_max", s.get("f2_price")))
                z_low, z_high = sorted([fmin, fmax])
                ftype    = s.get("fvg_type")
                mit2     = find_mitigation_v2(fvg_time, z_high, z_low, ftype, m15_df, eps=0.0, m15_index_is_close=True)
                if mit2:
                    s["mitigated"]          = True
                    s["mit_ts_open"]        = mit2["open"]
                    s["mit_bar_open_m15"]   = mit2["open"]
                    s["mit_ts_close_m15"]   = mit2["close"]
                    s["mit_ts_raw"]         = mit2["close"]
                    s["mit_policy"]         = s.get("mit_policy", "cutoff")
                    s["mit_cutoff"]         = mit2["open"]
                    s["cutoff"]             = mit2["open"]
            except Exception:
                pass

        # Базові поля
        fvg_time    = pd.to_datetime(s["fvg_time"])
        m5_bos_time = pd.to_datetime(s["bos_time"])
        s["m5_bos_time"] = m5_bos_time

        # --- (A) Вибір останніх K H4-BOS до FVG за confirm_time ---
        mask_h4 = h4_bos_df["confirm_time"] <= fvg_time
        if not mask_h4.any():
            # сетап потребує H4-контекст, інакше пропускаємо
            continue

        h4_slice = h4_bos_df.loc[mask_h4].sort_values("confirm_time")
        h4_tail  = h4_slice.tail(MAX_H4_KEEP).reset_index(drop=True)

        # Останній (найсвіжіший) BOS
        row_last   = h4_tail.iloc[-1]
        h4_open    = row_last["bos_time"]
        h4_confirm = row_last["confirm_time"]
        h4_dir     = row_last[DIR_COL]  # 'bullish'/'bearish'

        # FVG має бути після confirm останнього BOS
        if fvg_time < h4_confirm:
            continue

        # Напрям валідності беремо з останнього BOS
        s["fvg_type"] = _norm_fvg(s.get("fvg_type"))
        desired_fvg   = 'bullish' if h4_dir == 'bullish' else 'bearish'
        if s["fvg_type"] != desired_fvg:
            continue

        # --- (B) Обмеження вікна пошуку (gate + мітигація) ---
        gate         = max(h4_confirm, fvg_time, m5_bos_time)
        search_start = gate.floor("5min") + pd.Timedelta(minutes=5)

        z_low  = float(s.get('orig_fvg_min', s.get('fvg_min')))
        z_high = float(s.get('orig_fvg_max', s.get('fvg_max')))
        z_low, z_high = sorted([z_low, z_high])

        mit = find_mitigation_v2(fvg_time, z_high, z_low, s["fvg_type"], m15_df, eps=0.0, m15_index_is_close=M15_INDEX_IS_CLOSE)
        if mit is None:
            search_end       = search_start + pd.Timedelta(hours=H_MAX)
            s["mitigated"]   = False
            s["mit_ts_raw"]  = None
            s["mit_ts_open"] = None
            s["mit_cutoff"]  = None
        else:
            mit_open  = mit["open"]
            mit_close = mit["close"]
            # Мітигація до BOS → невалідно
            if mit_open <= m5_bos_time:
                s["valid"] = False
                continue
            search_end = mit_open
            s["mitigated"]          = True
            s["mit_ts_raw"]         = mit_close
            s["mit_ts_open"]        = mit_open
            s["mit_bar_open_m15"]   = mit_open
            s["mit_ts_close_m15"]   = mit_close
            s["mit_policy"]         = "cutoff"
            s["mit_cutoff"]         = mit_open
            s["cutoff"]             = mit_open

        # Фінальні бар’єри
        search_end = min(search_end, m5_max)
        if search_end <= search_start:
            continue
        if not (m5_bos_time < search_start):
            continue

        # Діагностика BOS після FVG
        delta = m5_bos_time - fvg_time
        secs  = int(delta.total_seconds()); mins = secs // 60
        s["bos_after_fvg_sec"] = secs
        s["bos_after_fvg_min"] = mins
        s["bos_after_fvg_str"] = f"{'-' if secs < 0 else ''}{abs(mins)}m {abs(secs) % 60}s"

        # --- (C) Ланцюжок підрядних BOS того ж напрямку (до FVG) ---
        rev = h4_slice.iloc[::-1]  # від найсвіжішого назад
        first_dir = rev.iloc[0][DIR_COL]
        cnt = 0
        for _, r in rev.iterrows():
            if r[DIR_COL] == first_dir:
                cnt += 1
            else:
                break
        s["h4_chain_count"]       = int(cnt)
        s["h4_chain_dir"]         = first_dir
        s["h4_chain_matches_fvg"] = (first_dir == desired_fvg)

        # --- (D) Розкладаємо останні K BOS у s: h4_1_*, h4_2_*, ... ---
        h4_rev = h4_tail.iloc[::-1].reset_index(drop=True)  # 0 = найсвіжіший
        for j in range(1, MAX_H4_KEEP+1):
            if j-1 < len(h4_rev):
                rj = h4_rev.iloc[j-1]
                s[f"h4_{j}_bos_time"]     = pd.to_datetime(rj.get("bos_time"), errors="coerce")
                s[f"h4_{j}_confirm_time"] = pd.to_datetime(rj.get("confirm_time"), errors="coerce")
                s[f"h4_{j}_dir"]          = rj.get(DIR_COL)
                s[f"h4_{j}_level"]        = rj.get("level", rj.get("h4_level", None))
                s[f"h4_{j}_close"]        = rj.get("close", rj.get("h4_close", None))
                raw_fract                 = rj.get("fract_time_kiev", rj.get("fract_time", None))
                s[f"h4_{j}_fract_time"]   = pd.to_datetime(raw_fract, errors="coerce")
            else:
                s[f"h4_{j}_bos_time"]     = None
                s[f"h4_{j}_confirm_time"] = None
                s[f"h4_{j}_dir"]          = None
                s[f"h4_{j}_level"]        = None
                s[f"h4_{j}_close"]        = None
                s[f"h4_{j}_fract_time"]   = None

        # (E) «Соло»-метадані (з останнього BOS) для сумісності/фільтрів
        s["h4_bos_time"]     = h4_open
        s["h4_confirm_time"] = h4_confirm
        s["h4_bos_dir"]      = h4_dir
        s["allowed_side"]    = 'long' if h4_dir == 'bullish' else 'short'

        def _f(x):
            try:
                return float(x)
            except:
                return None
        s["h4_level"] = _f(row_last.get("level", row_last.get("h4_level", None)))
        s["h4_close"] = _f(row_last.get("close", row_last.get("h4_close", None)))

        raw_fract = row_last.get("fract_time_kiev", row_last.get("fract_time", None))
        s["h4_fract_time"]     = pd.to_datetime(raw_fract, errors='coerce')
        s["h4_fract_time_str"] = s["h4_fract_time"].strftime("%Y-%m-%d %H:%M:%S") if not pd.isna(s["h4_fract_time"]) else None

        s["search_start"] = search_start
        s["search_end"]   = search_end

        # Стабільний ключ для джойну з результатами
        s["setup_id"] = f"{int(pd.Timestamp(fvg_time).value)}-{int(pd.Timestamp(m5_bos_time).value)}"

        assert h4_confirm <= fvg_time <= m5_bos_time
        assert search_start < search_end

        s["valid"] = True
        clean_setups.append(s)

    if not clean_setups:
        print("No setups in M5 coverage window (after H4 confirmation)")
        sys.exit(0)

    balance = 5000

    # Сортуємо для детермінованості
    clean_setups.sort(key=lambda s: pd.to_datetime(s.get("h4_bos_time") or s.get("fvg_time")))

    # Кеш фільтрів за h4_required
    filtered_cache = {}
    for h in h4_required_options:
        filtered_cache[h] = [
            s for s in clean_setups
            if s.get("valid", True)
            and s.get("h4_chain_matches_fvg", False)
            and int(s.get("h4_chain_count", 0)) >= h
        ]

    # Плоский список комбінацій
    param_grid = list(product(
        fib_levels,
        stop_offsets,
        rrs,
        be_multipliers,
        max_trades_options,
        h4_required_options
    ))
    total = len(param_grid)
    master_csv = os.path.join(results_dir, "all_results3.csv")
    start_time = time.time()

    # Пропуск уже зроблених комбо
    done_ids = set()
    if os.path.exists(master_csv):
        try:
            hdr = pd.read_csv(master_csv, nrows=0)
            if '_combo_id' in getattr(hdr, 'columns', []):
                done_ids = set(pd.read_csv(master_csv, usecols=['_combo_id'])['_combo_id'].astype(str).unique())
        except Exception:
            done_ids = set()

    buffered_results = []

    for idx, (fib_level, stop_offset, rr, be_mult, max_trades, h4_required) in enumerate(param_grid, start=1):
        print(f"[{idx}/{total}] fib={fib_level} stop={stop_offset} rr={rr} be={be_mult} maxtr={max_trades} h4_req={h4_required}")

        filtered_setups = filtered_cache.get(h4_required, [])
        if not filtered_setups:
            continue

        combo_id = f"fib{fib_level}_stop{stop_offset}_rr{rr}_be{be_mult}_maxtr{max_trades}_h4{h4_required}"
        if combo_id in done_ids:
            print("  -> SKIP (already done)")
            continue

        # Симуляція для цієї комбінації
        raw_results = simulate_entry(
            filtered_setups,
            m5_df,
            m1_df,
            fib_level=fib_level,
            stop_offset=stop_offset,
            rr=rr,
            balance=balance,
            be_multiplier=be_mult,
        )

        df_run = pd.DataFrame(raw_results or [])
        if df_run.empty:
            continue

        # --- Збирання метаданих H4 (включно з h4_j_*) для мерджу за setup_id ---
        setup_map_id = {}
        for s in filtered_setups:
            if "setup_id" not in s:
                continue
            d = {
                "h4_bos_time":     s.get("h4_bos_time"),
                "h4_confirm_time": s.get("h4_confirm_time"),
                "h4_bos_dir":      s.get("h4_bos_dir"),
                "h4_level":        s.get("h4_level"),
                "h4_close":        s.get("h4_close"),
                "h4_fract_time":   s.get("h4_fract_time"),
            }
            for j in range(1, MAX_H4_KEEP+1):
                d[f"h4_{j}_bos_time"]     = s.get(f"h4_{j}_bos_time")
                d[f"h4_{j}_confirm_time"] = s.get(f"h4_{j}_confirm_time")
                d[f"h4_{j}_dir"]          = s.get(f"h4_{j}_dir")
                d[f"h4_{j}_level"]        = s.get(f"h4_{j}_level")
                d[f"h4_{j}_close"]        = s.get(f"h4_{j}_close")
                d[f"h4_{j}_fract_time"]   = s.get(f"h4_{j}_fract_time")
            setup_map_id[s["setup_id"]] = d

        # Fallback мапа по fvg_time (на випадок відсутності setup_id)
        setup_map_fvg = {}
        for s in filtered_setups:
            try:
                k = pd.to_datetime(s["fvg_time"], errors="coerce").floor("min")
                if pd.notna(k):
                    d = {
                        "h4_bos_time":     s.get("h4_bos_time"),
                        "h4_confirm_time": s.get("h4_confirm_time"),
                        "h4_bos_dir":      s.get("h4_bos_dir"),
                        "h4_level":        s.get("h4_level"),
                        "h4_close":        s.get("h4_close"),
                        "h4_fract_time":   s.get("h4_fract_time"),
                    }
                    for j in range(1, MAX_H4_KEEP+1):
                        d[f"h4_{j}_bos_time"]     = s.get(f"h4_{j}_bos_time")
                        d[f"h4_{j}_confirm_time"] = s.get(f"h4_{j}_confirm_time")
                        d[f"h4_{j}_dir"]          = s.get(f"h4_{j}_dir")
                        d[f"h4_{j}_level"]        = s.get(f"h4_{j}_level")
                        d[f"h4_{j}_close"]        = s.get(f"h4_{j}_close")
                        d[f"h4_{j}_fract_time"]   = s.get(f"h4_{j}_fract_time")
                    setup_map_fvg[k] = d
            except Exception:
                pass

        mapped = False
        if "setup_id" in df_run.columns and setup_map_id:
            df_run["h4_bos_time"]     = df_run["setup_id"].map(lambda k: setup_map_id.get(k, {}).get("h4_bos_time"))
            df_run["h4_confirm_time"] = df_run["setup_id"].map(lambda k: setup_map_id.get(k, {}).get("h4_confirm_time"))
            df_run["h4_bos_dir"]      = df_run["setup_id"].map(lambda k: setup_map_id.get(k, {}).get("h4_bos_dir"))
            df_run["h4_level"]        = df_run["setup_id"].map(lambda k: setup_map_id.get(k, {}).get("h4_level"))
            df_run["h4_close"]        = df_run["setup_id"].map(lambda k: setup_map_id.get(k, {}).get("h4_close"))
            df_run["h4_fract_time"]   = df_run["setup_id"].map(lambda k: setup_map_id.get(k, {}).get("h4_fract_time"))
            for j in range(1, MAX_H4_KEEP+1):
                df_run[f"h4_{j}_bos_time"]     = df_run["setup_id"].map(lambda k, j=j: setup_map_id.get(k, {}).get(f"h4_{j}_bos_time"))
                df_run[f"h4_{j}_confirm_time"] = df_run["setup_id"].map(lambda k, j=j: setup_map_id.get(k, {}).get(f"h4_{j}_confirm_time"))
                df_run[f"h4_{j}_dir"]          = df_run["setup_id"].map(lambda k, j=j: setup_map_id.get(k, {}).get(f"h4_{j}_dir"))
                df_run[f"h4_{j}_level"]        = df_run["setup_id"].map(lambda k, j=j: setup_map_id.get(k, {}).get(f"h4_{j}_level"))
                df_run[f"h4_{j}_close"]        = df_run["setup_id"].map(lambda k, j=j: setup_map_id.get(k, {}).get(f"h4_{j}_close"))
                df_run[f"h4_{j}_fract_time"]   = df_run["setup_id"].map(lambda k, j=j: setup_map_id.get(k, {}).get(f"h4_{j}_fract_time"))
            mapped = True

        if (not mapped) and ("fvg_time" in df_run.columns) and setup_map_fvg:
            df_run["fvg_time"]    = pd.to_datetime(df_run["fvg_time"], errors="coerce")
            df_run["__fvg_key__"] = df_run["fvg_time"].dt.floor("min")
            meta_df = (
                pd.DataFrame.from_dict(setup_map_fvg, orient="index")
                .reset_index().rename(columns={"index": "__fvg_key__"})
            )
            df_run = df_run.merge(meta_df, on="__fvg_key__", how="left")
            df_run.drop(columns="__fvg_key__", inplace=True, errors="ignore")

        # Типи для часу/чисел
        time_cols = ["h4_bos_time", "h4_confirm_time", "h4_fract_time"] + \
                    [f"h4_{j}_bos_time" for j in range(1, MAX_H4_KEEP+1)] + \
                    [f"h4_{j}_confirm_time" for j in range(1, MAX_H4_KEEP+1)] + \
                    [f"h4_{j}_fract_time" for j in range(1, MAX_H4_KEEP+1)]
        for c in time_cols:
            if c in df_run.columns:
                df_run[c] = pd.to_datetime(df_run[c], errors="coerce")

        num_cols = ["h4_level", "h4_close"] + \
                   [f"h4_{j}_level" for j in range(1, MAX_H4_KEEP+1)] + \
                   [f"h4_{j}_close" for j in range(1, MAX_H4_KEEP+1)]
        for c in num_cols:
            if c in df_run.columns:
                df_run[c] = pd.to_numeric(df_run[c], errors="coerce")

        # Додаємо мета-параметри гріда
        meta = {
            "fib_level": fib_level,
            "stop_offset": stop_offset,
            "rr": rr,
            "be_multiplier": be_mult,
            "h4_required": h4_required,
            "max_trades": max_trades,
        }
        for k, v in meta.items():
            df_run[k] = v

        # Нормалізуємо час входу
        time_col = next((c for c in ["entry_time", "entry_time_m1", "entry_time_m5"] if c in df_run.columns), None)
        if time_col is not None:
            df_run[time_col] = pd.to_datetime(df_run[time_col], errors="coerce")
            try:
                df_run["entry_time"] = df_run[time_col].dt.tz_localize(None)
            except Exception:
                df_run["entry_time"] = df_run[time_col]
        
        blank_h4_columns(df_run, keep_k=int(h4_required), max_keep=MAX_H4_KEEP)

        df_run = df_run.sort_values("entry_time", ascending=False, ignore_index=True)

        # Маркуємо комбінацію і час
        df_run['_run_ts']   = pd.Timestamp.utcnow().isoformat()
        df_run['_combo_id'] = combo_id

        buffered_results.append(df_run)
        print(f"  -> buffered {len(df_run)} rows")

    # --- Запис у майстер-файл разом, одним махом ---
    master_csv = os.path.join(results_dir, "sol_all_results.csv")
    if buffered_results:
        new_all = pd.concat(buffered_results, ignore_index=True, sort=False)

        if os.path.exists(master_csv):
            try:
                existing = pd.read_csv(master_csv)
            except Exception:
                existing = pd.DataFrame()
        else:
            existing = pd.DataFrame()

        # Забираємо старі рядки з тими ж _combo_id (якщо є)
        if not existing.empty and '_combo_id' in existing.columns:
            new_ids = set(new_all['_combo_id'].astype(str).unique())
            existing = existing[~existing['_combo_id'].astype(str).isin(new_ids)]

        combined = pd.concat([existing, new_all], ignore_index=True, sort=False)

        # Atomic write
        tmp_csv = master_csv + ".tmp"
        combined.to_csv(tmp_csv, index=False)
        os.replace(tmp_csv, master_csv)

        abs_master = os.path.abspath(master_csv)
        print(f"Wrote {len(new_all)} new rows to {abs_master}; total {len(combined)}")

        # Діагностика розміру
        try:
            sz_bytes = os.path.getsize(abs_master)
            print(f"Master CSV size: {sz_bytes/1024/1024:.2f} MB; rows={len(combined):,}; cols={combined.shape[1]}")
        except Exception as e:
            print(f"Could not stat master CSV: {e}")

        # Parquet-копія (якщо є бекенд)
        try:
            parquet_path = os.path.join(results_dir, "all_results3.parquet")
            combined.to_parquet(parquet_path, index=False)
            print(f"Parquet copy saved to {os.path.abspath(parquet_path)}")
        except Exception as e:
            print(f"Parquet not written (pyarrow/fastparquet missing?): {e}")

        # Невелике прев’ю
        preview_cols = [
            "direction","outcome","entry_time","exit_time","entry_price","exit_price",
            "stop","take","fib_level","rr","be_multiplier","setup_id","fvg_time",
            "h4_bos_dir","h4_bos_time","h4_confirm_time"
        ] + sum(([f"h4_{j}_dir", f"h4_{j}_confirm_time"] for j in range(1, MAX_H4_KEEP+1)), [])
        preview_cols = [c for c in preview_cols if c in combined.columns]

        preview_path = os.path.join(results_dir, "all_results3_preview.csv")
        try:
            combined.tail(1000).loc[:, preview_cols].to_csv(preview_path, index=False)
            print(f"Preview (last 1000 rows) saved to {os.path.abspath(preview_path)}")
        except Exception as e:
            print(f"Could not write preview CSV: {e}")
    else:
        print("No new rows produced; nothing to write.")
