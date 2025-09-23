import pandas as pd
import numpy as np

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
