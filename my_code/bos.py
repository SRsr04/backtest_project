import pandas as pd
import numpy as np

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
