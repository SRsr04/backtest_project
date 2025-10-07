import pandas as pd
import numpy as np

def detect_fvg(last_3_candels):

    fvg = 0
    fvg_top = 0
    fvg_bottom = 0

    high = last_3_candels['high'].values
    low = last_3_candels['low'].values

    if low[-1] > high[0]:

        fvg = 1
        fvg_top = low[-1]
        fvg_bottom = high[0]

    elif high[-1] < low[0]:
        
        fvg = -1
        fvg_top = low[0]
        fvg_bottom = high[-1]

    return fvg, fvg_top, fvg_bottom


def detect_fractal(last_3_candels):
    
    fractal = 0
    fractal_level = 0

    high = last_3_candels['high'].values
    low = last_3_candels['low'].values

    if high[-2] > high[-1] and high[-2] > high[-3]:

        fractal += 1
        fractal_level = high[-2]

    if low[-2] < low[-1] and low[-2] < low[-3]:

        fractal -= 1
        fractal_level = low[-2]

    return fractal, fractal_level

def find_bos_from_fractals(df: pd.DataFrame, mode: str = 'close', max_lookahead = None):
    """
    Повертає копію df з додатковими колонками для рядків-фракталів:
      - fractal_price    : значення fractal_level у рядку фракталу
      - bos_close_price  : close свічки, що зробила BOS (NaN якщо не знайдено)
      - bos_index        : індекс (label) цієї свічки (NaN якщо не знайдено)
      - bos_time         : time цієї свічки (NaN якщо не знайдено)

    mode:
      - 'close' : вважаємо BOS коли close > fractal_level (для fractal==1)
                  або close < fractal_level (для fractal==-1)
      - 'body'  : вважаємо BOS коли мін(open,close) > fractal_level (для fractal==1)
                  або макс(open,close) < fractal_level (для fractal==-1)
    max_lookahead: максимальна кількість свічок у майбутнє (None -> до кінця)
    """
    df = df.copy()

    df['fractal_price'] = np.nan
    df['bos_close_price'] = np.nan
    df['bos_index'] = np.nan
    df['bos_time'] = pd.NaT

    # Індекси де є фрактали (і fractal_level не NaN)
    fractal_mask = df['fractal'].notna() & df['fractal_level'].notna()
    fractal_idxs = list(df.index[fractal_mask])

    pos_of_index = {idx: pos for pos, idx in enumerate(df.index)}
    n = len(df)

    for idx in fractal_idxs:
        f = df.at[idx, 'fractal']
        lvl = df.at[idx, 'fractal_level']
        df.at[idx, 'fractal_price'] = lvl

        start_pos = pos_of_index[idx] + 1
        if max_lookahead is None:
            end_pos = n
        else:
            end_pos = min(n, start_pos + int(max_lookahead))

        # Проходимо по майбутніх свічках
        found = False
        for pos in range(start_pos, end_pos):
            j = df.index[pos]
            o = float(df.iat[pos, df.columns.get_loc('open')])
            c = float(df.iat[pos, df.columns.get_loc('close')])

            if f == 1:
                if mode == 'close':
                    cond = c > lvl
                elif mode == 'body':
                    cond = min(o, c) > lvl
                else:
                    raise ValueError("mode must be 'close' or 'body'")
            elif f == -1:
                if mode == 'close':
                    cond = c < lvl
                elif mode == 'body':
                    cond = max(o, c) < lvl
                else:
                    raise ValueError("mode must be 'close' or 'body'")
            else:
                cond = False

            if cond:
                df.at[idx, 'bos_close_price'] = c
                df.at[idx, 'bos_index'] = j
                df.at[idx, 'bos_time'] = df.at[j, 'time'] if 'time' in df.columns else pd.NaT
                found = True
                break
    return df