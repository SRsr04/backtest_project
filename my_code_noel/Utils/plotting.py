import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf

def plotting_chart_technicals(ohlc, start, end):

    df = ohlc.copy()
    df = df[start:end].reset_index(drop=True)
    df = df.reset_index()
    df['time'] = pd.to_datetime(df['time'])
    df.index = pd.DatetimeIndex(df.time)
    
    # for col, default in [
    #     ('bos_flag', 0),
    #     ('bos_index', -1),
    #     ('bos_time', pd.NaT),
    #     ('bos_type', 0),
    #     ('bars_to_bos', np.nan),
    #     ('bos_price', np.nan)
    # ]:
    #     if col not in df.columns:
    #         df[col] = default

    tech = df[['fvg', 'fvg_bottom', 'fvg_top', 'fractal', 'fractal_level', 'index',]]
               #'bos_flag', 'bos_index', 'bos_time', 'bos_type', 'bos_price']]

    fig, ax = plt.subplots(figsize=(16, 9))

    mpf.plot(df, type='candle', ax=ax, volume=False, show_nontrading=False)

    for l, row in tech.iterrows():

        if not pd.isnull(row['fvg']):
            if row['fvg'] == 1:
                ax.hlines(y=row['fvg_top'], xmin = row['index'], xmax = row['index'] + 3, color='black', linestyle = '-', linewidth=1)
                ax.hlines(y=row['fvg_bottom'], xmin = row['index'] - 2, xmax = row['index'] + 3, color='black', linestyle = '-', linewidth=1)
            elif row['fvg'] == -1:
                ax.hlines(y=row['fvg_top'], xmin = row['index'] - 2, xmax = row['index'] + 3, color='black', linestyle = '-', linewidth=1)
                ax.hlines(y=row['fvg_bottom'], xmin = row['index'], xmax = row['index'] + 3, color='black', linestyle = '-', linewidth=1)
            else:
                None
        
        if not pd.isnull(row['fractal']):
            if row['fractal'] == 1:
                ax.scatter(x=row['index'] - 1, y=row['fractal_level'], color='black', marker='v', s=7)
            elif row['fractal'] == -1:
                ax.scatter(x=row['index'] - 1, y=row['fractal_level'], color='black', marker='^', s=7)
            else:
                None    
        
        # if int(row.get('bos_flag', 0)) == 1 and pd.notna(row.get('fractal_level', None)):
        #     start_idx = int(row['index'] - 1)
        #     end_idx = start_idx + 3

        #     # рівень для малювання — fractal_level (можна замінити на bos_price при бажанні)
        #     bos_level = row['fractal_level']
        #     color = 'black' if int(row.get('bos_type', 0)) == 1 else 'black'
        #     label = '↑' if int(row.get('bos_type', 0)) == 1 else '↓'

        #     # товста пунктирна лінія
        #     ax.hlines(bos_level, start_idx, end_idx, linestyle='-', linewidth=0.5)

        #     # стрілка напрямку пробою
        #     ax.annotate('',
        #                 xy=(end_idx, bos_level),
        #                 xytext=(start_idx, bos_level),
        #                 arrowprops=dict(arrowstyle='-|>', color=color, lw=0.5))

        #     # підпис біля кінця
        #     ax.text(end_idx + 0.1, bos_level, label,
        #             ha='center', va='center', color=color, fontsize=10, weight='light')


        
    plt.show()


