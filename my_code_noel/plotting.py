import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf


def plotting_chart(ohlc, start, end):

    df = ohlc.copy()
    df = df.reset_index()
    df['time'] = pd.to_datetime(df['time'])
    df.index = pd.DatetimeIndex(df.time)
    df = df[start:end]

    fig, ax = plt.subplots(figsize=(16, 9))

    mpf.plot(df, type='candle', ax=ax, volume=False, show_nontrading=False)
    plt.show()

def plotting_chart_technicals(ohlc, start, end):
    df = ohlc.copy()
    df['global_idx'] = df.index.to_numpy()

    df = df.loc[start:end].reset_index(drop=True)
    base_idx = df['global_idx'].iloc[0]

    df = df.reset_index()
    df['time'] = pd.to_datetime(df['time'])
    df.index = pd.DatetimeIndex(df['time'])
    df['bos_fractal_index_local'] = df['bos_fractal_index'] - base_idx

    technicals_df = df[['index', 'time', 'open', 'high', 'low',
                    'fractal', 'fractal_levels',
                    'bos', 'bos_level',
                    'bos_fractal_index', 'bos_fractal_index_local',
                    'fvg', 'fvg_bottom', 'fvg_top']]


    fig, ax = plt.subplots(figsize=(16, 9))
    mpf.plot(df, type='candle', ax=ax, volume=False, show_nontrading=False)
    fvg_tail = 10
    for _, row in technicals_df.iterrows():
        if not pd.isnull(row['fvg']):
            left = row['index'] - 2
            right = row['index'] + fvg_tail
            if row['fvg'] == 1:
                ax.hlines(row['fvg_top'], left + 2, right, color='black', linestyle='--', linewidth=1)
                ax.hlines(row['fvg_bottom'], left, right, color='black', linestyle='--', linewidth=1)
            elif row['fvg'] == -1:
                ax.hlines(row['fvg_top'], left + 2, right, color='black', linestyle='--', linewidth=1)
                ax.hlines(row['fvg_bottom'], left, right, color='black', linestyle='--', linewidth=1)

            if row['fractal'] == 1:
                padding = max((row['high'] - row['low']) * 0.05, 0.5)  # обери під свої масштаби
                ax.scatter(row['index'] - 1,
                        row['fractal_levels'] + padding,
                        color='green', marker='v', s=15)
            elif row['fractal'] == -1:
                padding = max((row['high'] - row['low']) * 0.05, 0.5)
                ax.scatter(row['index'] - 1,
                        row['fractal_levels'] - padding,
                        color='red', marker='^', s=15)

        if row['bos'] != 0 and not pd.isna(row['bos_level']):
            start_idx = row['bos_fractal_index_local']
            if pd.isna(start_idx):
                start_idx = max(row['index'] - 4, 0)
            else:
                start_idx = max(start_idx, 0)

            end_idx = min(row['index'] + 5, df['index'].iloc[-1])

            color = 'green' if row['bos'] == 1 else 'red'
            label = 'BOS↑' if row['bos'] == 1 else 'BOS↓'

            ax.annotate('', xy=(end_idx, row['bos_level']),
                        xytext=(start_idx, row['bos_level']),
                        arrowprops=dict(arrowstyle='-|>', color=color, lw=1.0))
            ax.text(end_idx + 0.4, row['bos_level'], label,
                    ha='left', va='center', color=color, fontsize=8, weight='light')
    for _, row in technicals_df.iterrows():
        if not pd.isnull(row['fvg']):
            if row['fvg'] == 1:
                ax.hlines(row['fvg_top'], row['index'], row['index'] + 10, color='black', linestyle='--', linewidth=1)
                ax.hlines(row['fvg_bottom'], row['index'] - 2, row['index'] + 10, color='black', linestyle='--', linewidth=1)
            elif row['fvg'] == -1:
                ax.hlines(row['fvg_top'], row['index'], row['index'] + 10, color='black', linestyle='--', linewidth=1)
                ax.hlines(row['fvg_bottom'], row['index'] - 2, row['index'] + 10, color='black', linestyles='--', linewidth=1)

            if row['fractal'] == 1:
                padding = max((row['high'] - row['low']) * 0.05, 0.5)  # обери під свої масштаби
                ax.scatter(row['index'] - 1,
                        row['fractal_levels'] + padding,
                        color='green', marker='v', s=15)
            elif row['fractal'] == -1:
                padding = max((row['high'] - row['low']) * 0.05, 0.5)
                ax.scatter(row['index'] - 1,
                        row['fractal_levels'] - padding,
                        color='red', marker='^', s=15)

        if row['bos'] != 0 and not pd.isna(row['bos_level']):
            start_idx = row['bos_fractal_index_local']
            if pd.isna(start_idx):
                start_idx = max(row['index'] - 4, 0)
            else:
                start_idx = max(start_idx, 0)

            end_idx = min(row['index'] + 5, df['index'].iloc[-1])

            color = 'green' if row['bos'] == 1 else 'red'
            label = 'BOS↑' if row['bos'] == 1 else 'BOS↓'

            ax.annotate('', xy=(end_idx, row['bos_level']),
                        xytext=(start_idx, row['bos_level']),
                        arrowprops=dict(arrowstyle='-|>', color=color, lw=1.0))
            ax.text(end_idx + 0.4, row['bos_level'], label,
                    ha='left', va='center', color=color, fontsize=8, weight='light')
    
    fvg_groups = (
            
        df[df['active_fvg_id'].notna()]
        .groupby('active_fvg_id')
        .agg({'index': 'max', 'fvg_top': 'last', 'fvg_bottom': 'last'})
    )

    for _, grp in fvg_groups.iterrows():
        x = grp['index'] + fvg_tail + 0.3  # трохи праворуч від кінця смуги
        ax.text(x, grp['fvg_top'], f"{grp['fvg_top']:.2f}",
                ha='left', va='center', color='black', fontsize=8)
        ax.text(x, grp['fvg_bottom'], f"{grp['fvg_bottom']:.2f}",
                ha='left', va='center', color='black', fontsize=8)

    plt.show()

    plt.show()


