import pandas as pd
import numpy as np
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

    if 'active_fvg_id' not in df.columns:
        df['active_fvg_id'] = np.nan


    columns = [
        'index', 'time', 'open', 'high', 'low',
        'fractal', 'fractal_levels',
        'bos', 'bos_level',
        'bos_fractal_index', 'bos_fractal_index_local',
        'fvg', 'fvg_bottom', 'fvg_top', 'active_fvg_id',
    ]
    technicals_df = df[columns]


    fig, ax = plt.subplots(figsize=(16, 9))
    mpf.plot(df, type='candle', ax=ax, volume=False, show_nontrading=False)
    fvg_tail = 10

    for _, row in technicals_df.iterrows():
        if row['fvg'] != 0 and pd.notna(row['fvg_top']) and pd.notna(row['fvg_bottom']):
            left = row['index']
            right = row['index'] + fvg_tail
            ax.hlines(row['fvg_top'], left, right, color='black', linestyle='--', linewidth=1)
            ax.hlines(row['fvg_bottom'], left, right, color='black', linestyle='--', linewidth=1)

        if row['fractal'] == 1 and pd.notna(row['fractal_levels']):
            padding = max((row['high'] - row['low']) * 0.05, 0.5)
            ax.scatter(row['index'], row['fractal_levels'] + padding,
                    color='green', marker='v', s=15)
        elif row['fractal'] == -1 and pd.notna(row['fractal_levels']):
            padding = max((row['high'] - row['low']) * 0.05, 0.5)
            ax.scatter(row['index'], row['fractal_levels'] - padding,
                    color='red', marker='^', s=15)

        if row['bos'] != 0 and pd.notna(row['bos_level']):
            start_time = row['bos_fractal_time']
            if pd.notna(start_time) and start_time in df['time'].values:
                start_idx = df.index.get_loc(start_time)
            else:
                start_idx = max(row['index'] - 4, 0)
            end_idx = min(row['index'] + 5, df['index'].iloc[-1])

            color = 'green' if row['bos'] == 1 else 'red'
            label = 'BOS↑' if row['bos'] == 1 else 'BOS↓'

            ax.annotate('', xy=(end_idx, row['bos_level']),
                        xytext=(start_idx, row['bos_level']),
                        arrowprops=dict(arrowstyle='-|>', color=color, lw=1.0))
            ax.text(end_idx + 0.4, row['bos_level'], label,
                    ha='left', va='center', color=color, fontsize=8, weight='light')

    if technicals_df['fvg'].abs().gt(0).any():
        if technicals_df['active_fvg_id'].notna().any():
            fvg_groups = (
                technicals_df[technicals_df['active_fvg_id'].notna()]
                .groupby('active_fvg_id')
                .agg(start_idx=('index', 'min'),
                    end_idx=('index', 'max'),
                    fvg_top=('fvg_top', 'last'),
                    fvg_bottom=('fvg_bottom', 'last'))
            )
            for _, grp in fvg_groups.iterrows():
                left = grp['start_idx']
                right = grp['end_idx'] + fvg_tail
                if pd.notna(grp['fvg_top']):
                    ax.hlines(grp['fvg_top'], left, right,
                            color='black', linestyle='--', linewidth=1)
                    ax.text(right + 0.1, grp['fvg_top'], f"{grp['fvg_top']:.2f}",
                            ha='left', va='center', color='black', fontsize=8)
                if pd.notna(grp['fvg_bottom']):
                    ax.hlines(grp['fvg_bottom'], left, right,
                            color='black', linestyle='--', linewidth=1)
                    ax.text(right + 0.1, grp['fvg_bottom'], f"{grp['fvg_bottom']:.2f}",
                            ha='left', va='center', color='black', fontsize=8)
        else:
            for _, row in technicals_df[technicals_df['fvg'].abs() > 0].iterrows():
                left = row['index']
                right = row['index'] + fvg_tail
                if pd.notna(row['fvg_top']):
                    ax.hlines(row['fvg_top'], left, right, color='black', linestyle='--', linewidth=1)
                    ax.text(right + 0.1, row['fvg_top'], f"{row['fvg_top']:.2f}",
                            ha='left', va='center', color='black', fontsize=8)
                if pd.notna(row['fvg_bottom']):
                    ax.hlines(row['fvg_bottom'], left, right, color='black', linestyle='--', linewidth=1)
                    ax.text(right + 0.1, row['fvg_bottom'], f"{row['fvg_bottom']:.2f}",
                            ha='left', va='center', color='black', fontsize=8)

plt.show()


