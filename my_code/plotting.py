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

def plotting_chart_fvg(ohlc, start, end):

    df = ohlc.copy()
    df = df[start:end].reset_index(drop=True)
    df = df.reset_index()
    df['time'] = pd.to_datetime(df['time'])
    df.index = pd.DatetimeIndex(df.time)

    fvg = df[['fvg', 'fvg_bottom', 'fvg_top', 'index']]

    fig, ax = plt.subplots(figsize=(16,9))

    mpf.plot(df, type='candle', ax=ax, volume=False, show_nontrading=False)
    
    for row in fvg.iterrows():
        if not pd.isnull(row['fvg']):
            if row['fvg'] == 1:
                ax.hlines(y = row['fvg_top'], xmin=row['index'], xmax=row['index'] + 20, color='green', linestyle='-', linewidth=2)
                ax.hlines(y = row['fvg_bottom'], xmin=row['index'] - 2, xmax=row['index'] + 20, color='green', linestyle='-', linewidth=2)
            elif row['fvg'] == -1:
                ax.hlines(y = row['fvg_top'], xmin=row['index'], xmax=row['index'] + 20, color='red', linestyle='-', linewidth=2)
                ax.hlines(y = row['fvg_bottom'], xmin=row['index'] - 2, xmax=row['index'] + 20, color='red', linestyle='-', linewidth=2)
            else:
                None
            
            
    plt.show()


