import numpy as np
import pandas as pd

from Utils.data_proccessing import get_historical_ohlc
from pybit.unified_trading import HTTP
from constants import API_KEY, API_SECRET

from strategy import get_signal
from Utils.tester import run_backtest
from technicals import detect_fractal, detect_fvg
from Utils.metrics import calculate_metrics


symbols_list = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']
tf_list = ['240', '60', '15', '5']

all_results = []

for symbol in symbols_list:
    for tf in tf_list:


        session = HTTP(demo=True, api_key=API_KEY, api_secret=API_SECRET)
        ohlc = get_historical_ohlc(session=session, symbol=symbol, interval=tf)

        results = {key: [0,0] for key in ['fvg', 'fvg_top', 'fvg_bottom', 'fractal', 'fractal_level', 'signal', 'stop_loss', 'take_profit']}

        for i in range(2, len(ohlc)):
            
            last_3_candels = ohlc.iloc[i-2:i+1]
            fvg, fvg_top, fvg_bottom = detect_fvg(last_3_candels)
            fractal, fractal_level = detect_fractal(last_3_candels)
            results['fvg'].append(fvg)
            results['fvg_top'].append(fvg_top)
            results['fvg_bottom'].append(fvg_bottom)
            results['fractal'].append(fractal)
            results['fractal_level'].append(fractal_level)

            current_candle = last_3_candels.iloc[-1].copy()
            current_candle['fvg'] = fvg
            current_candle['fvg_top'] = fvg_top
            current_candle['fvg_bottom'] = fvg_bottom

            signal, stop_loss, take_profit = get_signal(current_candle=current_candle)

            results['signal'].append(signal)
            results['stop_loss'].append(stop_loss)
            results['take_profit'].append(take_profit)


        for key, values in results.items():
            ohlc[key] = pd.Series(values, index=ohlc.index, name=key)

        ohlc['take_profit'] = pd.to_numeric(ohlc['take_profit'], errors='coerce')
        ohlc['stop_loss'] = pd.to_numeric(ohlc['stop_loss'], errors='coerce')


        trades_data = run_backtest(ohlc)
        trades_data.to_csv('trades_data.csv', index=False)
        profits = trades_data['profit']
        results_metrics = calculate_metrics(profits)
        results_metrics['symbol'] = symbol
        results_metrics['tf'] = tf
        all_results.append(results_metrics)

all_results = pd.concat(all_results, axis=0)
all_results.to_csv('all_results.csv', index=False)
print(all_results)




