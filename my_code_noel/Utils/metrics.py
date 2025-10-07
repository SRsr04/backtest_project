import numpy as np
import pandas as pd

def calculate_metrics(profits):

    metrics = {}

    profits = np.asarray(profits, dtype=np.float64)
    profits = profits[np.isfinite(profits)]

    total_profit = np.sum(profits)
    avg_profit = np.mean(profits)
    win_trades = profits[profits > 0]
    loss_trades = profits[profits < 0]

    metrics['Total Trades'] = len(profits)
    metrics['Total Net Profit'] = total_profit
    metrics['Average Profit'] = avg_profit
    metrics['Win rate'] = len(win_trades) / len(profits)
    metrics['Average Win'] = np.mean(win_trades) if len(win_trades) > 0 else 0.0
    metrics['Average Loss'] = np.mean(loss_trades) if len(loss_trades) > 0 else 0.0
    metrics['Profit Factor'] = abs(np.sum(win_trades) / np.sum(loss_trades)) if np.sum(loss_trades) != 0 else 0.0

    return round(pd.DataFrame([metrics]), 2)