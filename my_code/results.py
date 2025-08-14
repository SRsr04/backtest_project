import os
import glob
import re
import pandas as pd
import numpy as np
import logging
import sys
from datetime import timedelta

# Silence PeriodArray shuffle warnings
import warnings
warnings.filterwarnings(
    "ignore",
    message="you are shuffling a 'PeriodArray' object",
    category=UserWarning,
)

# =====================
# Config / Parameters
# =====================
logging.basicConfig(
    filename='stats_with_risk.log',
    filemode='w',
    level=logging.INFO,
    format='%(asctime)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# ==== Risk / Commission ====
INITIAL_BALANCE = 5000           # стартовий баланс
COMMISSION = 0.000325            # ОДНА ставка комісії на угоду (round-trip НЕ множимо)
RISK_REPORT = [0.005, 0.0075, 0.01, 0.015]  # 0.5%, 0.75%, 1%, 1.5% — для звітів

TOP_N = 3  # скільки топ-стратегій показувати у зведеннях

# =====================
# Utility functions
# =====================
def compute_hourly_stats(df):
    df = df.copy()
    df['hour'] = pd.to_datetime(df['entry_time']).dt.hour
    df = df[df['hour'].between(10, 12)]
    hourly = df.groupby('hour').agg(
        trades_count=('return', 'size'),
        win_rate=('outcome', lambda x: (x == 'take').mean() * 100),
        net_return=('return', 'sum'),
        avg_return=('return', 'mean'),
    ).reset_index()
    return hourly


def apply_position_sizing(df, initial_balance=INITIAL_BALANCE, risk_per_trade=0.01):
    df = df.copy()
    balance = initial_balance
    pct_returns = []
    equities = []

    for idx, r in df.iterrows():
        entry = r['entry_price']
        stop  = r['stop']
        exitp = r['exit_price']
        direction = r.get('direction', 'long')

        stop_dist = abs(entry - stop)
        if stop_dist <= 0 or pd.isna(exitp):
            pnl = 0.0
        else:
            # Фіксований $-ризик від стартового балансу (як у фазових симах)
            risk_usd = initial_balance * risk_per_trade
            size = risk_usd / stop_dist

            gross = (exitp - entry) * size if direction == 'long' else (entry - exitp) * size
            commission_cost = size * entry * COMMISSION  # комісія — ОДИН раз
            pnl = gross - commission_cost

        pct_return = pnl / balance if balance != 0 else 0.0
        pct_returns.append(pct_return)
        balance += pnl
        equities.append(balance)

        logging.info(
            f"Trade {idx}: dir={direction}, pnl={pnl:.2f}, balance={balance:.2f}, return={pct_return*100:.3f}%"
        )

    df['return'] = pct_returns
    df['equity'] = equities
    return df


def compute_metrics(df):
    net_return = (df['equity'].iloc[-1] / df['equity'].iloc[0] - 1) * 100
    rolling_max = df['equity'].cummax()
    drawdown = (df['equity'] - rolling_max) / rolling_max
    max_dd = drawdown.min() * 100

    wins = df[df['return'] > 0]['return']
    losses = df[df['return'] < 0]['return']
    win_rate = len(wins) / len(df) * 100 if len(df) > 0 else 0
    avg_win = wins.mean() * 100 if not wins.empty else 0
    avg_loss = losses.mean() * 100 if not losses.empty else 0

    pf = wins.sum() / (-losses.sum()) if losses.sum() < 0 else np.inf
    expectancy = (win_rate/100) * avg_win + (1 - win_rate/100) * avg_loss

    return {
        'Net Return %': net_return,
        'Max Drawdown %': max_dd,
        'Return/Drawdown': net_return / -max_dd if max_dd < 0 else np.nan,
        'Win Rate %': win_rate,
        'Avg Win %': avg_win,
        'Avg Loss %': avg_loss,
        'Profit Factor': pf,
        'Expectancy %': expectancy,
        'Trades Count': len(df)
    }


# ===== Helpers (PnL with commissions) =====

def compute_trade_pnls(df, balance, risk_pct):
    df = df.copy()
    risk_usd = balance * risk_pct
    df['position_size'] = risk_usd / (df['entry_price'] - df['stop']).abs()
    df['pnl'] = np.where(
        df['direction'] == 'long',
        (df['exit_price'] - df['entry_price']) * df['position_size'],
        (df['entry_price'] - df['exit_price']) * df['position_size']
    )
    commission_cost = df['position_size'] * df['entry_price'] * COMMISSION
    df['pnl'] = df['pnl'] - commission_cost
    return df


def summarize_pnls(df):
    wins = df[df['pnl'] > 0]['pnl']
    losses = df[df['pnl'] < 0]['pnl']
    return {
        'avg_win_usd': wins.mean(),
        'avg_loss_usd': losses.mean(),
        'win_rate': len(wins) / len(df) * 100 if len(df) else 0.0,
    }


def simulate_phase(trades_df, starting_balance, risk_pct, target_pct, max_dd_pct):
    balance = starting_balance
    peak = balance
    for idx, row in trades_df.iterrows():
        risk_usd = starting_balance * risk_pct
        pos_size = risk_usd / abs(row['entry_price'] - row['stop'])
        pnl = (row['exit_price'] - row['entry_price']) * pos_size if row['direction'] == 'long' \
              else (row['entry_price'] - row['exit_price']) * pos_size
        commission_cost = pos_size * row['entry_price'] * COMMISSION
        pnl -= commission_cost
        balance += pnl
        peak = max(peak, balance)
        dd = (peak - balance) / peak
        if dd > max_dd_pct:
            return False, balance, idx + 1
        if balance >= starting_balance * (1 + target_pct):
            return True, balance, idx + 1
    return False, balance, len(trades_df)


# ====== Prop-firm simulation with daily drawdown logic ======

def simulate_phase_prop(trades_df, starting_balance, risk_pct, target_pct, max_dd_pct=0.10, max_daily_dd_pct=0.05):
    df_sorted = trades_df.sort_values('entry_time').copy()
    balance = starting_balance
    peak = balance
    current_day = None
    day_start_balance = balance

    for idx, row in df_sorted.iterrows():
        day_key = pd.to_datetime(row['entry_time']).normalize()
        if (current_day is None) or (day_key != current_day):
            current_day = day_key
            day_start_balance = balance

        stop_dist = abs(row['entry_price'] - row['stop'])
        if stop_dist == 0 or pd.isna(row.get('exit_price')):
            continue

        risk_usd = starting_balance * risk_pct
        pos_size = risk_usd / stop_dist

        gross = (row['exit_price'] - row['entry_price']) * pos_size if row['direction'] == 'long' \
                else (row['entry_price'] - row['exit_price']) * pos_size
        commission_cost = pos_size * row['entry_price'] * COMMISSION
        pnl = gross - commission_cost
        balance += pnl

        peak = max(peak, balance)
        dd = (peak - balance) / peak if peak > 0 else 0.0
        if dd > max_dd_pct:
            return False, balance, idx + 1, 'max_dd'

        daily_dd = (day_start_balance - balance) / day_start_balance if day_start_balance > 0 else 0.0
        if daily_dd > max_daily_dd_pct:
            return False, balance, idx + 1, 'daily_dd'

        if balance >= starting_balance * (1 + target_pct):
            return True, balance, idx + 1, 'target'

    return False, balance, len(df_sorted), 'exhausted'


def slice_until_phase(trades_df, starting_balance, risk_pct, target_pct, max_dd_pct=0.10, max_daily_dd_pct=0.05):
    df_sorted = trades_df.sort_values('entry_time').copy()
    balance = starting_balance
    peak = balance
    current_day = None
    day_start_balance = balance

    cut_index = None
    reason = 'exhausted'
    success = False

    for idx, row in df_sorted.iterrows():
        day_key = pd.to_datetime(row['entry_time']).normalize()
        if (current_day is None) or (day_key != current_day):
            current_day = day_key
            day_start_balance = balance

        stop_dist = abs(row['entry_price'] - row['stop'])
        if stop_dist == 0 or pd.isna(row.get('exit_price')):
            continue

        risk_usd = starting_balance * risk_pct
        pos_size = risk_usd / stop_dist

        gross = (row['exit_price'] - row['entry_price']) * pos_size if row['direction'] == 'long' \
                else (row['entry_price'] - row['exit_price']) * pos_size
        commission_cost = pos_size * row['entry_price'] * COMMISSION
        pnl = gross - commission_cost
        balance += pnl

        peak = max(peak, balance)
        dd = (peak - balance) / peak if peak > 0 else 0.0
        if dd > max_dd_pct:
            cut_index = idx + 1
            reason = 'max_dd'
            success = False
            break

        daily_dd = (day_start_balance - balance) / day_start_balance if day_start_balance > 0 else 0.0
        if daily_dd > max_daily_dd_pct:
            cut_index = idx + 1
            reason = 'daily_dd'
            success = False
            break

        if balance >= starting_balance * (1 + target_pct):
            cut_index = idx + 1
            reason = 'target'
            success = True
            break

    if cut_index is None:
        cut_index = len(df_sorted)
        reason = 'exhausted'

    df_cut = df_sorted.iloc[:cut_index].copy()
    return df_cut, success, balance, cut_index, reason


def compute_dynamic_path(df, starting_balance, risk_pct):
    df_sorted = df.sort_values('entry_time').copy()
    balances, pnls, pct_returns = [], [], []
    balance = starting_balance
    for _, row in df_sorted.iterrows():
        stop_dist = abs(row['entry_price'] - row['stop'])
        if stop_dist == 0 or pd.isna(row.get('exit_price')):
            pnls.append(0.0)
            pct_returns.append(0.0)
            balances.append(balance)
            continue
        risk_usd = starting_balance * risk_pct
        pos_size = risk_usd / stop_dist
        gross = (row['exit_price'] - row['entry_price']) * pos_size if row['direction'] == 'long' \
                else (row['entry_price'] - row['exit_price']) * pos_size
        commission_cost = pos_size * row['entry_price'] * COMMISSION
        pnl = gross - commission_cost
        pnls.append(pnl)
        pct_returns.append(pnl / balance if balance != 0 else 0.0)
        balance += pnl
        balances.append(balance)
    df_sorted['pnl'] = pnls
    df_sorted['pct_return'] = pct_returns
    df_sorted['equity'] = balances
    return df_sorted


def compute_performance_metrics_percent(df, risk_pct=0.01):
    dfm = compute_dynamic_path(df, INITIAL_BALANCE, risk_pct)
    pct = dfm['pct_return'] * 100.0
    equity = dfm['equity']
    if len(equity) == 0:
        return {
            'Net Return %': 0.0,
            'Max Drawdown %': 0.0,
            'Return/Drawdown': np.nan,
            'Win Rate %': 0.0,
            'Avg Win %': 0.0,
            'Avg Loss %': 0.0,
            'Profit Factor': np.nan,
            'Expectancy %': 0.0,
            'Trades Count': 0,
        }
    net_return = (equity.iloc[-1] / equity.iloc[0] - 1.0) * 100.0
    highwater = equity.cummax()
    dd_pct_series = (equity / highwater - 1.0) * 100.0
    max_dd = dd_pct_series.min() if len(dd_pct_series) else 0.0

    wins_mask = pct > 0
    losses_mask = pct < 0
    win_rate = wins_mask.mean() * 100.0 if len(pct) else 0.0
    avg_win = pct[wins_mask].mean() if wins_mask.any() else 0.0
    avg_loss = pct[losses_mask].mean() if losses_mask.any() else 0.0

    pos_sum = dfm.loc[wins_mask, 'pnl'].sum()
    neg_sum = dfm.loc[losses_mask, 'pnl'].sum()
    pf = (pos_sum / abs(neg_sum)) if neg_sum < 0 else np.inf

    expectancy = (win_rate / 100.0) * avg_win + (1.0 - win_rate / 100.0) * avg_loss

    return {
        'Net Return %': net_return,
        'Max Drawdown %': max_dd,
        'Return/Drawdown': (net_return / -max_dd) if max_dd < 0 else np.nan,
        'Win Rate %': win_rate,
        'Avg Win %': avg_win,
        'Avg Loss %': avg_loss,
        'Profit Factor': pf,
        'Expectancy %': expectancy,
        'Trades Count': len(dfm),
    }


def out_of_sample_test(df, risk_pct=0.01):
    df_oos = df.copy()
    df_oos['entry_time'] = pd.to_datetime(df_oos['entry_time'])
    cutoff = df_oos['entry_time'].max() - timedelta(days=365)
    df_oos = df_oos[df_oos['entry_time'] >= cutoff].copy()
    if df_oos.empty:
        print("No Out-of-Sample data (last year) to test.")
        return
    print("\n=== Out-of-Sample (last year) Metrics ===")
    dyn = compute_dynamic_path(df_oos, INITIAL_BALANCE, risk_pct)
    net = (dyn['equity'].iloc[-1] / dyn['equity'].iloc[0] - 1.0) * 100.0
    wr = (dyn['pct_return'] > 0).mean() * 100.0
    print(f"Trades: {len(dyn)}, Net Return %: {net:.2f}, Win Rate %: {wr:.2f}")


def simulate_phase_prop_wrapper(df, risk_pct=0.01, target_pct=0.08, max_dd=0.10, max_daily_dd=0.05):
    return simulate_phase_prop(df, INITIAL_BALANCE, risk_pct, target_pct, max_dd, max_daily_dd)


def monte_carlo(df, runs=1000, target_pct=0.08, max_dd=0.10, max_daily_dd=0.05, risk_pct=0.01):
    failures = 0
    reasons = {'max_dd': 0, 'daily_dd': 0, 'exhausted': 0}
    for _ in range(runs):
        df_shuffled = df.sample(frac=1, random_state=None).reset_index(drop=True)
        success, _, _, reason = simulate_phase_prop_wrapper(df_shuffled, risk_pct, target_pct, max_dd, max_daily_dd)
        if not success:
            failures += 1
            if reason in reasons:
                reasons[reason] += 1
    rate = failures / runs * 100.0
    print(f"\n=== Monte Carlo ({runs} runs) Failure Rate === {failures}/{runs} ({rate:.1f}%)")
    if failures:
        print(f"  breakdown: max_dd={reasons['max_dd']}, daily_dd={reasons['daily_dd']}, exhausted={reasons['exhausted']}")


def stress_test(df, drop_frac=0.05, risk_pct=0.01):
    dyn = compute_dynamic_path(df.copy(), INITIAL_BALANCE, risk_pct)
    pct = dyn['pct_return'] * 100.0
    cutoff = pct.quantile(1 - drop_frac)
    pct_stripped = pct[pct < cutoff]
    if pct_stripped.empty:
        print(f"\n=== Stress Test (drop top {int(drop_frac*100)}%) Expectancy %: n/a")
        return
    w = (pct_stripped > 0).mean()
    aw = pct_stripped[pct_stripped > 0].mean() if (pct_stripped > 0).any() else 0.0
    al = pct_stripped[pct_stripped < 0].mean() if (pct_stripped < 0).any() else 0.0
    exp = w * aw + (1 - w) * al
    print(f"\n=== Stress Test (drop top {int(drop_frac*100)}%) Expectancy %: {exp:.2f}")


# ===== Monte Carlo scenario helpers =====

def _estimate_trades_to_target(df, target_pct=0.08, max_dd=0.10, max_daily_dd=0.05, risk_pct=0.01):
    df_sorted = df.sort_values('entry_time').copy()
    success, _, trades_used, _ = simulate_phase_prop_wrapper(df_sorted, risk_pct, target_pct, max_dd, max_daily_dd)
    if success:
        return trades_used
    return min(len(df_sorted), 250)


def _sample_contiguous_block(df_sorted, length):
    if length >= len(df_sorted):
        return df_sorted.copy()
    start = np.random.randint(0, len(df_sorted) - length + 1)
    return df_sorted.iloc[start:start+length].copy()


def _sample_month_blocks(df_sorted, min_trades):
    dfm = df_sorted.copy()
    dfm['month'] = dfm['entry_time'].dt.to_period('M')
    months = dfm['month'].dropna().unique()  # PeriodArray
    if len(months) == 0:
        return df_sorted.copy()

    perm = np.random.permutation(len(months))

    picked = []
    total = 0
    for i in perm:
        m = months[i]
        chunk = dfm[dfm['month'] == m].drop(columns=['month'])
        picked.append(chunk)
        total += len(chunk)
        if total >= min_trades:
            break

    res = pd.concat(picked, ignore_index=True) if picked else df_sorted.copy()
    return res.sort_values('entry_time').iloc[:min_trades].copy()


def rolling_windows_pass_rate(df, window_len, target_pct=0.08, max_dd=0.10, max_daily_dd=0.05, risk_pct=0.01):
    df_sorted = df.sort_values('entry_time').copy()
    n = len(df_sorted)
    if window_len >= n:
        success, _, used, reason = simulate_phase_prop_wrapper(df_sorted, risk_pct, target_pct, max_dd, max_daily_dd)
        return (1.0 if success else 0.0), {reason: 1}, used
    passes = 0
    reasons = {}
    used_list = []
    for s in range(0, n - window_len + 1):
        seg = df_sorted.iloc[s:s+window_len]
        success, _, used, reason = simulate_phase_prop_wrapper(seg, risk_pct, target_pct, max_dd, max_daily_dd)
        passes += int(success)
        reasons[reason] = reasons.get(reason, 0) + 1
        used_list.append(used)
    pass_rate = passes / (n - window_len + 1)
    med_used = float(np.median(used_list)) if used_list else 0.0
    return pass_rate, reasons, med_used


def monte_carlo_scenarios(df, runs=500, target_pct=0.08, max_dd=0.10, max_daily_dd=0.05, risk_pct=0.01):
    df_sorted = df.sort_values('entry_time').copy()
    L = _estimate_trades_to_target(df_sorted, target_pct, max_dd, max_daily_dd, risk_pct)

    # 1) iid shuffle
    iid_pass = 0
    iid_used = []
    iid_reasons = {'max_dd': 0, 'daily_dd': 0, 'exhausted': 0}
    for _ in range(runs):
        shuf = df_sorted.sample(frac=1, random_state=None).reset_index(drop=True)
        success, _, used, reason = simulate_phase_prop_wrapper(shuf, risk_pct, target_pct, max_dd, max_daily_dd)
        iid_pass += int(success)
        iid_used.append(used)
        if not success:
            if reason in iid_reasons:
                iid_reasons[reason] += 1
    iid_row = {
        'scenario': 'iid_shuffle',
        'runs': runs,
        'pass_rate_%': iid_pass / runs * 100.0,
        'median_trades': float(np.median(iid_used)) if iid_used else 0.0,
        'fail_max_dd': iid_reasons['max_dd'],
        'fail_daily_dd': iid_reasons['daily_dd'],
        'fail_exhausted': iid_reasons['exhausted'],
    }

    # 2) contiguous block of length L
    block_pass = 0
    block_used = []
    block_reasons = {'max_dd': 0, 'daily_dd': 0, 'exhausted': 0}
    for _ in range(runs):
        seg = _sample_contiguous_block(df_sorted, L)
        success, _, used, reason = simulate_phase_prop_wrapper(seg, risk_pct, target_pct, max_dd, max_daily_dd)
        block_pass += int(success)
        block_used.append(used)
        if not success and reason in block_reasons:
            block_reasons[reason] += 1
    block_row = {
        'scenario': f'contiguous_L={L}',
        'runs': runs,
        'pass_rate_%': block_pass / runs * 100.0,
        'median_trades': float(np.median(block_used)) if block_used else 0.0,
        'fail_max_dd': block_reasons['max_dd'],
        'fail_daily_dd': block_reasons['daily_dd'],
        'fail_exhausted': block_reasons['exhausted'],
    }

    # 3) month-bag (sample months until >= L trades)
    month_pass = 0
    month_used = []
    month_reasons = {'max_dd': 0, 'daily_dd': 0, 'exhausted': 0}
    for _ in range(runs):
        seg = _sample_month_blocks(df_sorted, L)
        success, _, used, reason = simulate_phase_prop_wrapper(seg, risk_pct, target_pct, max_dd, max_daily_dd)
        month_pass += int(success)
        month_used.append(used)
        if not success and reason in month_reasons:
            month_reasons[reason] += 1
    month_row = {
        'scenario': 'month_bag',
        'runs': runs,
        'pass_rate_%': month_pass / runs * 100.0,
        'median_trades': float(np.median(month_used)) if month_used else 0.0,
        'fail_max_dd': month_reasons['max_dd'],
        'fail_daily_dd': month_reasons['daily_dd'],
        'fail_exhausted': month_reasons['exhausted'],
    }

    # 4) rolling windows over the whole series
    roll_rate, roll_reasons, roll_med_used = rolling_windows_pass_rate(df_sorted, L, target_pct, max_dd, max_daily_dd, risk_pct)
    roll_row = {
        'scenario': f'rolling_L={L}',
        'runs': max(1, len(df_sorted) - L + 1),
        'pass_rate_%': roll_rate * 100.0,
        'median_trades': roll_med_used,
        'fail_max_dd': roll_reasons.get('max_dd', 0),
        'fail_daily_dd': roll_reasons.get('daily_dd', 0),
        'fail_exhausted': roll_reasons.get('exhausted', 0),
    }

    out = pd.DataFrame([iid_row, block_row, month_row, roll_row])
    return out


def trades_to_target_stats(df, runs=500, target_pct=0.08, max_dd=0.10, max_daily_dd=0.05, risk_pct=0.01):
    df_sorted = df.sort_values('entry_time').copy()
    success_c, _, used_c, _ = simulate_phase_prop_wrapper(df_sorted, risk_pct, target_pct, max_dd, max_daily_dd)
    chrono_used = used_c

    used_iid = []
    for _ in range(runs):
        shuf = df_sorted.sample(frac=1, random_state=None).reset_index(drop=True)
        success, _, used, _ = simulate_phase_prop_wrapper(shuf, risk_pct, target_pct, max_dd, max_daily_dd)
        used_iid.append(used)

    out = {
        'chrono': int(chrono_used),
        'iid_median': float(np.median(used_iid)),
        'iid_p10': float(np.percentile(used_iid, 10)),
        'iid_p90': float(np.percentile(used_iid, 90)),
    }
    return out


# =====================
# Main script
# =====================
if __name__ == '__main__':
    # 0) Root results directory
    results_dir = '/Users/synyshyn_04/BTC_backtest_new/backtest_results'
    print(f"Аналізуємо файли в директорії: {results_dir}")

    # 1) Collect all CSVs
    pattern = os.path.join(results_dir, '**', '*.csv')
    all_files = glob.glob(pattern, recursive=True)
    # Drop summary files
    all_files = [
        f for f in all_files
        if not os.path.basename(f).startswith('summary')
        and not os.path.basename(f).endswith('_hourly.csv')
    ]
    print(f"Знайдено {len(all_files)} файлів з результатами угод.")

    # 2) Load all trades + parse params from filename
    all_trades_list = []
    for f_path in all_files:
        df = pd.read_csv(f_path)
        if df.empty:
            continue

        # If some files might lack these columns, guard:
        required_cols = {'entry_time','exit_time','entry_price','exit_price','stop','direction','outcome'}
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            print(f" ! Skip {os.path.basename(f_path)}: missing cols {missing}")
            continue

        df['entry_time'] = pd.to_datetime(df['entry_time'])
        df['exit_time']  = pd.to_datetime(df['exit_time'])

        filename = os.path.basename(f_path)
        base = filename[:-4]
        parts = base.split('_')
        # Expect: ["1R","H1F"|"NOH1","f0.75","s10.0","r4.4","n2"]
        try:
            be_mult     = float(parts[0][:-1])
            h1f         = (parts[1] == 'H1F')
            fib_level   = float(parts[2][1:])
            stop_offset = float(parts[3][1:])
            rr          = float(parts[4][1:])
            max_trades  = int(parts[5][1:])
        except Exception:
            # If naming differs, keep going with defaults but store filename
            be_mult, h1f, fib_level, stop_offset, rr, max_trades = (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)

        df['filename']      = filename
        df['be_multiplier'] = be_mult
        df['h1_filter']     = h1f
        df['fib_level']     = fib_level
        df['stop_offset']   = stop_offset
        df['rr']            = rr
        df['max_trades']    = max_trades

        all_trades_list.append(df)

    if not all_trades_list:
        print("Не вдалося завантажити жодної угоди. Аналіз неможливий.")
        sys.exit(0)

    # 3) Concatenate
    master_df = pd.concat(all_trades_list, ignore_index=True)
    master_df['entry_time'] = pd.to_datetime(master_df['entry_time'])
    master_df['exit_time']  = pd.to_datetime(master_df['exit_time'])
    print(f"Всього завантажено {len(master_df)} угод з усіх файлів.")

    # 4) Filters: hours 07–23, same-day, weekdays
    master_df = master_df[master_df['entry_time'].dt.hour.between(7, 23)]
    master_df = master_df[
        master_df['entry_time'].dt.normalize() == master_df['exit_time'].dt.normalize()
    ]
    master_df = master_df[master_df['entry_time'].dt.weekday < 5]

    # 5) Limit trades per day by max_trades
    group_cols = [
        'filename', 'be_multiplier', 'h1_filter',
        'fib_level', 'stop_offset', 'rr', 'max_trades'
    ]
    master_df = (
        master_df.sort_values('entry_time')
        .groupby([*group_cols, master_df['entry_time'].dt.date])
        .apply(lambda g: g.head(int(g['max_trades'].iloc[0])))
        .reset_index(drop=True)
    )

    print(f"Після фільтрації залишилось {len(master_df)} угод для аналізу.")

    # 6) Summary metrics per parameter set for EACH risk
    summary_all = []
    for risk in RISK_REPORT:
        summary_list = []
        for params, group_df in master_df.groupby(group_cols):
            metrics = compute_performance_metrics_percent(group_df, risk_pct=risk)
            param_dict = dict(zip(group_cols, params))
            summary_list.append({**param_dict, **metrics, 'risk_pct': risk})
        if not summary_list:
            continue
        sdf = pd.DataFrame(summary_list)
        summary_all.append(sdf)

        # Top-N by Profit Factor (overall, не за BE)
        display_cols = [
            'risk_pct', 'be_multiplier', 'h1_filter',
            'fib_level', 'stop_offset', 'rr',
            'max_trades', 'Trades Count',
            'Profit Factor', 'Net Return %',
            'Avg Win %', 'Avg Loss %', 'filename',
        ]
        print(f"\n=== Top {TOP_N} сетапів за Profit Factor (risk={risk*100:.2f}%) ===")
        print(sdf.nlargest(TOP_N, 'Profit Factor')[display_cols].to_string(index=False))

    if not summary_all:
        print("summary_df порожній — немає даних для Top-N попереднього перегляду")
        sys.exit(0)

    summary_df = pd.concat(summary_all, ignore_index=True)

    # 7) Консенсусний Top-3 через усі ризики (середній ранг PF і R/D)
    rank_df = summary_df.copy()
    # Використовуємо transform (а не apply), щоб індекси співпадали з фреймом
    rank_df['rank_pf'] = rank_df.groupby('risk_pct')['Profit Factor'] \
                                .transform(lambda s: s.rank(ascending=False, method='average'))
    # Для R/D спершу підмінимо NaN на -inf, потім також transform
    rd_series = rank_df['Return/Drawdown'].fillna(-np.inf)
    rank_df['rank_rd'] = rd_series.groupby(rank_df['risk_pct']) \
                                 .transform(lambda s: s.rank(ascending=False, method='average'))

    cons = (rank_df.groupby('filename', as_index=False)[['rank_pf','rank_rd']].mean())
    cons['cons_rank'] = (cons['rank_pf'] + cons['rank_rd']) / 2

    meta = (summary_df.sort_values('risk_pct')
                      .drop_duplicates('filename')
                      [['filename','be_multiplier','h1_filter','fib_level','stop_offset','rr','max_trades']])
    cons = cons.merge(meta, on='filename', how='left')
    top3_cons = cons.nsmallest(TOP_N, 'cons_rank')

    print("\n=== Консенсусний Top 3 сетапів через усі ризики (середній ранг PF та R/D) ===")
    print(top3_cons[['filename','cons_rank','be_multiplier','h1_filter','fib_level','stop_offset','rr','max_trades']].to_string(index=False))

    # Збережемо консенсусний топ-3 у CSV
    top3_path = os.path.join(results_dir, 'summary_top3_consensus.csv')
    top3_cons.to_csv(top3_path, index=False)

    # 8) Top Strategies for Prop (NO DD FILTER — маркуємо DD≤10%)
    print("\n=== Top Strategies for Prop (NO DD filter; DD_OK_10 flag) ===")
    reps = summary_df.drop_duplicates('filename')[
        ['filename','be_multiplier','h1_filter','fib_level','stop_offset','rr','max_trades','Trades Count']
    ]

    prop_rows = []
    for _, rep in reps.iterrows():
        fname = rep['filename']
        fpath = os.path.join(results_dir, fname)
        try:
            trades_df = pd.read_csv(fpath)
            trades_df = trades_df.dropna(subset=['exit_price']).copy()
            trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
            trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])

            # метрики при дефолтному ризику (1%), щоб порівнювати проп-логіку стабільно
            perf = compute_performance_metrics_percent(trades_df, risk_pct=0.01)
            success8, _, trades8, _ = simulate_phase_prop_wrapper(trades_df.copy(), risk_pct=0.01, target_pct=0.08, max_dd=0.10, max_daily_dd=0.05)

            row_out = {**rep.to_dict(), **perf,
                       'phase_pass_8': success8,
                       'phase_trades_8': trades8}
            prop_rows.append(row_out)
        except Exception as e:
            print(f" ! Skip {fname}: {e}")
            continue

    prop_df = pd.DataFrame(prop_rows)
    if prop_df.empty:
        print("— Немає кандидатів для ранжування —")
        sys.exit(0)

    prop_df['Max DD Abs %'] = prop_df['Max Drawdown %'].abs()
    prop_df['DD_OK_10'] = prop_df['Max DD Abs %'] <= 10.0
    prop_df['be_norm'] = np.where((prop_df['be_multiplier'] - prop_df['rr']).abs() < 1e-9,
                                  0.0, prop_df['be_multiplier'])

    ranked = prop_df.sort_values(['Return/Drawdown','Expectancy %','phase_trades_8'],
                                 ascending=[False, False, True])
    ranked = ranked.drop_duplicates(
        subset=['be_norm','h1_filter','fib_level','stop_offset','rr','max_trades'], keep='first'
    ).copy()

    print("\n=== Top 3 Strategies per BE (no DD filter; DD_OK_10 indicates DD≤10%) ===")
    prop_display_cols = [
        'DD_OK_10','be_norm','be_multiplier','h1_filter','fib_level','stop_offset','rr','max_trades',
        'Trades Count','Net Return %','Max Drawdown %','Return/Drawdown',
        'Win Rate %','Avg Win %','Avg Loss %','Profit Factor','Expectancy %','phase_trades_8','filename'
    ]

    for be_val, sub in ranked.groupby('be_norm', sort=False):
        sub_sorted = sub.sort_values(['Return/Drawdown','Expectancy %','phase_trades_8'],
                                     ascending=[False, False, True])
        top3 = sub_sorted.head(3)
        print(f"\n-- BE={be_val} --")
        if top3.empty:
            print("(немає кандидатів)")
            continue
        print(top3[prop_display_cols].to_string(index=False))

    # 9) Detailed analysis for Consensus Top-3
    print("\n=== Detailed Analysis for Consensus Top 3 ===")
    for _, crow in top3_cons.iterrows():
        fname = crow['filename']
        print(f"\n--- Strategy: {fname} ---")
        fpath = os.path.join(results_dir, fname)
        trades_df = pd.read_csv(fpath)
        trades_df = trades_df.dropna(subset=['exit_price']).copy()
        trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
        trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])

        # PnL Summary by Risk % (USD) — fixed-$ risk path
        results = []
        for rp in RISK_REPORT:
            dyn_rp = compute_dynamic_path(trades_df.copy(), INITIAL_BALANCE, rp)
            wins = dyn_rp[dyn_rp['pnl'] > 0]['pnl']
            losses = dyn_rp[dyn_rp['pnl'] < 0]['pnl']
            s = {
                'avg_win_usd': wins.mean(),
                'avg_loss_usd': losses.mean(),
                'win_rate': (dyn_rp['pnl'] > 0).mean() * 100.0,
                'risk_pct': rp,
            }
            results.append(s)
        stats_df = pd.DataFrame(results).set_index('risk_pct')
        print("=== PnL Summary by Risk % ===")
        print(stats_df)

        # Overall performance (including commission) — для кожного ризику
        for rp in RISK_REPORT:
            print(f"\nOverall performance (including commission) @ risk {rp*100:.2f}%:")
            perf = compute_performance_metrics_percent(trades_df, risk_pct=rp)
            for k, v in perf.items():
                print(f" {k}: {v}")

        # Phase-limited performance (stop at 8% або limit) — для кожного ризику
        for rp in RISK_REPORT:
            df_cut, success_cut, final_bal_cut, trades_used_cut, reason_cut = slice_until_phase(
                trades_df.copy(), INITIAL_BALANCE, rp, 0.08, 0.10, 0.05
            )
            print(f"\nPhase-limited performance (stop at 8% or limit) @ risk {rp*100:.2f}%:")
            perf_cut = compute_performance_metrics_percent(df_cut, risk_pct=rp)
            for k, v in perf_cut.items():
                print(f" {k}: {v}")
            print(f" status: {'PASS' if success_cut else 'FAIL'} | reason: {reason_cut} | trades_used: {trades_used_cut} | final_balance: {final_bal_cut:.2f}")

        # Phase simulations (по 8% і 5%) на КОЖНОМУ ризику
        for rp in RISK_REPORT:
            phases = [
                {'target_pct': 0.08, 'max_dd_pct': 0.10},
                {'target_pct': 0.05, 'max_dd_pct': 0.10},
            ]
            phase_rows = []
            for ph in phases:
                success, final_bal, trades_used, reason = simulate_phase_prop_wrapper(
                    trades_df.copy(), risk_pct=rp, target_pct=ph['target_pct'], max_dd=ph['max_dd_pct'], max_daily_dd=0.05
                )
                phase_rows.append({
                    'risk_%': rp*100,
                    'target_%': ph['target_pct'] * 100,
                    'success': success,
                    'final_balance': final_bal,
                    'trades_used': trades_used,
                    'reason': reason,
                })
            phase_df = pd.DataFrame(phase_rows).set_index(['risk_%','target_%'])
            print("\n=== Phase Simulation Results ===")
            print(phase_df)

        # OOS / Monte Carlo / Stress Test @ 1% (для стабільного порівняння)
        out_of_sample_test(trades_df, risk_pct=0.01)
        monte_carlo(trades_df, runs=1000, target_pct=0.08, max_dd=0.10, risk_pct=0.01)
        stress_test(trades_df, drop_frac=0.05, risk_pct=0.01)

        # Scenario-based Monte Carlo (Phase 8%, 1% ризик)
        scen_df = monte_carlo_scenarios(trades_df, runs=500, target_pct=0.08, max_dd=0.10, max_daily_dd=0.05, risk_pct=0.01)
        print("\n=== Monte Carlo Scenarios (Phase 8%, fixed $ risk @1%) ===")
        print(scen_df.to_string(index=False))

        # Trades-to-target (8% і 5%) @1%
        t8 = trades_to_target_stats(trades_df, runs=500, target_pct=0.08, max_dd=0.10, max_daily_dd=0.05, risk_pct=0.01)
        print(f"\nTrades to 8% (@1% risk): chrono={t8['chrono']}, iid median={t8['iid_median']:.0f} (p10–p90: {t8['iid_p10']:.0f}–{t8['iid_p90']:.0f})")
        t5 = trades_to_target_stats(trades_df, runs=500, target_pct=0.05, max_dd=0.10, max_daily_dd=0.05, risk_pct=0.01)
        print(f"Trades to 5%  (@1% risk): chrono={t5['chrono']}, iid median={t5['iid_median']:.0f} (p10–p90: {t5['iid_p10']:.0f}–{t5['iid_p90']:.0f})")