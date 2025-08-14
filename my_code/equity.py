import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ── Налаштування ───────────────────────────────────────────────────────────────
# Базова папка з результатами
RESULTS_DIR = '/Users/synyshyn_04/BTC_backtest_new/backtest_results'

# ТРИ файли, які треба опрацювати ТА одразу вивести в термінал
FILES = [
    '0R_H1F_f0.618_s10.0_r5.0_n2.csv',
    '0R_NOH1_f0.618_s10.0_r5.0_n2.csv',
    '0R_NOH1_f0.5_s10.0_r6.0_n2.csv',
]

# Увімкнути/вимкнути фільтрацію вихідних (беремо тільки будні)
FILTER_WEEKDAYS = True

# Чи вирізати найгірші години (перелік нижче)
PRUNE_BAD_HOURS = True

# Мапа "поганих" годин для кожного з трьох сетапів (за останнім твоїм виводом)
BAD_HOURS_BY_FILE = {
    '0R_H1F_f0.618_s10.0_r5.0_n2.csv': {7, 10, 11, 13, 16, 20, 21, 22, 23},
    '0R_NOH1_f0.618_s10.0_r5.0_n2.csv': {7, 10, 11, 13, 16, 20, 21, 22, 23},
    '0R_NOH1_f0.5_s10.0_r6.0_n2.csv'  : {7, 9, 10, 11, 13, 16, 21, 22, 23},
}

# Зберігати графіки? Якщо None — показувати у вікні; якщо рядок — зберігати у файл(c) в цій папці
OUTPUT_DIR = None  # приклад: 'plots'
# ───────────────────────────────────────────────────────────────────────────────


def _compute_trade_return(row: pd.Series) -> float:
    """
    PnL у відсотках (де 0.01 = +1%).
    Використовує тільки entry/exit та direction. Комісію тут НЕ враховуємо,
    бо вона вже повинна бути врахована в сирих CSV (або додавай тут за потреби).
    """
    ep = float(row['entry_price'])
    xp = float(row['exit_price'])
    if row['direction'] == 'long':
        return (xp - ep) / ep
    else:
        return (ep - xp) / ep


def compute_equity_curve(returns: pd.Series) -> pd.Series:
    eq = (1 + returns.fillna(0)).cumprod()
    eq.name = 'equity'
    return eq


def compute_drawdowns(equity: pd.Series) -> pd.DataFrame:
    hw = equity.cummax()
    dd = equity - hw
    dd_pct = dd / hw.replace(0, np.nan)
    return pd.DataFrame({'equity': equity, 'highwater': hw, 'drawdown': dd, 'drawdown_pct': dd_pct})


def max_drawdown(ddf: pd.DataFrame):
    md_usd = ddf['drawdown'].min()
    md_pct = ddf['drawdown_pct'].min()
    return md_usd, md_pct


def annual_return(equity: pd.Series, periods_per_year=252) -> float:
    n = len(equity)
    if n <= 1:
        return np.nan
    total = equity.iloc[-1] / equity.iloc[0]
    return total ** (periods_per_year / n) - 1


def calmar_ratio(equity: pd.Series, dd: pd.DataFrame, periods_per_year=252) -> float:
    ann = annual_return(equity, periods_per_year)
    _, md_pct = max_drawdown(dd)
    return (ann / abs(md_pct)) if (md_pct is not None and md_pct < 0) else np.nan


def _metrics_from_returns(returns: pd.Series) -> dict:
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    pf = (wins.sum() / abs(losses.sum())) if losses.sum() != 0 else np.inf

    eq = compute_equity_curve(returns)
    dd = compute_drawdowns(eq)
    md_usd, md_pct = max_drawdown(dd)
    r_over_d = (eq.iloc[-1] - 1) / abs(md_pct) if (md_pct is not None and md_pct < 0) else np.nan

    return {
        'Trades Count': len(returns),
        'Win Rate %': (len(wins) / len(returns) * 100) if len(returns) else 0.0,
        'Avg Win %': wins.mean() * 100 if len(wins) else 0.0,
        'Avg Loss %': losses.mean() * 100 if len(losses) else 0.0,
        'Profit Factor': pf,
        'Net Return %': (eq.iloc[-1] - 1) * 100 if len(eq) else 0.0,
        'Max Drawdown %': md_pct * 100 if md_pct is not None else np.nan,
        'Return/Drawdown': r_over_d,
        'Calmar': calmar_ratio(eq, dd),
    }


def _plot_equity(eq_df: pd.DataFrame, title: str, out_path: str | None):
    dd = compute_drawdowns(eq_df['equity'])
    plt.figure(figsize=(10, 5))
    plt.plot(dd.index, dd['equity'], label='Equity')
    plt.fill_between(dd.index, dd['highwater'], dd['equity'],
                     where=(dd['drawdown'] < 0), alpha=0.3, label='Drawdown')
    plt.title(title)
    plt.xlabel('Trade #')
    plt.ylabel('Growth Factor')
    plt.legend()
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, dpi=300)
        plt.close()
    else:
        plt.show()


def _prepare_df(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {'entry_price', 'exit_price', 'direction'}
    if not required.issubset(df.columns):
        raise ValueError(f"{os.path.basename(csv_path)}: потрібні колонки {required}")

    # відкидаємо відсутні виходи
    df = df.dropna(subset=['exit_price']).copy()

    # нормалізуємо час (якщо є)
    time_col = None
    for cand in ['entry_time', 'entry_time_m1', 'entry_time_m5']:
        if cand in df.columns:
            time_col = cand
            break
    if time_col:
        df['entry_time'] = pd.to_datetime(df[time_col])
        try:
            df['entry_time'] = df['entry_time'].dt.tz_localize(None)
        except TypeError:
            pass
        df['hour'] = df['entry_time'].dt.hour
        df['weekday'] = df['entry_time'].dt.weekday
    else:
        # якщо часу немає — створимо сурогатний індекс/годину (прюнінг годин тоді не спрацює)
        df['entry_time'] = pd.NaT
        df['hour'] = -1
        df['weekday'] = -1

    # будні, якщо потрібно
    if FILTER_WEEKDAYS and 'weekday' in df.columns:
        df = df[df['weekday'] < 5].copy()

    # підрахунок по трейдам
    df['ret'] = df.apply(_compute_trade_return, axis=1)
    return df


def _print_block(title: str, metrics: dict):
    print(title)
    print(f"  Trades Count   : {metrics['Trades Count']}")
    print(f"  Win Rate %     : {metrics['Win Rate %']:.2f}")
    print(f"  Profit Factor  : {metrics['Profit Factor']:.3f}")
    print(f"  Avg Win %      : {metrics['Avg Win %']:.3f}")
    print(f"  Avg Loss %     : {metrics['Avg Loss %']:.3f}")
    print(f"  Net Return %   : {metrics['Net Return %']:.3f}")
    print(f"  Max Drawdown % : {metrics['Max Drawdown %']:.3f}")
    print(f"  Return/Drawdown: {metrics['Return/Drawdown']:.3f}")
    print(f"  Calmar         : {metrics['Calmar']:.3f}")
    print()


def process_one(basename: str):
    csv_path = basename if os.path.isabs(basename) else os.path.join(RESULTS_DIR, basename)
    if not os.path.exists(csv_path):
        print(f"[WARN] File not found: {csv_path}")
        return

    print("=" * 120)
    print(f"=== {os.path.basename(csv_path)} ===")

    df = _prepare_df(csv_path)

    # Метрики ДО прунінгу
    m_before = _metrics_from_returns(df['ret'])
    _print_block("— Before prune —", m_before)

    if PRUNE_BAD_HOURS:
        bad = BAD_HOURS_BY_FILE.get(os.path.basename(csv_path), set())
        if 'hour' in df.columns and len(bad) > 0 and df['hour'].ge(0).any():
            dfp = df[~df['hour'].isin(bad)].copy()
            m_after = _metrics_from_returns(dfp['ret'])
            _print_block(f"— After prune (removed hours: {sorted(bad)}) —", m_after)

            # графіки (опц.)
            if OUTPUT_DIR is not None:
                eq_before = compute_equity_curve(df['ret']).to_frame('equity')
                eq_after  = compute_equity_curve(dfp['ret']).to_frame('equity')
                stem = os.path.splitext(os.path.basename(csv_path))[0]
                _plot_equity(eq_before, f'Equity BEFORE prune — {stem}',
                             os.path.join(OUTPUT_DIR, f'{stem}_before_prune.png'))
                _plot_equity(eq_after,  f'Equity AFTER prune — {stem}',
                             os.path.join(OUTPUT_DIR, f'{stem}_after_prune.png'))
        else:
            print("— After prune — (skip; немає 'entry_time'/'hour' або не задано BAD_HOURS)")
    else:
        print("Prune disabled.")

    print()


def main():
    for fname in FILES:
        process_one(fname)

if __name__ == '__main__':
    main()