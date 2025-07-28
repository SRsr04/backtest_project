import pandas as pd
import numpy as np
import pytz
import logging
import os
from datetime import datetime

# Імпортуємо наші власні модулі
from strategy_configs import STRATEGIES
from backtest_engine import find_potential_setups, run_simulation
from utils import custom_date_parser, calculate_drawdown, calculate_max_losing_streak

def setup_logging():
    """Налаштовує логування для всього проєкту."""
    log_dir = 'logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file_path = os.path.join(log_dir, 'backtest_run.log')
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    if root_logger.handlers:
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
    file_handler = logging.FileHandler(log_file_path, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    logging.info("Логування налаштовано.")

def load_and_prepare_data(config):
    """
    Завантажує, обробляє та ОБ'ЄДНУЄ дані на основі конфігурації.
    """
    logging.info(f"Завантаження даних для стратегії: {config['name']}...")
    print(f"Завантаження даних для {config['name']}...")
    
    kiev_tz = pytz.timezone('Europe/Kiev')
    files = config['files']

    def handle_timezone(dt_series):
        if dt_series.dt.tz is None:
            return dt_series.dt.tz_localize(kiev_tz, ambiguous='NaT', nonexistent='NaT')
        return dt_series.dt.tz_convert(kiev_tz)

    def load_csv(file_path, date_col):
        df = pd.read_csv(file_path, encoding='utf-8')
        dt_series = pd.to_datetime(df[date_col].apply(custom_date_parser), errors='coerce')
        df[date_col] = handle_timezone(dt_series)
        df.dropna(subset=[date_col], inplace=True)
        return df.set_index(date_col).sort_index().drop_duplicates(keep='first')

    # Завантаження всіх даних з конфігурації
    df_main_candles = load_csv(files['candles'], 'datetime')
    if df_main_candles.empty: return pd.DataFrame()

    df_h1_candles = load_csv(files['h1_candles'], 'datetime')
    df_h4_candles = load_csv(files['h4_candles'], 'datetime')
    df_fvg = load_csv(files['fvg'], 'time')
    df_bos = load_csv(files['bos'], 'bos_time_kiev')
    
    # Перейменування колонок та підготовка даних
    df_fvg.rename(columns={'type': 'fvg_type', 'min': 'fvg_min', 'max': 'fvg_max'}, inplace=True)
    df_bos.rename(columns={'type': 'bos_type', 'level': 'bos_level'}, inplace=True)
    
    df_h1_candles['h1_direction'] = np.select(
        [df_h1_candles['close'] > df_h1_candles['open'], df_h1_candles['close'] < df_h1_candles['open']],
        ['bullish', 'bearish'], default='neutral'
    )
    df_h4_candles['h4_direction'] = np.select(
        [df_h4_candles['close'] > df_h4_candles['open'], df_h4_candles['close'] < df_h4_candles['open']],
        ['bullish', 'bearish'], default='neutral'
    )
    
    # Об'єднання в один DataFrame
    df_h1_candles = df_h1_candles.add_suffix('_h1')
    df_h4_candles = df_h4_candles.add_suffix('_h4')

    df_master = pd.merge_asof(df_main_candles, df_h1_candles, left_index=True, right_index=True, direction='backward')
    df_master = pd.merge_asof(df_master, df_h4_candles, left_index=True, right_index=True, direction='backward')

    if config.get('mitigation_tf') == 'm15':
        if 'm15_candles' in files:
            df_m15_candles = load_csv(files['m15_candles'], 'datetime')
            df_m15_candles = df_m15_candles.add_suffix('_m15')
            df_master = pd.merge_asof(df_master, df_m15_candles, left_index=True, right_index=True, direction='backward')
    
    df_master = df_master.join(df_fvg[['fvg_type', 'fvg_min', 'fvg_max']], how='left')
    df_master = df_master.join(df_bos[['bos_type', 'bos_level']], how='left')

    # Розрахунок фракталів та очистка
    df_master['is_high_fractal'] = (df_master['high'].shift(1) < df_master['high']) & (df_master['high'] > df_master['high'].shift(-1))
    df_master['is_low_fractal'] = (df_master['low'].shift(1) > df_master['low']) & (df_master['low'] < df_master['low'].shift(-1))

    columns_to_fill = {
        'fvg_type': '', 'bos_type': '', 
        'h1_direction_h1': '', 'h4_direction_h4': ''
    }
    df_master.fillna(value=columns_to_fill, inplace=True)
    df_master.fillna(0, inplace=True)
    
    logging.info(f"Підготовка даних завершена. Розмір фінального DataFrame: {len(df_master)}.")
    return df_master


if __name__ == "__main__":
    setup_logging()
    
    kiev_tz = pytz.timezone('Europe/Kiev')
    start_date = kiev_tz.localize(datetime(2020, 7, 19))
    end_date = kiev_tz.localize(datetime(2025, 7, 18, 23, 59, 59))
    
    initial_account_balance = 5000.0
    risk_per_trade = 1.0
    target_rr = 2.6

    for config in STRATEGIES:
        print("-" * 50)
        logging.info(f"===== Початок бектесту для: {config['name']} =====")
        
        try:
            df_master = load_and_prepare_data(config)
            df_master = df_master.loc[start_date:end_date]
            if df_master.empty:
                logging.warning("DataFrame порожній після фільтрації за датою.")
                continue

            potential_setups = find_potential_setups(df_master, config)
            
            trade_results_df, equity_values = run_simulation(
                df_master, potential_setups, config, 
                initial_account_balance, risk_per_trade, target_rr
            )
            
            if not trade_results_df.empty:
                results_dir = 'results'
                os.makedirs(results_dir, exist_ok=True)
                file_path = os.path.join(results_dir, f"trades_{config['name']}.csv")
                
                trade_results_df.sort_values(by='exit_time', ascending=False, inplace=True)
                trade_results_df.to_csv(file_path, index=False, date_format='%Y-%m-%d %H:%M:%S')
                print(f"Результати збережено у файл: {file_path}")

                # ЗМІНА: Передаємо initial_account_balance у функцію
                max_dd_absolute, max_dd_percent = calculate_drawdown(equity_values, initial_account_balance)
                total_trades = len(trade_results_df)
                wins = len(trade_results_df[trade_results_df['outcome'] == 'win'])
                losses = len(trade_results_df[trade_results_df['outcome'] == 'loss'])
                breakevens = len(trade_results_df[trade_results_df['outcome'] == 'breakeven'])
                total_pnl = trade_results_df['pnl'].sum()
                win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
                max_losing_streak = calculate_max_losing_streak(trade_results_df['outcome'])

                # Погодинна статистика
                trade_results_df['entry_hour'] = pd.to_datetime(trade_results_df['entry_time']).dt.hour
                hourly_stats = trade_results_df.groupby('entry_hour').agg(
                    total_trades=('entry_hour', 'size'),
                    wins=('outcome', lambda x: (x == 'win').sum()),
                    total_pnl=('pnl', 'sum')
                ).reset_index()
                hourly_stats['win_rate'] = (hourly_stats['wins'] / hourly_stats['total_trades'] * 100).fillna(0)

                # Статистика по днях тижня
                trade_results_df['entry_day'] = pd.to_datetime(trade_results_df['entry_time']).dt.day_name()
                day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                daily_stats = trade_results_df.groupby('entry_day').agg(
                    total_trades=('entry_day', 'size'),
                    wins=('outcome', lambda x: (x == 'win').sum()),
                    total_pnl=('pnl', 'sum')
                ).reindex(day_order).dropna().reset_index()
                daily_stats['win_rate'] = (daily_stats['wins'] / daily_stats['total_trades'] * 100).fillna(0)

                # --- ВИВІД РЕЗУЛЬТАТІВ ---
                print("\n         ЗАГАЛЬНІ РЕЗУЛЬТАТИ БЕКТЕСТУ")
                print("="*55)
                print(f"Стратегія: {config['name']}")
                print(f"Угоди: {total_trades} (W: {wins}, L: {losses}, BE: {breakevens})")
                print(f"WinRate: {win_rate:.2f}%")
                print(f"PnL: {total_pnl:.2f}")
                print(f"Максимальна просадка (від початкового балансу): ${max_dd_absolute:.2f} ({max_dd_percent:.2f}%)")
                print(f"Максимальний лузстрік: {max_losing_streak} угод")
                print("="*55)

                print("\nАналіз ефективності за годинами входу:")
                print(hourly_stats.to_string(index=False, float_format="%.2f"))

                print("\nАналіз ефективності за днями тижня:")
                print(daily_stats.to_string(index=False, float_format="%.2f"))

            else:
                print("\nБектест не виявив жодних угод для аналізу.")

        except Exception as e:
            logging.error(f"Помилка під час виконання бектесту для {config['name']}: {e}", exc_info=True)
            
    print("-" * 50)
    print("Всі бектести завершено.")
