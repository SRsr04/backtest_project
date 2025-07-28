import pandas as pd
import numpy as np
import pytz
import logging
import os
from datetime import datetime, timedelta
import tqdm # Для відображення прогресу в консолі

# --- Налаштування логування ---
LOG_FILE_NAME = 'btc/m15h1_h4.log'
# Перевірка та створення директорії для логів, якщо вона не існує
log_dir = os.path.dirname(LOG_FILE_NAME)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG) # Встановлюємо рівень DEBUG для запису всіх повідомлень

# Видаляємо існуючі обробники, щоб уникнути дублювання логів при повторному запуску
if root_logger.handlers:
    for handler in root_logger.handlers:
        root_logger.removeHandler(handler)

# Обробник для запису логів у файл
file_handler = logging.FileHandler(LOG_FILE_NAME, mode='w', encoding='utf-8')
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)
root_logger.addHandler(file_handler)

# Обробник для виводу логів у консоль (тільки WARNING і вище)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.WARNING) # У консоль виводимо лише попередження та помилки
console_formatter = logging.Formatter('%(levelname)s: %(message)s')
console_handler.setFormatter(console_formatter)
root_logger.addHandler(console_handler)

logging.info(f"Логування налаштовано. Детальні логи записуються у файл '{LOG_FILE_NAME}'. Прогрес виконання буде відображатися в терміналі.")

# --- 1. Допоміжні функції ---
def custom_date_parser(date_str):
    """Парсер дат, що обробляє різні формати та повертає NaT для некоректних."""
    if pd.isna(date_str):
        return pd.NaT
    try:
        # Спробувати формат з часовою зоною
        return pd.to_datetime(date_str, format='%Y-%m-%d %H:%M:%S%z', errors='coerce')
    except ValueError:
        # Якщо не вдалося, спробувати інші поширені формати або повертати NaT
        return pd.to_datetime(date_str, errors='coerce')

def is_trading_hour(current_time: pd.Timestamp) -> bool:
    if current_time.tz is None:
        logging.warning(f"Час {current_time} не має часової зони. Локалізуємо до 'Europe/Kiev'.")
        kiev_tz = pytz.timezone('Europe/Kiev')
        current_time = current_time.tz_localize(kiev_tz)

    if current_time.dayofweek >= 5: # 5 = Saturday, 6 = Sunday
        logging.debug(f"Час {current_time.strftime('%Y-%m-%d %H:%M')} не торговий: вихідний день.")
        return False
    if 8 <= current_time.hour < 23:
        return True
    else:
        logging.debug(f"Час {current_time.strftime('%Y-%m-%d %H:%M')} не торговий: поза робочими годинами.")
        return False

def check_trade_close(trade, current_candle_data):
    """
    Перевіряє, чи була закрита поточна угода (тейк-профіт, стоп-лосс).
    Повертає 'win', 'loss' або 'pending'.
    """
    trade_type = trade['type']
    stop_loss = trade['stop_loss']
    take_profit = trade['take_profit']
    entry_price = trade['entry_price']
    position_size = trade['position_size']

    high = current_candle_data['high']
    low = current_candle_data['low']
    close = current_candle_data['close']
    current_time = current_candle_data.name

    outcome = 'pending'
    exit_time = current_time
    exit_price = np.nan
    pnl = 0

    logging.debug(f"\n--- Перевірка свічки {current_time} для угоди {trade_type.upper()} ---")
    logging.debug(f"Candle: High={high:.4f}, Low={low:.4f}, Close={close:.4f}")
    logging.debug(f"Trade {trade['entry_time']}: Entry={entry_price:.4f}, SL={stop_loss:.4f}, TP={take_profit:.4f}")


    if trade_type == 'long':
        # Перевірка на Тейк-Профіт
        if high >= take_profit:
            outcome = 'win'
            exit_price = take_profit
            logging.debug(f"Лонг: ✅ Тейк-профіт спрацював на {current_time} за ціною {take_profit:.4f}.")
        # Перевірка на Стоп-Лосс (якщо не Тейк-Профіт)
        elif low <= stop_loss:
            outcome = 'loss'
            exit_price = stop_loss
            logging.debug(f"Лонг: ❌ Стоп-лосс спрацював на {current_time} за ціною {stop_loss:.4f}.")

    elif trade_type == 'short':
        # Перевірка на Тейк-Профіт
        if low <= take_profit:
            outcome = 'win'
            exit_price = take_profit
            logging.debug(f"Шорт: ✅ Тейк-профіт спрацював на {current_time} за ціною {take_profit:.4f}.")
        # Перевірка на Стоп-Лосс (якщо не Тейк-Профіт)
        elif high >= stop_loss:
            outcome = 'loss'
            exit_price = stop_loss
            logging.debug(f"Шорт: ❌ Стоп-лосс спрацював на {current_time} за ціною {stop_loss:.4f}.")

    # Розрахунок PnL, якщо угода закрита
    if outcome != 'pending':
        if trade_type == 'long':
            pnl = (exit_price - entry_price) * position_size
        elif trade_type == 'short':
            pnl = (entry_price - exit_price) * position_size
        logging.debug(f"Фактично розрахований PnL: {pnl:.2f}")
    else:
        logging.debug("Угода все ще активна.")

    return outcome, exit_time, exit_price, pnl

def calculate_drawdown(equity_values):
    """
    Розраховує максимальну просадку (Max Drawdown) з кривої капіталу.
    :param equity_values: Список або Series значень балансу рахунку.
    :return: Кортеж (максимальна_просадка_абсолютна, максимальна_просадка_відсоткова)
    """
    if not equity_values:
        return 0, 0

    # Отримуємо початкове значення балансу. Важливо, щоб це було перше значення в серії.
    # Перевіряємо, чи це pandas Series, чи звичайний список.
    initial_equity_val = equity_values.iloc[0] if isinstance(equity_values, pd.Series) else equity_values[0]
    
    peak_equity = initial_equity_val
    max_drawdown = 0.0
    max_drawdown_percent = 0.0

    for current_equity in equity_values:
        # Оновлюємо найвищий пік, якщо поточний баланс вищий
        if current_equity > peak_equity:
            peak_equity = current_equity

        # Розраховуємо поточну просадку від найвищого піку
        current_drawdown = peak_equity - current_equity
        
        # Запобігаємо діленню на нуль, якщо peak_equity = 0, хоча для балансу це малоймовірно
        current_drawdown_percent = (current_drawdown / peak_equity) * 100 if peak_equity != 0 else 0

        # Оновлюємо максимальну просадку (якщо поточна більша)
        if current_drawdown > max_drawdown:
            max_drawdown = current_drawdown

        if current_drawdown_percent > max_drawdown_percent:
            max_drawdown_percent = current_drawdown_percent

    return max_drawdown, max_drawdown_percent


def load_and_prepare_data(candles_path, fvg_path, bos_path, h1_candles_path, h4_candles_path):
    """
    Завантажує, обробляє та ОБ'ЄДНУЄ дані для стратегії M15/H1 з H4 фільтром.
    
    Returns:
        pd.DataFrame: Єдиний DataFrame, що містить всю необхідну інформацію для бектесту.
    """
    kiev_tz = pytz.timezone('Europe/Kiev')

    def handle_timezone(dt_series):
        """Допоміжна функція для коректної обробки часових зон."""
        if dt_series.dt.tz is None:
            return dt_series.dt.tz_localize(kiev_tz, ambiguous='NaT', nonexistent='NaT')
        else:
            return dt_series.dt.tz_convert(kiev_tz)

    def load_csv_to_datetime_index(file_path, date_col_name='datetime'):
        df = pd.read_csv(file_path, encoding='utf-8')
        dt_series = pd.to_datetime(df[date_col_name].apply(custom_date_parser), errors='coerce')
        df[date_col_name] = handle_timezone(dt_series)
        df.dropna(subset=[date_col_name], inplace=True)
        df = df.set_index(date_col_name).sort_index()
        return df[~df.index.duplicated(keep='first')]

    # --- КРОК 1: Завантаження всіх даних ---
    logging.info("Завантаження даних...")
    df_m15_candles = load_csv_to_datetime_index(candles_path)
    df_h1_candles = load_csv_to_datetime_index(h1_candles_path)
    df_h4_candles = load_csv_to_datetime_index(h4_candles_path)
    
    df_fvg = load_csv_to_datetime_index(fvg_path, 'time')
    df_fvg.rename(columns={'type': 'fvg_type', 'min': 'fvg_min', 'max': 'fvg_max', 'middle': 'fvg_middle'}, inplace=True)

    df_bos = load_csv_to_datetime_index(bos_path, 'bos_time_kiev')
    df_bos.rename(columns={'type': 'bos_type', 'level': 'bos_level'}, inplace=True)
    
    # --- КРОК 2: Підготовка даних перед об'єднанням ---
    logging.info("Підготовка даних перед об'єднанням...")
    df_h4_candles['h4_direction'] = np.select(
        [df_h4_candles['close'] > df_h4_candles['open'], df_h4_candles['close'] < df_h4_candles['open']],
        ['bullish', 'bearish'],
        default='neutral'
    )
    
    # --- КРОК 3: ОБ'ЄДНАННЯ ДАНИХ В ЄДИНИЙ "МАЙСТЕР-DATAFRAME" ---
    logging.info("Починаємо об'єднання даних...")
    df_h1_candles = df_h1_candles.add_suffix('_h1')
    df_h4_candles = df_h4_candles.add_suffix('_h4')

    df_master = pd.merge_asof(df_m15_candles, df_h1_candles, left_index=True, right_index=True, direction='backward')
    df_master = pd.merge_asof(df_master, df_h4_candles, left_index=True, right_index=True, direction='backward')
    
    df_master = df_master.join(df_fvg[['fvg_type', 'fvg_min', 'fvg_max', 'fvg_middle']], how='left')
    df_master = df_master.join(df_bos[['bos_type', 'bos_level']], how='left')

    # --- КРОК 4: Розрахунок індикаторів та фінальна очистка ---
    logging.info("Розраховуємо фрактали та проводимо фінальну очистку...")
    df_master['is_high_fractal'] = (df_master['high'].shift(1) < df_master['high']) & (df_master['high'] > df_master['high'].shift(-1))
    df_master['is_low_fractal'] = (df_master['low'].shift(1) > df_master['low']) & (df_master['low'] < df_master['low'].shift(-1))

    for col in ['fvg_type', 'bos_type', 'h4_direction_h4']: df_master[col].fillna('', inplace=True)
    for col in ['fvg_min', 'fvg_max', 'fvg_middle', 'bos_level']: df_master[col].fillna(0, inplace=True)
    
    logging.info(f"Підготовка даних завершена. Розмір фінального DataFrame: {len(df_master)}.")
    return df_master

def run_optimized_backtest_m15h1_h4_no_be(df_master, start_date, end_date, initial_balance, risk_per_trade, target_rr, params):
    """
    Виконує повний ОПТИМІЗОВАНИЙ бектест для стратегії M15/H1 з H4 фільтром,
    АЛЕ БЕЗ логіки переміщення в беззбиток.
    Повертає DataFrame з результатами угод та список значень equity curve.
    """
    logging.info("Починаємо оптимізований бектест M15/H1 (з H4 фільтром, без БУ)...")
    print("Запускаємо оптимізований бектест M15/H1 (з H4 фільтром, без БУ)...")

    df_backtest = df_master.loc[start_date:end_date].copy()
    if df_backtest.empty:
        logging.warning("DataFrame для бектесту порожній після фільтрації за датами.")
        return pd.DataFrame(), [initial_balance] # Повертаємо пустий DataFrame і equity з початковим балансом

    fvg_events = df_backtest[df_backtest['fvg_type'] != ''].copy()
    potential_setups, used_bos_times = [], set()
    SEARCH_WINDOW = timedelta(hours=48)

    print(f"Етап 1/2: Знайдено {len(fvg_events)} H1 FVG. Аналізуємо з урахуванням H4 фільтру...")
    for fvg_time, fvg_row in tqdm.tqdm(fvg_events.iterrows(), desc="Пошук сетапів (M15/H1/H4)"):
        
        if not is_trading_hour(fvg_time): continue

        # --- ЗАСТОСУВАННЯ H4 ФІЛЬТРА ---
        h4_direction = fvg_row.get('h4_direction_h4') # Використовуємо .get() на випадок відсутності ключа
        if h4_direction is None or h4_direction == '': # Додано перевірку на порожній рядок
            continue # Пропускаємо, якщо напрямок H4 не визначено
        
        if (fvg_row['fvg_type'] == 'bullish' and h4_direction != 'bullish') or \
           (fvg_row['fvg_type'] == 'bearish' and h4_direction != 'bearish'):
            continue
        # --- КІНЕЦЬ H4 ФІЛЬТРА ---

        df_search = df_backtest.loc[fvg_time + timedelta(minutes=15) : fvg_time + SEARCH_WINDOW]
        if df_search.empty: continue
        
        # Перевірка на "заповнення" FVG H1 свічками
        if (fvg_row['fvg_type'] == 'bullish' and (df_search['close_h1'] < fvg_row['fvg_min']).any()) or \
           (fvg_row['fvg_type'] == 'bearish' and (df_search['close_h1'] > fvg_row['fvg_max']).any()):
            continue
        
        if fvg_row['fvg_type'] == 'bullish':
            f1_cand = df_search[(df_search['is_high_fractal']) & (df_search['high'] > fvg_row['fvg_max'])]
            for _, f1 in f1_cand.iterrows():
                f2_cand = df_search[(df_search.index > f1.name) & (df_search['is_low_fractal']) & (df_search['low'] <= fvg_row['fvg_max'])]
                if f2_cand.empty: continue
                f2 = f2_cand.loc[f2_cand['low'].idxmin()]
                if f1['high'] <= f2['low']: continue
                bos_cand = df_search[(df_search.index > f2.name) & (df_search['bos_type'] == 'bullish') & (df_search['close'] > f1['high']) & (~df_search.index.isin(used_bos_times))]
                if bos_cand.empty: continue
                bos = bos_cand.iloc[0]
                fib = f2['low'] + (f1['high'] - f2['low']) * 0.62
                entry_cand = df_search[(df_search.index > bos.name) & (df_search['low'] <= fib)]
                if entry_cand.empty or (entry_cand.index[0] - bos.name > timedelta(hours=4)) or not is_trading_hour(entry_cand.index[0]): continue
                potential_setups.append({'entry_time': entry_cand.index[0], 'entry_price': fib, 'stop_loss': f2['low'], 'type': 'long', 'fvg_row': fvg_row, 'f1_row': f1, 'f2_row': f2, 'bos_row': bos})
                used_bos_times.add(bos.name)
                break
        else: # Bearish
            f1_cand = df_search[(df_search['is_low_fractal']) & (df_search['low'] < fvg_row['fvg_min'])]
            for _, f1 in f1_cand.iterrows():
                f2_cand = df_search[(df_search.index > f1.name) & (df_search['is_high_fractal']) & (df_search['high'] >= fvg_row['fvg_min'])]
                if f2_cand.empty: continue
                f2 = f2_cand.loc[f2_cand['high'].idxmax()]
                if f1['low'] >= f2['high']: continue
                bos_cand = df_search[(df_search.index > f2.name) & (df_search['bos_type'] == 'bearish') & (df_search['close'] < f1['low']) & (~df_search.index.isin(used_bos_times))]
                if bos_cand.empty: continue
                bos = bos_cand.iloc[0]
                fib = f2['high'] - (f2['high'] - f1['low']) * 0.62
                entry_cand = df_search[(df_search.index > bos.name) & (df_search['high'] >= fib)]
                if entry_cand.empty or (entry_cand.index[0] - bos.name > timedelta(hours=4)) or not is_trading_hour(entry_cand.index[0]): continue
                potential_setups.append({'entry_time': entry_cand.index[0], 'entry_price': fib, 'stop_loss': f2['high'], 'type': 'short', 'fvg_row': fvg_row, 'f1_row': f1, 'f2_row': f2, 'bos_row': bos})
                used_bos_times.add(bos.name)
                break

    if not potential_setups:
        logging.info("Не знайдено потенційних сетапів після фільтрації.")
        return pd.DataFrame(), [initial_balance]
    potential_setups.sort(key=lambda x: x['entry_time'])
    
    all_trades, current_balance, active_trade = [], initial_balance, None
    equity_curve_values = [initial_balance] # Ініціалізуємо криву капіталу початковим балансом

    print(f"\nЕтап 2/2: Знайдено {len(potential_setups)} сетапів. Симулюємо угоди...")
    for setup in tqdm.tqdm(potential_setups, desc="Симуляція угод (M15/H1/H4, без БУ)"):
        if active_trade is None and (not all_trades or setup['entry_time'] >= all_trades[-1]['exit_time']):
            risk_abs = abs(setup['entry_price'] - setup['stop_loss'])
            if risk_abs == 0: continue
            
            # Важливо: Розмір позиції повинен розраховуватися від *поточного* балансу
            pos_size = (current_balance * risk_per_trade / 100) / risk_abs
            
            tp = setup['entry_price'] + (risk_abs * target_rr) if setup['type'] == 'long' else setup['entry_price'] - (risk_abs * target_rr)
            
            active_trade = {'type': setup['type'], 'entry_time': setup['entry_time'], 'entry_price': setup['entry_price'], 'stop_loss': setup['stop_loss'], 'take_profit': tp, 'position_size': pos_size, 'initial_stop_loss': setup['stop_loss'], 'fvg_type': setup['fvg_row']['fvg_type'], 'fvg_start_time': setup['fvg_row'].name, 'fvg_range': f"{setup['fvg_row']['fvg_min']:.1f}-{setup['fvg_row']['fvg_max']:.1f}", 'fractal_1_price': setup['f1_row']['high'] if setup['type'] == 'long' else setup['f1_row']['low'], 'fractal_2_price': setup['f2_row']['low'] if setup['type'] == 'long' else setup['f2_row']['high']}

            for _, candle in df_backtest[df_backtest.index > active_trade['entry_time']].iterrows():
                outcome, exit_time, exit_price, pnl = check_trade_close(active_trade, candle)
                if outcome != 'pending':
                    active_trade.update({'outcome': outcome, 'exit_time': exit_time, 'exit_price': exit_price, 'pnl': pnl})
                    current_balance += pnl
                    equity_curve_values.append(current_balance) # <-- ЗБЕРІГАЄМО ЗНАЧЕННЯ БАЛАНСУ
                    all_trades.append(active_trade)
                    active_trade = None
                    break
        elif active_trade is not None:
             pass 
        else:
            pass

    if not all_trades:
        logging.info("Немає закритих угод для аналізу.")
        return pd.DataFrame(), equity_curve_values
    
    results_df = pd.DataFrame(all_trades)
    desired_order = ['type', 'fvg_type', 'fvg_start_time', 'fvg_range', 'fractal_1_price', 'fractal_2_price', 'entry_time', 'entry_price', 'initial_stop_loss', 'stop_loss', 'take_profit', 'outcome', 'pnl', 'exit_time', 'exit_price']
    results_df = results_df.reindex(columns=[col for col in desired_order if col in results_df.columns])
    
    results_dir, file_path = 'backtest_results', os.path.join('backtest_results', "m15h1h4.csv")
    os.makedirs(results_dir, exist_ok=True)
    results_df.to_csv(file_path, index=False, date_format='%Y-%m-%d %H:%M:%S')
    print(f"\nБектест (без БУ) завершено. Знайдено {len(all_trades)} угод. Результати в '{file_path}'")
    
    return results_df, equity_curve_values # Повертаємо і DataFrame, і список equity


# --- ГОЛОВНИЙ БЛОК ВИКОНАННЯ (МОДИФІКОВАНИЙ) ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    CANDLES_FILE = 'm15_candels.csv'
    FVG_FILE = 'fvg_h1.csv'
    BOS_FILE = 'bos_m15.csv'
    H1_CANDLES_FILE = 'h1_candels.csv'
    H4_CANDLES_FILE = 'h4_candels.csv'

    try:
        df_master = load_and_prepare_data(
            CANDLES_FILE, 
            FVG_FILE, 
            BOS_FILE, 
            H1_CANDLES_FILE,
            H4_CANDLES_FILE
        )
        if df_master.empty: raise ValueError("DataFrame порожній. Перевірте файли даних або функцію load_and_prepare_data.")

        kiev_tz = pytz.timezone('Europe/Kiev')
        start_date = kiev_tz.localize(datetime(2020, 7, 19))
        end_date = kiev_tz.localize(datetime(2025, 7, 18, 23, 59, 59))
        
        initial_account_balance = 10000.0
        
        trade_results_df, equity_values = run_optimized_backtest_m15h1_h4_no_be(
            df_master, 
            start_date, 
            end_date, 
            initial_account_balance, 
            1.0, 
            2.6, 
            {}
        )

        if not trade_results_df.empty:
            max_dd_absolute, max_dd_percent = calculate_drawdown(equity_values)

            total_trades = len(trade_results_df)
            wins = len(trade_results_df[trade_results_df['outcome'] == 'win'])
            losses = len(trade_results_df[trade_results_df['outcome'] == 'loss'])
            
            total_pnl = trade_results_df['pnl'].sum()
            win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
            
            print("\n         BTC (M15/H1 + H4 Filter)")
            print("="*60)
            print(f"Угоди: {total_trades} (W: {wins}, L: {losses})")
            print(f"WinRate: {win_rate:.2f}%")
            print(f"PnL: {total_pnl:.2f}")
            print(f"Максимальна просадка: ${max_dd_absolute:.2f} ({max_dd_percent:.2f}%)")
            print("="*60)

            trade_results_df['entry_hour'] = pd.to_datetime(trade_results_df['entry_time']).dt.hour
            hourly_stats = trade_results_df.groupby('entry_hour').agg(total_trades=('entry_hour', 'size'), wins=('outcome', lambda x: (x == 'win').sum()), total_pnl=('pnl', 'sum')).reset_index()
            hourly_stats['average_pnl_per_trade'] = hourly_stats['total_pnl'] / hourly_stats['total_trades']
            hourly_stats['win_rate'] = (hourly_stats['wins'] / hourly_stats['total_trades'] * 100).fillna(0)
            print("\nАналіз ефективності за годинами входу:")
            print(hourly_stats.to_string(index=False, float_format="%.2f"))
        else:
            print("\nБектест не виявив жодних угод для аналізу.")
            if equity_values:
                max_dd_absolute, max_dd_percent = calculate_drawdown(equity_values)
                print(f"Максимальна просадка (без угод): ${max_dd_absolute:.2f} ({max_dd_percent:.2f}%)")

    except FileNotFoundError as e:
        logging.error(f"Помилка: Файл не знайдено - {e}.")
    except Exception as e:
        logging.error(f"Виникла непередбачена помилка: {e}", exc_info=True)
