import pandas as pd
import numpy as np
import pytz
import logging
import os
from datetime import datetime, timedelta
import tqdm # Для відображення прогресу в консолі

# --- Налаштування логування ---
LOG_FILE_NAME = 'btc/m15h1_be.log'
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
    """
    Перевіряє, чи поточний час знаходиться в межах торгових годин (Пн-Пт, 08:00-22:59 Київського часу).
    """
    # Переконаємось, що час має часову зону
    if current_time.tz is None:
        logging.warning(f"Час {current_time} не має часової зони в is_trading_hour. Локалізуємо до 'Europe/Kiev'.")
        kiev_tz = pytz.timezone('Europe/Kiev')
        current_time = current_time.tz_localize(kiev_tz)
    elif current_time.tz != pytz.timezone('Europe/Kiev'):
        current_time = current_time.tz_convert(pytz.timezone('Europe/Kiev'))
    
    # Перевірка на вихідні дні
    if current_time.dayofweek >= 5:  # 5 = Saturday, 6 = Sunday
        logging.debug(f"Час {current_time.strftime('%Y-%m-%d %H:%M')} не торговий: вихідний день.")
        return False
        
    # Перевірка на торгові години (з 08:00 до 22:59 включно)
    if 8 <= current_time.hour < 23:
        return True
    else:
        logging.debug(f"Час {current_time.strftime('%Y-%m-%d %H:%M')} не торговий: поза робочими годинами.")
        return False

def breakeven_info(entry_price: float, volume: float, side: str, fee_percent: float = 0.0325):
    """
    Calculates potential commission and breakeven price (for stop).

    :param entry_price: Entry price
    :param volume: Trade volume
    :param side: "long" or "short"
    :param fee_percent: Commission per side in %
    """
    notional = entry_price * volume
    # Commission is usually taken from the notional for each side of the trade
    # If fee_percent is already the total commission for a round trip (entry + exit),
    # then we use it directly without multiplying by 2.
    total_fee = notional * (fee_percent / 100) # Removed * 2 as per user's clarification

    breakeven_price = 0.0 # Initialize
    if side.lower() == "long":
        breakeven_price = (notional + total_fee) / volume
    elif side.lower() == "short":
        breakeven_price = (notional - total_fee) / volume
    else:
        logging.error("❌ Trade side must be 'long' or 'short'")
        return 0.0, 0.0 # Return 0.0 for commission and price if side is incorrect

    return total_fee, breakeven_price # Return both commission and breakeven price

def check_trade_close(trade, current_candle_data):
    """
    Checks if the current trade was closed (take-profit, stop-loss, or breakeven),
    with logic for first TP/SL/BE hit from candle open price.
    Returns 'win', 'loss', 'breakeven' or 'pending'.
    Now accounts for commission in PnL calculation.
    """
    trade_type = trade['type']
    # Use the current 'stop_loss', which can be moved to BE
    current_active_stop_loss = trade['stop_loss']
    take_profit = trade['take_profit']
    entry_price = trade['entry_price']
    position_size = trade['position_size']
    total_commission_for_trade = trade['commission_paid_total']
    breakeven_price_with_fee = trade['breakeven_price_with_fee']

    open_price = current_candle_data['open']
    high = current_candle_data['high']
    low = current_candle_data['low']
    close = current_candle_data['close']
    current_time = current_candle_data.name

    outcome = 'pending'
    exit_time = current_time
    exit_price = np.nan
    pnl = 0

    logging.debug(f"\n--- Checking candle {current_time} for trade {trade_type.upper()} ---")
    logging.debug(f"Candle: Open={open_price:.4f}, High={high:.4f}, Low={low:.4f}, Close={close:.4f}")
    logging.debug(f"Trade: Entry={entry_price:.4f}, SL={current_active_stop_loss:.4f}, TP={take_profit:.4f}, Breakeven(with fee)={breakeven_price_with_fee:.4f}, Commission={total_commission_for_trade:.4f}")

    EPSILON = 1e-9

    tp_hit = False
    sl_hit = False

    if trade_type == 'long':
        if high >= take_profit:
            tp_hit = True
        if low <= current_active_stop_loss: # Use current_active_stop_loss
            sl_hit = True
    elif trade_type == 'short':
        if low <= take_profit:
            tp_hit = True
        if high >= current_active_stop_loss: # Use current_active_stop_loss
            sl_hit = True

    if tp_hit and sl_hit:
        # Logic to determine what came first if both levels are hit
        if trade_type == 'long':
            # Long: if open already beyond TP/SL, take them, otherwise which is closer to Open
            if open_price >= take_profit:
                outcome = 'win'
                exit_price = take_profit
                logging.debug(f"Long: ✅ Take-profit hit on {current_time} at price {take_profit:.4f} (Open above TP).")
            elif open_price <= current_active_stop_loss:
                exit_price = current_active_stop_loss
                if abs(current_active_stop_loss - breakeven_price_with_fee) < EPSILON:
                    outcome = 'breakeven'
                    logging.debug(f"Long: 🅿️ Breakeven hit on {current_time} at price {exit_price:.4f} (Open below BE/SL).")
                else:
                    outcome = 'loss'
                    logging.debug(f"Long: ❌ Stop-loss hit on {current_time} at price {exit_price:.4f} (Open below SL).")
            elif take_profit - open_price <= open_price - current_active_stop_loss:
                outcome = 'win'
                exit_price = take_profit
                logging.debug(f"Long: ✅ Take-profit hit on {current_time} at price {take_profit:.4f} (TP closer/equidistant to Open).")
            else:
                exit_price = current_active_stop_loss
                if abs(current_active_stop_loss - breakeven_price_with_fee) < EPSILON:
                    outcome = 'breakeven'
                    logging.debug(f"Long: 🅿️ Breakeven hit on {current_time} at price {exit_price:.4f} (BE closer to Open).")
                else:
                    outcome = 'loss'
                    logging.debug(f"Long: ❌ Stop-loss hit on {current_time} at price {exit_price:.4f} (SL closer to Open).")

        elif trade_type == 'short':
            # Short: analogous logic, but mirrored
            if open_price <= take_profit:
                outcome = 'win'
                exit_price = take_profit
                logging.debug(f"Short: ✅ Take-profit hit on {current_time} at price {take_profit:.4f} (Open below TP).")
            elif open_price >= current_active_stop_loss:
                exit_price = current_active_stop_loss
                if abs(current_active_stop_loss - breakeven_price_with_fee) < EPSILON:
                    outcome = 'breakeven'
                    logging.debug(f"Short: 🅿️ Breakeven hit on {current_time} at price {exit_price:.4f} (Open above BE/SL).")
                else:
                    outcome = 'loss'
                    logging.debug(f"Short: ❌ Stop-loss hit on {current_time} at price {exit_price:.4f} (Open above SL).")
            elif open_price - take_profit <= current_active_stop_loss - open_price:
                outcome = 'win'
                exit_price = take_profit
                logging.debug(f"Short: ✅ Take-profit hit on {current_time} at price {take_profit:.4f} (TP closer/equidistant to Open).")
            else:
                exit_price = current_active_stop_loss
                if abs(current_active_stop_loss - breakeven_price_with_fee) < EPSILON:
                    outcome = 'breakeven'
                    logging.debug(f"Short: 🅿️ Breakeven hit on {current_time} at price {exit_price:.4f} (BE closer to Open).")
                else:
                    outcome = 'loss'
                    logging.debug(f"Short: ❌ Stop-loss hit on {current_time} at price {exit_price:.4f} (SL closer to Open).")

    elif tp_hit:
        outcome = 'win'
        exit_price = take_profit
        logging.debug(f"{trade_type.upper()}: ✅ Take-profit hit on {current_time} at price {take_profit:.4f}.")

    elif sl_hit:
        exit_price = current_active_stop_loss # Use current_active_stop_loss
        if abs(current_active_stop_loss - breakeven_price_with_fee) < EPSILON:
            outcome = 'breakeven'
            logging.debug(f"{trade_type.upper()}: 🅿️ Breakeven hit on {current_time} at price {exit_price:.4f} (with commission).")
        else:
            outcome = 'loss'
            logging.debug(f"{trade_type.upper()}: ❌ Stop-loss hit on {current_time} at price {exit_price:.4f}.")

    if outcome != 'pending':
        if trade_type == 'long':
            pnl = (exit_price - entry_price) * position_size
        elif trade_type == 'short':
            pnl = (entry_price - exit_price) * position_size

        pnl -= total_commission_for_trade
        logging.debug(f"Actual calculated PnL (with commission): {pnl:.4f} (Outcome: {outcome})")
    else:
        logging.debug("Trade is still active.")

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


def load_and_prepare_data(candles_path, fvg_path, bos_path, h1_candles_path):
    """
    Завантажує, обробляє та ОБ'ЄДНУЄ дані для стратегії M15/H1 (з логікою БУ).
    
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
    df_m15_candles = load_csv_to_datetime_index(candles_path) # Основний ТФ - M15
    df_h1_candles = load_csv_to_datetime_index(h1_candles_path)
    
    df_fvg = load_csv_to_datetime_index(fvg_path, 'time') # FVG з H1
    df_fvg.rename(columns={'type': 'fvg_type', 'min': 'fvg_min', 'max': 'fvg_max', 'middle': 'fvg_middle'}, inplace=True)

    df_bos = load_csv_to_datetime_index(bos_path, 'bos_time_kiev') # BOS з M15
    df_bos.rename(columns={'type': 'bos_type', 'level': 'bos_level'}, inplace=True)
    df_bos['fract_time_kiev'] = handle_timezone(pd.to_datetime(df_bos['fract_time_kiev'].apply(custom_date_parser), errors='coerce'))
    
    # --- КРОК 2: Підготовка даних перед об'єднанням ---
    logging.info("Підготовка даних перед об'єднанням...")
    # Готуємо ціни фракталів для BOS з M15 свічок
    fractal_high_prices = df_m15_candles['high']
    fractal_low_prices = df_m15_candles['low']
    
    bullish_fractal_prices = df_bos['fract_time_kiev'].map(fractal_high_prices)
    bearish_fractal_prices = df_bos['fract_time_kiev'].map(fractal_low_prices)
    df_bos['bos_fractal_price'] = np.where(df_bos['bos_type'] == 'bullish', bullish_fractal_prices, bearish_fractal_prices)

    # --- КРОК 3: ОБ'ЄДНАННЯ ДАНИХ В ЄДИНИЙ "МАЙСТЕР-DATAFRAME" ---
    logging.info("Починаємо об'єднання даних...")
    df_h1_candles = df_h1_candles.add_suffix('_h1')

    # Збагачуємо M15 дані даними з H1
    df_master = pd.merge_asof(df_m15_candles, df_h1_candles, left_index=True, right_index=True, direction='backward')
    
    # Додаємо FVG та BOS
    df_master = df_master.join(df_fvg[['fvg_type', 'fvg_min', 'fvg_max', 'fvg_middle']], how='left')
    df_master = df_master.join(df_bos[['bos_type', 'bos_level', 'bos_fractal_price']], how='left')

    # --- КРОК 4: Розрахунок індикаторів та фінальна очистка ---
    logging.info("Розраховуємо фрактали та проводимо фінальну очистку...")
    df_master['is_high_fractal'] = (df_master['high'].shift(1) < df_master['high']) & (df_master['high'] > df_master['high'].shift(-1))
    df_master['is_low_fractal'] = (df_master['low'].shift(1) > df_master['low']) & (df_master['low'] < df_master['low'].shift(-1))

    for col in ['fvg_type', 'bos_type']: df_master[col].fillna('', inplace=True)
    for col in ['fvg_min', 'fvg_max', 'fvg_middle', 'bos_level', 'bos_fractal_price']: df_master[col].fillna(0, inplace=True)
    
    logging.info(f"Підготовка даних завершена. Розмір фінального DataFrame: {len(df_master)}.")
    return df_master


def run_optimized_backtest_m15h1_h4_filter(df_master, start_date, end_date, initial_balance, risk_per_trade, target_rr, params):
    """
    Виконує повний ОПТИМІЗОВАНИЙ бектест для стратегії M15/H1 з H4 фільтром та логікою БУ.
    Повертає DataFrame з результатами угод та список значень equity curve.
    """
    logging.info("Починаємо оптимізований бектест M15/H1 (з H4 фільтром та БУ)...")
    print("Запускаємо оптимізований бектест M15/H1 (з H4 фільтром та БУ)...")

    df_backtest = df_master.loc[start_date:end_date].copy()
    if df_backtest.empty:
        logging.warning("DataFrame для бектесту порожній після фільтрації за датами.")
        return pd.DataFrame(), [initial_balance]

    fvg_events = df_backtest[df_backtest['fvg_type'] != ''].copy()
    potential_setups, used_bos_times = [], set()
    SEARCH_WINDOW = timedelta(hours=48)

    print(f"Етап 1/2: Знайдено {len(fvg_events)} H1 FVG. Аналізуємо з урахуванням H4 фільтру...")
    for fvg_time, fvg_row in tqdm.tqdm(fvg_events.iterrows(), desc="Пошук сетапів (M15/H1/H4)"):
        
        if not is_trading_hour(fvg_time): continue

        h4_direction = fvg_row.get('h4_direction_h4')
        if h4_direction is None or h4_direction == '':
            continue
        
        if (fvg_row['fvg_type'] == 'bullish' and h4_direction != 'bullish') or \
           (fvg_row['fvg_type'] == 'bearish' and h4_direction != 'bearish'):
            continue
        
        df_search = df_backtest.loc[fvg_time + timedelta(minutes=15) : fvg_time + SEARCH_WINDOW]
        if df_search.empty: continue
        
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
    # --- ЗМІНА: Ініціалізуємо криву капіталу ---
    equity_curve_values = [initial_balance]

    print(f"\nЕтап 2/2: Знайдено {len(potential_setups)} сетапів. Симулюємо угоди...")
    for setup in tqdm.tqdm(potential_setups, desc="Симуляція угод (M15/H1/H4 з БУ)"):
        if active_trade is None and (not all_trades or setup['entry_time'] >= all_trades[-1]['exit_time']):
            risk_abs = abs(setup['entry_price'] - setup['stop_loss'])
            if risk_abs == 0: continue
            pos_size = (current_balance * risk_per_trade / 100) / risk_abs
            tp = setup['entry_price'] + (risk_abs * target_rr) if setup['type'] == 'long' else setup['entry_price'] - (risk_abs * target_rr)
            
            comm, be_price = breakeven_info(setup['entry_price'], pos_size, setup['type'])
            
            active_trade = {'type': setup['type'], 'entry_time': setup['entry_time'], 'entry_price': setup['entry_price'], 'stop_loss': setup['stop_loss'], 'take_profit': tp, 'position_size': pos_size, 'initial_stop_loss': setup['stop_loss'], 'commission_paid_total': comm, 'breakeven_price_with_fee': be_price, 'breakeven_activated': False, 'fvg_type': setup['fvg_row']['fvg_type'], 'fvg_start_time': setup['fvg_row'].name, 'fvg_range': f"{setup['fvg_row']['fvg_min']:.1f}-{setup['fvg_row']['fvg_max']:.1f}", 'fractal_1_price': setup['f1_row']['high'] if setup['type'] == 'long' else setup['f1_row']['low'], 'fractal_2_price': setup['f2_row']['low'] if setup['type'] == 'long' else setup['f2_row']['high'], 'next_bos_after_entry_time': pd.NaT, 'next_bos_after_entry_type': None}

            for _, candle in df_backtest[df_backtest.index > active_trade['entry_time']].iterrows():
                outcome, exit_time, exit_price, pnl = check_trade_close(active_trade, candle)
                if outcome != 'pending':
                    active_trade.update({'outcome': outcome, 'exit_time': exit_time, 'exit_price': exit_price, 'pnl': pnl})
                    current_balance += pnl
                    # --- ЗМІНА: Зберігаємо значення балансу ---
                    equity_curve_values.append(current_balance)
                    all_trades.append(active_trade)
                    active_trade = None
                    break
                
                if not active_trade['breakeven_activated'] and candle['bos_type'] != '':
                    is_valid_bos_type = (active_trade['type'] == 'long' and candle['bos_type'] == 'bullish') or \
                                         (active_trade['type'] == 'short' and candle['bos_type'] == 'bearish')
                    
                    bos_fractal_price = candle.get('bos_fractal_price', None)
                    if bos_fractal_price is None or pd.isna(bos_fractal_price):
                        continue

                    if is_valid_bos_type:
                        entry_price, bos_close_price = active_trade['entry_price'], candle['bos_level']
                        
                        move_to_be = False
                        if active_trade['type'] == 'long':
                            move_to_be = (bos_close_price > entry_price and bos_fractal_price > entry_price)
                        elif active_trade['type'] == 'short':
                            move_to_be = (bos_close_price < entry_price and bos_fractal_price < entry_price)
                        
                        if move_to_be:
                            active_trade.update({
                                'stop_loss': active_trade['breakeven_price_with_fee'],
                                'breakeven_activated': True,
                                'next_bos_after_entry_time': candle.name,
                                'next_bos_after_entry_type': candle['bos_type']
                            })

    if not all_trades:
        logging.info("Немає закритих угод для аналізу.")
        return pd.DataFrame(), equity_curve_values
    
    results_df = pd.DataFrame(all_trades)
    desired_order = ['type', 'fvg_type', 'fvg_start_time', 'fvg_range', 'fractal_1_price', 'fractal_2_price', 'entry_time', 'entry_price', 'initial_stop_loss', 'stop_loss', 'take_profit', 'outcome', 'pnl', 'exit_time', 'exit_price', 'commission_paid_total', 'breakeven_price_with_fee', 'breakeven_activated', 'next_bos_after_entry_time', 'next_bos_after_entry_type']
    results_df = results_df.reindex(columns=[col for col in desired_order if col in results_df.columns])
    
    results_dir, file_path = 'backtest_results', os.path.join('backtest_results', "m15h1h4_be.csv")
    os.makedirs(results_dir, exist_ok=True)
    results_df.to_csv(file_path, index=False, date_format='%Y-%m-%d %H:%M:%S')
    print(f"\nБектест завершено. Знайдено {len(all_trades)} угод. Результати в '{file_path}'")
    
    # --- ЗМІНА: Повертаємо і DataFrame, і список equity ---
    return results_df, equity_curve_values


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Шляхи до файлів для стратегії M15/H1/H4
    CANDLES_FILE = 'm15_candels.csv'
    FVG_FILE = 'fvg_h1.csv'
    BOS_FILE = 'bos_m15.csv'
    H1_CANDLES_FILE = 'h1_candels.csv'
    H4_CANDLES_FILE = 'h4_candels.csv'

    try:
        df_master = load_and_prepare_data( # Ця функція повинна бути визначена у вашому файлі
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
        
        initial_account_balance = 5000.0 # ЗМІНЕНО НА 5000.0
        
        # --- ЗМІНА: Тепер отримуємо два значення з функції бектесту ---
        trade_results_df, equity_values = run_optimized_backtest_m15h1_h4_filter(
            df_master, 
            start_date, 
            end_date, 
            initial_account_balance, 
            1.0, 
            2.6, 
            {}
        )

        if not trade_results_df.empty:
            # --- ЗМІНА: Розрахунок просадки ---
            max_dd_absolute, max_dd_percent = calculate_drawdown(equity_values)

            total_trades = len(trade_results_df)
            wins = len(trade_results_df[trade_results_df['outcome'] == 'win'])
            losses = len(trade_results_df[trade_results_df['outcome'] == 'loss'])
            breakevens = len(trade_results_df[trade_results_df['outcome'] == 'breakeven'])
            
            total_pnl = trade_results_df['pnl'].sum()
            win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
            
            print("\n         BTC (M15/H1 + H4 Filter + BE)")
            print("="*60)
            print(f"Угоди: {total_trades} (W: {wins}, L: {losses}, BE: {breakevens})")
            print(f"WinRate: {win_rate:.2f}%")
            print(f"PnL: {total_pnl:.2f}")
            print(f"Максимальна просадка: ${max_dd_absolute:.2f} ({max_dd_percent:.2f}%)") # <-- ДОДАНО ТУТ
            print("="*60)

            trade_results_df['entry_hour'] = pd.to_datetime(trade_results_df['entry_time']).dt.hour
            hourly_stats = trade_results_df.groupby('entry_hour').agg(total_trades=('entry_hour', 'size'), wins=('outcome', lambda x: (x == 'win').sum()), total_pnl=('pnl', 'sum')).reset_index()
            hourly_stats['average_pnl_per_trade'] = hourly_stats['total_pnl'] / hourly_stats['total_trades']
            hourly_stats['win_rate'] = (hourly_stats['wins'] / hourly_stats['total_trades'] * 100).fillna(0)
            print("\nАналіз ефективності за годинами входу:")
            print(hourly_stats.to_string(index=False, float_format="%.2f"))
        else:
            print("\nБектест не виявив жодних угод для аналізу.")
            # --- ЗМІНА: Виводимо просадку навіть без угод ---
            if equity_values:
                max_dd_absolute, max_dd_percent = calculate_drawdown(equity_values)
                print(f"Максимальна просадка (без угод): ${max_dd_absolute:.2f} ({max_dd_percent:.2f}%)")

    except FileNotFoundError as e:
        logging.error(f"Помилка: Файл не знайдено - {e}.")
    except Exception as e:
        logging.error(f"Виникла непередбачена помилка: {e}", exc_info=True)
