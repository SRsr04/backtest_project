import pandas as pd
import numpy as np
import pytz
import tqdm
from datetime import datetime, timedelta
import logging

# --- 1. КОНФІГУРАЦІЯ ТА ПАРАМЕТРИ ---
CONFIG = {
    'name': 'm5m15_ruleset_v2',
    'fvg_tf': 'm15',
    'entry_tf': 'm5',
    'files': {
        'candles_ltf': 'm5_candels.csv',
        'candles_htf': 'm15_candels.csv',
        'fvg': 'fvg_m15.csv',
        'bos': 'bos_m5.csv',
    }
}
INITIAL_BALANCE = 5000.0
RISK_PER_TRADE_PERCENT = 1.0
TARGET_RR = 2.6

# --- 2. ДОПОМІЖНІ ФУНКЦІЇ ---

def setup_logging():
    """Налаштовує логування для виводу ДЕТАЛЬНОЇ діагностики у файл."""
    # Усі повідомлення, включаючи DEBUG, будуть записуватись у файл.
    # filemode='w' означає, що файл буде перезаписуватись при кожному запуску.
    logging.basicConfig(level=logging.DEBUG,
                        filename='backtest_debug.log', 
                        filemode='w',
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    
    # Додамо також вивід INFO повідомлень в консоль, щоб бачити основні етапи
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logging.getLogger().addHandler(console_handler)

    logging.info("Логування налаштовано. Детальна діагностика буде у файлі 'backtest_debug.log'")


def custom_date_parser(date_str):
    if pd.isna(date_str): return pd.NaT
    return pd.to_datetime(date_str, errors='coerce')

def is_trading_hour(current_time: pd.Timestamp) -> bool:
    # Правило 5.1
    return 7 <= current_time.hour < 23 and current_time.dayofweek < 5

def breakeven_info(entry_price, position_size, trade_type):
    commission_percent = 0.04
    commission = entry_price * position_size * (commission_percent / 100) * 2
    commission_per_unit = commission / position_size
    be_price = entry_price + commission_per_unit if trade_type == 'long' else entry_price - commission_per_unit
    return commission, be_price

def calculate_drawdown(equity_curve, initial_balance):
    if not equity_curve: return 0.0, 0.0
    peak = initial_balance
    max_drawdown_abs = 0.0
    for equity in equity_curve:
        if equity > peak:
            peak = equity
        drawdown = peak - equity
        if drawdown > max_drawdown_abs:
            max_drawdown_abs = drawdown
    percent_drawdown = (max_drawdown_abs / initial_balance) * 100 if initial_balance > 0 else 0
    return max_drawdown_abs, percent_drawdown


def calculate_max_losing_streak(outcomes):
    if not outcomes: return 0
    max_streak = 0
    current_streak = 0
    for outcome in outcomes:
        if outcome == 'loss':
            current_streak += 1
        else:
            if current_streak > max_streak:
                max_streak = current_streak
            current_streak = 0
    if current_streak > max_streak:
        max_streak = current_streak
    return max_streak

def load_and_prepare_data(config):
    """
    Завантажує та готує дані з надійною обробкою часових зон.
    """
    print("Завантаження даних (v3, з виправленням часових зон)...")
    kiev_tz = pytz.timezone('Europe/Kiev')
    files = config['files']

    def robust_time_processing(df, time_col_name):
        """Надійно обробляє колонку з часом."""
        # Крок 1: Конвертуємо в datetime, приводячи все до єдиного стандарту UTC.
        # Це коректно обробить і рядки з часовою зоною, і без неї.
        dt_series = pd.to_datetime(df[time_col_name], utc=True, errors='coerce')
        
        # Крок 2: Тепер, коли всі дати гарантовано в форматі datetime, конвертуємо в київський час.
        df[time_col_name] = dt_series.dt.tz_convert(kiev_tz)
        df.dropna(subset=[time_col_name], inplace=True)
        df.set_index(time_col_name, inplace=True)
        df.sort_index(inplace=True)
        return df[~df.index.duplicated(keep='first')]

    # --- Завантаження файлів з використанням нової функції ---
    df_ltf = robust_time_processing(pd.read_csv(files['candles_ltf']), 'datetime')
    df_htf = robust_time_processing(pd.read_csv(files['candles_htf']), 'datetime')
    df_fvg = robust_time_processing(pd.read_csv(files['fvg']), 'time')
    df_bos = robust_time_processing(pd.read_csv(files['bos']), 'bos_time_kiev')

    # --- Решта коду залишається без змін ---
    df_htf = df_htf.add_suffix('_htf')
    df_htf['time_htf'] = df_htf.index

    df_fvg.rename(columns={'type': 'fvg_type', 'min': 'fvg_min', 'max': 'fvg_max'}, inplace=True)
    df_bos.rename(columns={'type': 'bos_type'}, inplace=True)

    print("Об'єднання даних...")
    df_master = pd.merge_asof(df_ltf, df_htf, left_index=True, right_index=True, direction='backward')
    df_master = df_master.join(df_fvg[['fvg_type', 'fvg_min', 'fvg_max']], how='left')
    df_master = df_master.join(df_bos[['bos_type']], how='left')
    print("Дані успішно об'єднані.")

    df_master['is_high_fractal'] = (df_master['high'].shift(1) > df_master['high'].shift(2)) & (df_master['high'].shift(1) > df_master['high'])
    df_master['is_low_fractal'] = (df_master['low'].shift(1) < df_master['low'].shift(2)) & (df_master['low'].shift(1) < df_master['low'])

    df_master.fillna({
        'is_high_fractal': False, 'is_low_fractal': False,
        'fvg_type': '', 'bos_type': ''
    }, inplace=True)
    
    df_master.dropna(subset=['close_htf'], inplace=True)

    print("Підготовка даних завершена. Запускаємо основну логіку...")
    return df_master
# --- 3. НОВА ЛОГІКА БЕКТЕСТУ ---

def run_backtest(df, config):
    print("Початок симуляції бектесту (v5, виправлення 'зомбі FVG')...")
    
    # --- Змінні стану системи ---
    all_trades = []
    open_trade = None
    active_setup = None
    
    # Змінні для пошуку нового сетапу
    active_fvg = None
    active_f1 = None
    active_f2 = None
    
    # Допоміжні змінні
    consumed_fvgs = set()
    trades_today_count = 0
    last_processed_date = None
    losing_streak = 0
    days_to_skip = 0
    
    current_balance = INITIAL_BALANCE
    equity_curve = [INITIAL_BALANCE]
    outcomes = []

    for i in tqdm.tqdm(range(2, len(df)), desc="Симуляція свічка за свічкою"):
        current_time = df.index[i]
        current_candle = df.iloc[i]

        # --- БЛОК 1: Управління часом та лімітами ---
        if last_processed_date is None or current_time.date() != last_processed_date:
            last_processed_date = current_time.date()
            trades_today_count = 0
            if days_to_skip > 0 and current_time.dayofweek < 5:
                days_to_skip -= 1
                logging.info(f"[{current_time.date()}] Торговий день пропущено через лузстрік. Залишилось пропустити: {days_to_skip} дн.")

        if days_to_skip > 0: continue
        
        # --- БЛОК 2: Управління відкритою позицією ---
        if open_trade:
            if i % 5000 == 0:
                 logging.debug(f"[{current_time}] Супроводжуємо відкриту угоду ({open_trade['type']})...")

            trade_closed = False
            exit_price = -1
            outcome = 'pending'

            if (not open_trade.get('be_triggered') and current_time > open_trade['entry_time'] and
                ((open_trade['type'] == 'long' and current_candle['bos_type'] == 'bullish') or
                 (open_trade['type'] == 'short' and current_candle['bos_type'] == 'bearish'))):
                open_trade['sl'] = open_trade['be_price']
                open_trade['be_triggered'] = True
                logging.info(f"[{current_time}] BOS у напрямку позиції. SL переведено в BE.")
            
            if open_trade['type'] == 'long':
                if current_candle['low'] <= open_trade['sl']: trade_closed, exit_price, outcome = True, open_trade['sl'], 'loss'
                elif current_candle['high'] >= open_trade['tp']: trade_closed, exit_price, outcome = True, open_trade['tp'], 'win'
            else: # Short
                if current_candle['high'] >= open_trade['sl']: trade_closed, exit_price, outcome = True, open_trade['sl'], 'loss'
                elif current_candle['low'] <= open_trade['tp']: trade_closed, exit_price, outcome = True, open_trade['tp'], 'win'

            if not trade_closed and current_time.hour == 23 and current_time.minute >= 55:
                trade_closed, exit_price, outcome = True, current_candle['close'], 'eod_close'
                logging.info(f"[{current_time}] Примусове закриття угоди в кінці дня (EOD).")

            if trade_closed:
                if outcome in ['win', 'loss']:
                    pnl = (exit_price - open_trade['entry_price']) * open_trade['pos_size'] - open_trade['commission'] if open_trade['type'] == 'long' else (open_trade['entry_price'] - exit_price) * open_trade['pos_size'] - open_trade['commission']
                    if outcome == 'loss': pnl = -abs(pnl)
                else: # eod_close
                    pnl = (exit_price - open_trade['entry_price']) * open_trade['pos_size'] if open_trade['type'] == 'long' else (open_trade['entry_price'] - exit_price) * open_trade['pos_size']
                    pnl -= open_trade.get('commission', 0) / 2
                
                current_balance += pnl
                equity_curve.append(current_balance)
                outcomes.append(outcome)
                
                if outcome == 'loss': losing_streak += 1
                else: losing_streak = 0
                
                if losing_streak >= 5:
                    days_to_skip = 2
                    losing_streak = 0
                    logging.info(f"[{current_time}] Досягнуто лузстрік 5. Торгівля зупинена на 2 дні.")
                
                open_trade['outcome'] = outcome
                open_trade['pnl'] = pnl
                open_trade['exit_time'] = current_time
                all_trades.append(open_trade)
                logging.info(f"[{current_time}] Угоду закрито. Тип: {open_trade['type']}, Результат: {outcome}, PnL: {pnl:.2f}")
                open_trade = None
                continue
            
            continue

        # --- БЛОК 3: Управління активним сетапом (віртуальний ордер) ---
        if active_setup:
            if current_time > active_setup['bos_time'] + timedelta(hours=1):
                logging.info(f"[{current_time}] Активний сетап скасовано через закінчення терміну дії.")
                active_setup = None
            else:
                entry_price, tp_price = active_setup['entry_price'], active_setup['tp']
                
                if (active_setup['type'] == 'long' and current_candle['high'] >= tp_price) or \
                   (active_setup['type'] == 'short' and current_candle['low'] <= tp_price):
                    logging.info(f"[{current_time}] Активний сетап скасовано, ціна досягла TP без входу.")
                    active_setup = None
                elif (active_setup['type'] == 'long' and current_candle['low'] <= entry_price) or \
                     (active_setup['type'] == 'short' and current_candle['high'] >= entry_price):
                    
                    if is_trading_hour(current_time) and active_setup['bos_time'].date() == current_time.date() and trades_today_count < 2:
                        risk_abs = abs(entry_price - active_setup['sl'])
                        pos_size = (current_balance * (RISK_PER_TRADE_PERCENT / 100)) / risk_abs
                        commission, be_price = breakeven_info(entry_price, pos_size, active_setup['type'])
                        
                        open_trade = {**active_setup, 'entry_time': current_time, 'pos_size': pos_size, 'commission': commission, 'be_price': be_price, 'be_triggered': False}
                        consumed_fvgs.add(open_trade['fvg_time'])
                        trades_today_count += 1
                        
                        logging.info(f"[{current_time}] ВХІД В УГОДУ! Тип: {open_trade['type']}, Ціна: {entry_price:.5f}")
                        
                        active_setup = active_fvg = active_f1 = active_f2 = None
                        continue

        if trades_today_count >= 2: continue

        # --- БЛОК 4: Пошук нового сетапу ---
        if current_candle['fvg_type'] != '' and current_time not in consumed_fvgs:
            if active_fvg is None or active_fvg['start_time'] != current_time:
                active_fvg = {
                    'start_time': current_time, 'type': current_candle['fvg_type'],
                    'min': current_candle['fvg_min'], 'max': current_candle['fvg_max'],
                    'search_start_time': current_time + timedelta(minutes=15)
                }
                logging.info(f"Знайдено/Оновлено до нового FVG ({active_fvg['type']}) на {current_time}.")
                active_f1 = active_f2 = None
        
        if active_fvg:
            if current_candle['time_htf'] > active_fvg['start_time']:
                if (active_fvg['type'] == 'bullish' and current_candle['close_htf'] > active_fvg['max']) or \
                   (active_fvg['type'] == 'bearish' and current_candle['close_htf'] < active_fvg['min']):
                    logging.info(f"FVG ({active_fvg['start_time']}) мітиговано. Скидання пошуку.")
                    active_fvg = active_f1 = active_f2 = None
                    continue

            if current_time >= active_fvg['search_start_time']:
                is_bullish_context = active_fvg['type'] == 'bullish'
                
                if not active_f1:
                    if not is_trading_hour(current_time):
                        logging.debug(f"[{current_time}] Пошук f1: Не торговий час.")
                    else:
                        if (is_bullish_context and current_candle['is_high_fractal']):
                            fractal_level = df.iloc[i-1]['high']
                            logging.debug(f"[{current_time}] Пошук f1 (Bullish): Знайдено High Fractal з вершиною {fractal_level:.2f}.")
                            if fractal_level > active_fvg['max']:
                                active_f1 = df.iloc[i-1]
                                logging.info(f"ЗНАЙДЕНО F1 ({active_f1.name}) для FVG {active_fvg['start_time']}.")
                            else:
                                logging.debug(f"[{current_time}] Пошук f1: Фрактал НЕ підходить (вершина {fractal_level:.2f} <= max FVG {active_fvg['max']:.2f}).")
                        
                        elif (not is_bullish_context and current_candle['is_low_fractal']):
                            fractal_level = df.iloc[i-1]['low']
                            logging.debug(f"[{current_time}] Пошук f1 (Bearish): Знайдено Low Fractal з дном {fractal_level:.2f}.")
                            if fractal_level < active_fvg['min']:
                                active_f1 = df.iloc[i-1]
                                logging.info(f"ЗНАЙДЕНО F1 ({active_f1.name}) для FVG {active_fvg['start_time']}.")
                            else:
                                logging.debug(f"[{current_time}] Пошук f1: Фрактал НЕ підходить (дно {fractal_level:.2f} >= min FVG {active_fvg['min']:.2f}).")

                elif not active_f2:
                    if not is_trading_hour(current_time):
                        logging.debug(f"[{current_time}] Пошук f2: Не торговий час.")
                    else:
                        if (is_bullish_context and current_candle['is_high_fractal'] and current_candle['high'] > active_fvg['max']):
                            active_f1 = df.iloc[i-1]
                            active_f2 = None
                            logging.info(f"[{current_time}] Знайдено новий F1 ({active_f1.name}), старий скасовано.")
                        elif (is_bullish_context and current_candle['is_low_fractal'] and current_candle['low'] <= active_fvg['max']):
                            active_f2 = df.iloc[i-1]
                            logging.info(f"[{current_time}] Знайдено F2 ({active_f2.name}).")
                        elif (not is_bullish_context and current_candle['is_low_fractal'] and current_candle['low'] < active_fvg['min']):
                            active_f1 = df.iloc[i-1]
                            active_f2 = None
                            logging.info(f"[{current_time}] Знайдено новий F1 ({active_f1.name}), старий скасовано.")
                        elif (not is_bullish_context and current_candle['is_high_fractal'] and current_candle['high'] >= active_fvg['min']):
                            active_f2 = df.iloc[i-1]
                            logging.info(f"[{current_time}] Знайдено F2 ({active_f2.name}).")

                elif active_f1 and active_f2:
                    if not is_trading_hour(current_time):
                        logging.debug(f"[{current_time}] BOS: Не торговий час.")
                    elif (is_bullish_context and current_candle['bos_type'] == 'bullish' and current_candle['close'] > active_f1['high']) or \
                         (not is_bullish_context and current_candle['bos_type'] == 'bearish' and current_candle['close'] < active_f1['low']):
                        
                        trade_type = 'long' if is_bullish_context else 'short'
                        f1_price = active_f1['high'] if trade_type == 'long' else active_f1['low']
                        f2_price = active_f2['low'] if trade_type == 'long' else active_f2['high']
                        
                        entry_price = f2_price + (f1_price - f2_price) * 0.62 if trade_type == 'long' else f2_price - (f2_price - f1_price) * 0.62
                        sl_price = f2_price
                        tp_price = entry_price + (entry_price - sl_price) * TARGET_RR if trade_type == 'long' else entry_price - (sl_price - entry_price) * TARGET_RR
                        
                        active_setup = {
                            'type': trade_type, 'fvg_time': active_fvg['start_time'], 'bos_time': current_time,
                            'entry_price': entry_price, 'sl': sl_price, 'tp': tp_price,
                            'f1_time': active_f1.name, 'f2_time': active_f2.name
                        }
                        logging.info(f"!!! {current_time}: Сформовано новий АКТИВНИЙ СЕТАП. Тип: {trade_type}, Вхід: {entry_price:.5f}.")
                        
                        active_fvg = active_f1 = active_f2 = None
    
    print("Симуляція завершена.")
    return pd.DataFrame(all_trades), equity_curve, outcomes


# --- 4. ОСНОВНИЙ БЛОК ВИКОНАННЯ ---
if __name__ == "__main__":
    setup_logging()
    try:
        df_master = load_and_prepare_data(CONFIG)
        
        results, equity, outcomes = run_backtest(df_master, CONFIG)
        
        print("\n--- Результати бектесту ---")
        if results.empty:
            print("Не знайдено жодної угоди.")
        else:
            total_trades = len(results)
            wins = len(results[results['outcome'] == 'win'])
            losses = len(results[results['outcome'] == 'loss'])
            win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
            total_pnl = results['pnl'].sum()
            final_balance = INITIAL_BALANCE + total_pnl
            
            max_dd_abs, max_dd_perc = calculate_drawdown(equity, INITIAL_BALANCE)
            max_ls = calculate_max_losing_streak(outcomes)

            print(f"Початковий баланс: ${INITIAL_BALANCE:.2f}")
            print(f"Кінцевий баланс: ${final_balance:.2f}")
            print(f"Загальний PnL: ${total_pnl:.2f} ({(total_pnl/INITIAL_BALANCE*100):.2f}%)")
            print(f"Угоди: {total_trades} (W: {wins}, L: {losses})")
            print(f"WinRate: {win_rate:.2f}%")
            print(f"Макс. просадка: ${max_dd_abs:.2f} ({max_dd_perc:.2f}%)")
            print(f"Макс. лузстрік: {max_ls} угод")
            
            results.to_csv(f"trades_{CONFIG['name']}.csv", index=False)
            pd.Series(equity).to_csv(f"equity_{CONFIG['name']}.csv", index=False)
            print(f"\nРезультати збережено у файли trades_{CONFIG['name']}.csv та equity_{CONFIG['name']}.csv")

    except FileNotFoundError as e:
        print(f"\nПОМИЛКА: Файл не знайдено. Перевірте шляхи: {e}")
    except Exception as e:
        print(f"\nВиникла непередбачена помилка: {e}")
        import traceback
        traceback.print_exc()