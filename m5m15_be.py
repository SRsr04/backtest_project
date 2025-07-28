import pandas as pd
import numpy as np
import logging
from datetime import timedelta, datetime
import pytz
import tqdm
import os 

LOG_DIR = 'btc'
# Замість фіксованого імені файлу, краще додати timestamp, щоб мати історію логів
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE_NAME = os.path.join(LOG_DIR, '1_m5m15_be.log') # Додано timestamp до імені файлу

# Перевірка та створення директорії для логів, якщо вона не існує
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
    # logging.info(f"Створено директорію для логів: '{LOG_DIR}'")
    # Цей log.info також буде спробу логувати в консоль, тому його можна прибрати
    # або переконатися, що він виконається до налаштування хендлерів.
    # Для чистоти, краще залишити його для файлу.

root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG) # Встановлюємо рівень DEBUG для запису всіх повідомлень у файл

# Видаляємо існуючі обробники, щоб уникнути дублювання логів при повторному запуску
# Це важливо, якщо ваш скрипт може запускатися кілька разів в одній сесії.
if root_logger.handlers:
    for handler in root_logger.handlers:
        root_logger.removeHandler(handler)

# Обробник для запису логів у файл (ВСІ логи DEBUG і вище)
file_handler = logging.FileHandler(LOG_FILE_NAME, mode='w', encoding='utf-8')
file_handler.setLevel(logging.DEBUG) # Записуємо всі логи у файл
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)
root_logger.addHandler(file_handler)
root_logger.info(f"Логування налаштовано. Детальні логи записуються у файл '{LOG_FILE_NAME}'. Прогрес виконання буде відображатися в терміналі.")


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
    
def breakeven_info(entry_price: float, volume: float, side: str, fee_percent: float = 0.0325):
    """
    Розраховує потенційну комісію і ціну беззбитковості (для стопу).

    :param entry_price: ціна входу
    :param volume: обʼєм угоди
    :param side: "long" або "short"
    :param fee_percent: комісія за сторону в % (наприклад, 0.0325 для 0.0325%)
    :return: (total_fee, breakeven_price)
    """
    if not isinstance(entry_price, (int, float)) or entry_price <= 0:
        logging.error("Некоректна ціна входу (entry_price).")
        return 0.0, 0.0
    if not isinstance(volume, (int, float)) or volume <= 0:
        logging.error("Некоректний об'єм угоди (volume).")
        return 0.0, 0.0
    if side.lower() not in ["long", "short"]:
        logging.error("Сторона угоди має бути 'long' або 'short'.")
        return 0.0, 0.0
    if not isinstance(fee_percent, (int, float)) or fee_percent < 0:
        logging.error("Некоректний відсоток комісії (fee_percent).")
        return 0.0, 0.0

    notional = entry_price * volume
    # Комісія береться від номіналу за кожну сторону угоди (вхід + вихід)
    total_fee = notional * (fee_percent / 100)

    breakeven_price = 0.0
    if side.lower() == "long":
        breakeven_price = (notional + total_fee) / volume
    elif side.lower() == "short":
        breakeven_price = (notional - total_fee) / volume

    return total_fee, breakeven_price
def check_trade_close(trade, current_candle_data):
    """
    Перевіряє, чи була закрита поточна угода (тейк-профіт, стоп-лосс або беззбиток).
    Застосовує консервативний підхід: якщо SL/BE та TP досягнуті в одній свічці,
    угода вважається закритою по стоп-лоссу/беззбитку (пріоритет SL/BE над TP).
    Враховує комісію для розрахунку PnL.

    Args:
        trade (dict): Словник з деталями відкритої угоди. Повинен містити:
                      'type', 'entry_price', 'stop_loss', 'take_profit', 'position_size',
                      'breakeven_price_with_fee', 'commission_paid_total'.
        current_candle_data (pd.Series): Рядок даних для поточної свічки
                                       (з 'open', 'high', 'low', 'close', та індексом time).

    Returns:
        tuple: (outcome, exit_time, exit_price, pnl)
               outcome: 'win' (TP), 'loss' (SL), 'breakeven' (BE), 'pending' (ще відкрита).
               exit_time (pd.Timestamp): Час закриття угоди.
               exit_price (float): Ціна закриття угоди.
               pnl (float): Прибуток/збиток від угоди (після комісії).
    """
    trade_type = trade['type']
    # 'stop_loss' тут - це початковий SL або пересунутий BE
    current_active_stop_loss = trade['stop_loss'] 
    take_profit = trade['take_profit']
    entry_price = trade['entry_price']
    position_size = trade['position_size']
    
    # Нові поля для беззбитку та комісії
    breakeven_price_with_fee = trade.get('breakeven_price_with_fee', 0.0)
    total_commission_for_trade = trade.get('commission_paid_total', 0.0)

    # Дані поточної свічки
    open_price = current_candle_data['open'] # Додано open_price для точнішого визначення
    high = current_candle_data['high']
    low = current_candle_data['low']
    current_time = current_candle_data.name

    outcome = 'pending'
    exit_time = current_time
    exit_price = np.nan
    pnl = 0

    logging.debug(f"\n--- Перевірка свічки {current_time.strftime('%Y-%m-%d %H:%M')} для угоди {trade_type.upper()} ---")
    logging.debug(f"Candle: Open={open_price:.4f}, High={high:.4f}, Low={low:.4f}")
    logging.debug(f"Угода: Entry={entry_price:.4f}, Active SL={current_active_stop_loss:.4f}, TP={take_profit:.4f}, BE={breakeven_price_with_fee:.4f}, Comm={total_commission_for_trade:.4f}")

    # Визначення, чи досягнуті рівні
    sl_reached = False
    tp_reached = False
    be_reached = False # Новий прапорець для беззбитку

    # Порівняння з плаваючою комою
    EPSILON = 1e-9

    if trade_type == 'long':
        # Перевірка досягнення SL (або BE, якщо SL на ньому)
        if low <= current_active_stop_loss:
            sl_reached = True
        
        # Перевірка досягнення TP
        if high >= take_profit:
            tp_reached = True
        
        # Перевірка досягнення BE, якщо стоп ще не був переміщений на BE
        # АБО якщо поточний_активний_стоп_лосс - це вже і є BE
        if low <= breakeven_price_with_fee: # Ціна опустилася до рівня BE
            be_reached = True

        # Логіка пріоритету: спочатку перевіряємо SL/BE, потім TP
        if sl_reached and tp_reached:
            # Обидва досягнуті. Якщо open вже "перетнув" обидва, то закриття по тому, хто ближче до open.
            # Якщо open між ними, то хто був досягнутий першим після open.
            
            # Якщо open вже нижче SL, це втрата
            if open_price <= current_active_stop_loss:
                outcome = 'loss'
                exit_price = current_active_stop_loss
                logging.debug(f"Лонг: ❌❌ SL ({current_active_stop_loss:.4f}) та TP ({take_profit:.4f}) досягнуто. Open ({open_price:.4f}) вже нижче SL. Вихід по SL.")
            # Якщо open вже вище TP, це виграш
            elif open_price >= take_profit:
                outcome = 'win'
                exit_price = take_profit
                logging.debug(f"Лонг: ✅✅ SL ({current_active_stop_loss:.4f}) та TP ({take_profit:.4f}) досягнуто. Open ({open_price:.4f}) вже вище TP. Вихід по TP.")
            # Якщо open між SL та TP, дивимося, хто ближче до open (або хто був досягнутий першим)
            else:
                # В цій логіці, ми даємо пріоритет SL/BE над TP, якщо вони досягнуті одночасно
                # тобто, якщо свічка торкнулась SL/BE, навіть якщо вона потім торкнулась TP
                # (консервативний підхід)
                if abs(open_price - current_active_stop_loss) <= abs(open_price - take_profit):
                     exit_price = current_active_stop_loss
                     if abs(current_active_stop_loss - breakeven_price_with_fee) < EPSILON:
                         outcome = 'breakeven'
                         logging.debug(f"Лонг: 🅿️🅿️ BE ({breakeven_price_with_fee:.4f}) та TP ({take_profit:.4f}) досягнуто в одній свічці. Пріоритет BE.")
                     else:
                         outcome = 'loss'
                         logging.debug(f"Лонг: ❌❌ SL ({current_active_stop_loss:.4f}) та TP ({take_profit:.4f}) досягнуто в одній свічці. Пріоритет SL.")
                else:
                    outcome = 'win'
                    exit_price = take_profit
                    logging.debug(f"Лонг: ✅✅ TP ({take_profit:.4f}) та SL ({current_active_stop_loss:.4f}) досягнуто в одній свічці. Пріоритет TP (бо він ближче до Open).")
        
        elif sl_reached: # Якщо тільки SL/BE досягнутий
            exit_price = current_active_stop_loss
            if abs(current_active_stop_loss - breakeven_price_with_fee) < EPSILON: # Якщо SL знаходиться на рівні BE
                outcome = 'breakeven'
                logging.debug(f"Лонг: 🅿️ Беззбиток спрацював на {current_time.strftime('%Y-%m-%d %H:%M')} за ціною {exit_price:.4f}.")
            else:
                outcome = 'loss'
                logging.debug(f"Лонг: ❌ Стоп-лосс спрацював на {current_time.strftime('%Y-%m-%d %H:%M')} за ціною {exit_price:.4f}.")
        
        elif tp_reached: # Якщо тільки TP досягнутий
            outcome = 'win'
            exit_price = take_profit
            logging.debug(f"Лонг: ✅ Тейк-профіт спрацював на {current_time.strftime('%Y-%m-%d %H:%M')} за ціною {take_profit:.4f}.")

    elif trade_type == 'short':
        # Перевірка досягнення SL (або BE, якщо SL на ньому)
        if high >= current_active_stop_loss:
            sl_reached = True

        # Перевірка досягнення TP
        if low <= take_profit:
            tp_reached = True

        # Перевірка досягнення BE, якщо стоп ще не був переміщений на BE
        if high >= breakeven_price_with_fee: # Ціна піднялася до рівня BE
            be_reached = True

        # Логіка пріоритету для шорту
        if sl_reached and tp_reached:
            # Якщо open вже вище SL, це втрата
            if open_price >= current_active_stop_loss:
                outcome = 'loss'
                exit_price = current_active_stop_loss
                logging.debug(f"Шорт: ❌❌ SL ({current_active_stop_loss:.4f}) та TP ({take_profit:.4f}) досягнуто. Open ({open_price:.4f}) вже вище SL. Вихід по SL.")
            # Якщо open вже нижче TP, це виграш
            elif open_price <= take_profit:
                outcome = 'win'
                exit_price = take_profit
                logging.debug(f"Шорт: ✅✅ SL ({current_active_stop_loss:.4f}) та TP ({take_profit:.4f}) досягнуто. Open ({open_price:.4f}) вже нижче TP. Вихід по TP.")
            # Якщо open між SL та TP, дивимося, хто ближче до open (або хто був досягнутий першим)
            else:
                # Пріоритет SL/BE над TP
                if abs(open_price - current_active_stop_loss) <= abs(open_price - take_profit):
                    exit_price = current_active_stop_loss
                    if abs(current_active_stop_loss - breakeven_price_with_fee) < EPSILON:
                        outcome = 'breakeven'
                        logging.debug(f"Шорт: 🅿️🅿️ BE ({breakeven_price_with_fee:.4f}) та TP ({take_profit:.4f}) досягнуто в одній свічці. Пріоритет BE.")
                    else:
                        outcome = 'loss'
                        logging.debug(f"Шорт: ❌❌ SL ({current_active_stop_loss:.4f}) та TP ({take_profit:.4f}) досягнуто в одній свічці. Пріоритет SL.")
                else:
                    outcome = 'win'
                    exit_price = take_profit
                    logging.debug(f"Шорт: ✅✅ TP ({take_profit:.4f}) та SL ({current_active_stop_loss:.4f}) досягнуто в одній свічці. Пріоритет TP (бо він ближче до Open).")
        
        elif sl_reached:
            exit_price = current_active_stop_loss
            if abs(current_active_stop_loss - breakeven_price_with_fee) < EPSILON:
                outcome = 'breakeven'
                logging.debug(f"Шорт: 🅿️ Беззбиток спрацював на {current_time.strftime('%Y-%m-%d %H:%M')} за ціною {exit_price:.4f}.")
            else:
                outcome = 'loss'
                logging.debug(f"Шорт: ❌ Стоп-лосс спрацював на {current_time.strftime('%Y-%m-%d %H:%M')} за ціною {exit_price:.4f}.")
        
        elif tp_reached:
            outcome = 'win'
            exit_price = take_profit
            logging.debug(f"Шорт: ✅ Тейк-профіт спрацював на {current_time.strftime('%Y-%m-%d %H:%M')} за ціною {take_profit:.4f}.")
    
    # Розрахунок PnL, якщо угода закрита
    if outcome != 'pending':
        if trade_type == 'long':
            pnl = (exit_price - entry_price) * position_size
        elif trade_type == 'short':
            pnl = (entry_price - exit_price) * position_size
        
        # Віднімаємо загальну комісію з PnL
        pnl -= total_commission_for_trade
        logging.debug(f"Фактично розрахований PnL (з комісією): {pnl:.4f} (Outcome: {outcome})")
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



def load_and_prepare_data(candles_path, fvg_path, bos_path, higher_tf_candles_path):
    """Завантажує, обробляє та ОБ'ЄДНУЄ всі дані в єдиний "Майстер-DataFrame"."""
    kiev_tz = pytz.timezone('Europe/Kiev')
    
    def handle_timezone(dt_series):
        """Допоміжна функція для коректної обробки часових зон."""
        if dt_series.dt.tz is None:
            # Якщо дані "наївні", локалізуємо їх до Києва
            return dt_series.dt.tz_localize(kiev_tz, ambiguous='NaT', nonexistent='NaT')
        else:
            # Якщо дані вже "обізнані", конвертуємо їх до Києва
            return dt_series.dt.tz_convert(kiev_tz)

    def load_csv_to_datetime_index(file_path, date_col_name='datetime'):
        df = pd.read_csv(file_path, encoding='utf-8')
        # Застосовуємо парсер та коректно обробляємо часову зону
        dt_series = pd.to_datetime(df[date_col_name].apply(custom_date_parser), errors='coerce')
        df[date_col_name] = handle_timezone(dt_series)
        
        df.dropna(subset=[date_col_name], inplace=True)
        df = df.set_index(date_col_name).sort_index()
        return df[~df.index.duplicated(keep='first')]

    df_candles = load_csv_to_datetime_index(candles_path)
    df_higher_tf_candles = load_csv_to_datetime_index(higher_tf_candles_path)
    
    df_fvg = load_csv_to_datetime_index(fvg_path, 'time')
    df_fvg.rename(columns={'type': 'fvg_type', 'min': 'fvg_min', 'max': 'fvg_max', 'middle': 'fvg_middle'}, inplace=True)

    df_bos = load_csv_to_datetime_index(bos_path, 'bos_time_kiev')
    df_bos.rename(columns={'type': 'bos_type', 'level': 'bos_level'}, inplace=True)
    
    # Застосовуємо ту ж логіку для fract_time_kiev
    fract_time_series = pd.to_datetime(df_bos['fract_time_kiev'].apply(custom_date_parser), errors='coerce')
    df_bos['fract_time_kiev'] = handle_timezone(fract_time_series)
    
    fractal_high_prices = df_candles['high'].to_dict()
    fractal_low_prices = df_candles['low'].to_dict()

    def get_fractal_price(row):
        fract_time = row['fract_time_kiev']
        if row['bos_type'] == 'bullish':
            return fractal_high_prices.get(fract_time)
        elif row['bos_type'] == 'bearish':
            return fractal_low_prices.get(fract_time)
        return None
    
    df_bos['bos_fractal_price'] = df_bos.apply(get_fractal_price, axis=1)
    
    logging.info("Починаємо об'єднання даних...")
    df_master = pd.merge_asof(df_candles, df_higher_tf_candles.add_suffix('_m15'), left_index=True, right_index=True, direction='backward')
    df_master = df_master.join(df_fvg[['fvg_type', 'fvg_min', 'fvg_max', 'fvg_middle']], how='left')
    df_master = df_master.join(df_bos[['bos_type', 'bos_level', 'bos_fractal_price']], how='left')

    df_master['is_high_fractal'] = (df_master['high'].shift(1) < df_master['high']) & (df_master['high'] > df_master['high'].shift(-1))
    df_master['is_low_fractal'] = (df_master['low'].shift(1) > df_master['low']) & (df_master['low'] < df_master['low'].shift(-1))

    for col in ['fvg_type', 'bos_type']: df_master[col].fillna('', inplace=True)
    for col in ['fvg_min', 'fvg_max', 'fvg_middle', 'bos_level', 'bos_fractal_price']: df_master[col].fillna(0, inplace=True)
    
    logging.info(f"Підготовка даних завершена. Розмір фінального DataFrame: {len(df_master)}.")
    return df_master

# --- ФУНКЦІЯ БЕКТЕСТУ ---

def run_optimized_backtest(df_master, start_date, end_date, initial_balance, risk_per_trade, target_rr, params):
    """
    Виконує повний ОПТИМІЗОВАНИЙ бектест для стратегії M5/M15
    БЕЗ H1 фільтра та З логікою переміщення в беззбиток.
    Повертає DataFrame з результатами угод та список значень equity curve.
    """
    logging.info("Починаємо оптимізований бектест M5/M15...")
    print("Запускаємо оптимізований бектест M5/M15...")

    df_backtest = df_master.loc[start_date:end_date].copy()
    if df_backtest.empty:
        logging.warning("DataFrame для бектесту порожній після фільтрації за датами.")
        return pd.DataFrame(), [initial_balance]

    fvg_events = df_backtest[df_backtest['fvg_type'] != ''].copy()
    potential_setups, used_bos_times = [], set()
    SEARCH_WINDOW = timedelta(hours=12)

    print(f"Етап 1/2: Знайдено {len(fvg_events)} FVG. Шукаємо сетапи...")
    for fvg_time, fvg_row in tqdm.tqdm(fvg_events.iterrows(), desc="Пошук сетапів"):
        if not is_trading_hour(fvg_time): continue
        df_search = df_backtest.loc[fvg_time + timedelta(minutes=5) : fvg_time + SEARCH_WINDOW]
        if df_search.empty: continue
        if (fvg_row['fvg_type'] == 'bullish' and (df_search['close_m15'] < fvg_row['fvg_min']).any()) or \
           (fvg_row['fvg_type'] == 'bearish' and (df_search['close_m15'] > fvg_row['fvg_max']).any()):
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
                if entry_cand.empty or (entry_cand.index[0] - bos.name > timedelta(minutes=30)) or not is_trading_hour(entry_cand.index[0]): continue
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
                if entry_cand.empty or (entry_cand.index[0] - bos.name > timedelta(minutes=30)) or not is_trading_hour(entry_cand.index[0]): continue
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
    for setup in tqdm.tqdm(potential_setups, desc="Симулюємо угоди"):
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
                
                # --- ЛОГІКА ПЕРЕМІЩЕННЯ В БЕЗЗБИТОК (БУ) ---
                if not active_trade['breakeven_activated'] and candle['bos_type'] != '':
                    is_valid_bos_type = (active_trade['type'] == 'long' and candle['bos_type'] == 'bullish') or \
                                        (active_trade['type'] == 'short' and candle['bos_type'] == 'bearish')
                    
                    bos_fractal_price = candle.get('bos_fractal_price', None)
                    if bos_fractal_price is None or pd.isna(bos_fractal_price):
                        continue
                    
                    if is_valid_bos_type:
                        entry_price = active_trade['entry_price']
                        bos_close_price = candle['bos_level']
                        
                        move_to_be = False
                        if active_trade['type'] == 'long' and bos_close_price > entry_price and bos_fractal_price > entry_price:
                            move_to_be = True
                        elif active_trade['type'] == 'short' and bos_close_price < entry_price and bos_fractal_price < entry_price:
                            move_to_be = True
                        
                        if move_to_be:
                            active_trade.update({
                                'stop_loss': active_trade['breakeven_price_with_fee'], 
                                'breakeven_activated': True, 
                                'next_bos_after_entry_time': candle.name, 
                                'next_bos_after_entry_type': candle['bos_type']
                            })
                # --- КІНЕЦЬ ЛОГІКИ БУ ---

    if not all_trades:
        logging.info("Немає закритих угод для аналізу.")
        return pd.DataFrame(), equity_curve_values
    
    results_df = pd.DataFrame(all_trades)
    desired_order = ['type', 'fvg_type', 'fvg_start_time', 'fvg_range', 'fractal_1_price', 'fractal_2_price', 'entry_time', 'entry_price', 'initial_stop_loss', 'stop_loss', 'take_profit', 'outcome', 'pnl', 'exit_time', 'exit_price', 'commission_paid_total', 'breakeven_price_with_fee', 'breakeven_activated', 'next_bos_after_entry_time', 'next_bos_after_entry_type']
    results_df = results_df.reindex(columns=[col for col in desired_order if col in results_df.columns])
    
    results_dir, file_path = 'backtest_results', os.path.join('backtest_results', "optimized_be_m5m15_final.csv")
    os.makedirs(results_dir, exist_ok=True)
    results_df.to_csv(file_path, index=False, date_format='%Y-%m-%d %H:%M:%S')
    print(f"\nБектест завершено. Знайдено {len(all_trades)} угод. Результати в '{file_path}'")
    
    # --- ЗМІНА: Повертаємо і DataFrame, і список equity ---
    return results_df, equity_curve_values


# --- ГОЛОВНИЙ БЛОК ВИКОНАННЯ ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    CANDLES_FILE = 'm5_candels.csv'
    FVG_FILE = 'fvg_m15.csv'
    BOS_FILE = 'bos_m5.csv'
    HIGHER_TF_CANDLES_FILE = 'm15_candels.csv' # Це мабуть close_m15

    try:
        # load_and_prepare_data має бути визначена у вашому файлі.
        # Для цього сетапу вона повинна приймати 4 аргументи.
        df_master = load_and_prepare_data(
            CANDLES_FILE, 
            FVG_FILE, 
            BOS_FILE, 
            HIGHER_TF_CANDLES_FILE
        )
        if df_master.empty: raise ValueError("DataFrame порожній.")

        kiev_tz = pytz.timezone('Europe/Kiev')
        start_date = kiev_tz.localize(datetime(2020, 7, 19))
        end_date = kiev_tz.localize(datetime(2025, 7, 18, 23, 59, 59))
        
        initial_account_balance = 5000.0 # ЗМІНЕНО НА 5000.0
        
        # --- ЗМІНА: Тепер отримуємо два значення з функції бектесту ---
        trade_results_df, equity_values = run_optimized_backtest(
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
            
            print("\n         ЗBTC  M5M15")
            print("="*40)
            print(f"Угоди: {total_trades} (W: {wins}, L: {losses}, BE: {breakevens})")
            print(f"WinRate: {win_rate:.2f}%")
            print(f"PnL: {total_pnl:.2f}")
            print(f"Максимальна просадка: ${max_dd_absolute:.2f} ({max_dd_percent:.2f}%)") # <-- ДОДАНО ТУТ
            print("="*40)

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