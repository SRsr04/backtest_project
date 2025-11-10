# strategy.py

# strategy.py (з ПОВЕРНУТИМ фільтром SMA)

# strategy.py (з логікою скасування ордерів при інвалідації)

from collections import deque
from datetime import datetime, UTC
from pybit.unified_trading import HTTP
from typing import List, Tuple, Optional, Deque, Dict, Any
import csv
import os

# ... (LIVE_CONFIG, Indicators, MovingAverages, _qty_from_risk - БЕЗ ЗМІН) ...
LIVE_CONFIG = {
    "symbol": "BTCUSDT",
    "rr": 2.5, "risk_usd": 500,
    "sma_fast_period": 14,
    "sma_slow_period": 28,
    "max_trades_per_day": 2,
    "forbidden_entry_days": [],
    "forbidden_entry_hours": []
}

class Indicators: # Ваш код без змін
    @staticmethod
    def detect_fvg(three_candles):
        if not three_candles or len(three_candles) < 3: return (0, 0, 0)
        if not all(isinstance(c, tuple) and len(c) > 3 for c in three_candles): return (0,0,0)
        if three_candles[2][3] > three_candles[0][2]: return (1, three_candles[2][3], three_candles[0][2])
        elif three_candles[2][2] < three_candles[0][3]: return (-1, three_candles[0][3], three_candles[2][2])
        return (0, 0, 0)
    @staticmethod
    def detect_fractal(three_candles):
        if not three_candles or len(three_candles) < 3: return (0, 0)
        if not all(isinstance(c, tuple) and len(c) > 3 for c in three_candles): return (0,0)
        fractal_signal, fractal_level = 0, 0
        is_high = three_candles[1][2] > three_candles[0][2] and three_candles[1][2] > three_candles[2][2]
        is_low = three_candles[1][3] < three_candles[0][3] and three_candles[1][3] < three_candles[2][3]
        if is_high: fractal_signal, fractal_level = 1, three_candles[1][2]
        if is_low: fractal_signal = -1; fractal_level = three_candles[1][3]
        return (fractal_signal, fractal_level)

class MovingAverages: # Ваш код без змін
    @staticmethod
    def calculate_sma(data: List[Tuple], period: int) -> List[Optional[float]]:
        if period <= 0: raise ValueError("Period must be positive")
        sma_values = [None] * len(data)
        if len(data) < period: return sma_values
        window: Deque[float] = deque(maxlen=period); current_sum = 0.0
        for i in range(len(data)):
            if not isinstance(data[i], tuple) or len(data[i]) <= 4: continue
            close_price = data[i][4]
            if len(window) == period: current_sum -= window[0]
            window.append(close_price); current_sum += close_price
            if len(window) == period: sma_values[i] = current_sum / period
        return sma_values
    @staticmethod
    def calculate_ema(data: List[Tuple], period: int) -> List[Optional[float]]:
        if period <= 0: raise ValueError("Period must be positive")
        ema_values = [None] * len(data);
        if not data: return ema_values
        alpha = 2 / (period + 1); sma_values = MovingAverages.calculate_sma(data, period)
        first_valid_idx = -1
        for idx, sma in enumerate(sma_values):
            if sma is not None: ema_values[idx] = sma; first_valid_idx = idx; break
        if first_valid_idx == -1: return ema_values
        for i in range(first_valid_idx + 1, len(data)):
            if not isinstance(data[i], tuple) or len(data[i]) <= 4: ema_values[i] = ema_values[i-1]; continue
            close_price = data[i][4]; ema_values[i] = alpha * close_price + (1 - alpha) * ema_values[i-1]
        return ema_values

def _qty_from_risk(entry_price: float, stop_loss: float, risk_usd: float) -> str: # Ваш код без змін
    dist = abs(entry_price - stop_loss);
    if dist <= 1e-9: return '0'
    qty = risk_usd / dist; return f"{qty:.3f}"

# --- Клас TradingSetup (Без змін) ---
class TradingSetup:
    def __init__(self, fvg_data: Dict[str, Any], strategy_ref: 'TradingStrategy'):
        self.strategy = strategy_ref; self.fvg_data = fvg_data
        self.mitigation_time: float = float('inf'); self.is_active = True
        self.f1_details: Optional[Dict] = None; self.f2_details: Optional[Dict] = None
        self.state = "AWAITING_F1_F2_BOS" 
        print(f"[{datetime.now(UTC).strftime('%Y-%m-%d %H:%M')}] LOG: Створено новий сетап для FVG {self.fvg_data['ts']} (Dir={self.fvg_data['dir']})")

    def check_mitigation(self, m15_candle: Tuple):
        if not self.is_active: return
        m15_ts, _, _, _, m15_close, _ = m15_candle
        if m15_ts <= self.fvg_data['ts']: return
        fvg_dir = self.fvg_data['dir']; fvg_top = self.fvg_data['top']; fvg_bottom = self.fvg_data['bottom']
        is_long_mitigated = fvg_dir == 1 and m15_close < fvg_bottom
        is_short_mitigated = fvg_dir == -1 and m15_close > fvg_top
        if is_long_mitigated or is_short_mitigated:
            print(f"[{datetime.now(UTC).strftime('%Y-%m-%d %H:%M')}] LOG Mitigation: FVG ({self.fvg_data['ts']}) МІТИГОВАНО свічкою {m15_ts} (Close={m15_close}). Сетап деактивовано.")
            self.mitigation_time = m15_ts; self.is_active = False

    def check_fractal(self, fractal_data: Dict[str, Any]):
        if not self.is_active: return
        fractal_signal = fractal_data['signal']; fractal_level = fractal_data['level']
        fractal_formation_ts = fractal_data['formation_ts']; fractal_confirmation_ts = fractal_data['confirmation_ts']
        fvg_ts = self.fvg_data['ts']; fvg_dir = self.fvg_data['dir']
        fvg_top = self.fvg_data['top']; fvg_bottom = self.fvg_data['bottom']
        is_potential_f1 = fractal_confirmation_ts > fvg_ts
        is_valid_f1_long = is_potential_f1 and fvg_dir == 1 and fractal_signal == 1 and fractal_level > fvg_top
        is_valid_f1_short = is_potential_f1 and fvg_dir == -1 and fractal_signal == -1 and fractal_level < fvg_bottom
        is_valid_f1 = is_valid_f1_long or is_valid_f1_short
        if is_valid_f1:
            if not self.f1_details or fractal_formation_ts > self.f1_details['formation_ts']:
                print(f"[{datetime.now(UTC).strftime('%Y-%m-%d %H:%M')}] LOG: Сетап FVG {fvg_ts} ЗНАЙШОВ/ОНОВИВ F1! Тип={fractal_signal}, Рівень={fractal_level} (Conf @ {fractal_confirmation_ts})")
                self.f1_details = fractal_data.copy(); self.f1_details['dir'] = fractal_signal
                self.f2_details = None; self.state = "AWAITING_F2"
                print(f"[{datetime.now(UTC).strftime('%Y-%m-%d %H:%M')}] LOG: F1 оновлено. Скидаємо F2. Стан -> AWAITING_F2")
        if self.f1_details and fractal_confirmation_ts > self.f1_details['confirmation_ts']:
            is_valid_f2_long = fvg_dir == 1 and fractal_signal == -1 and fvg_bottom < fractal_level < fvg_top
            is_valid_f2_short = fvg_dir == -1 and fractal_signal == 1 and fvg_bottom < fractal_level < fvg_top
            correct_pair_type = (fvg_dir == 1 and self.f1_details['dir'] == 1 and fractal_signal == -1) or \
                                (fvg_dir == -1 and self.f1_details['dir'] == -1 and fractal_signal == 1)
            if correct_pair_type and (is_valid_f2_long or is_valid_f2_short):
                if not self.f2_details or fractal_formation_ts > self.f2_details['formation_ts']:
                    print(f"[{datetime.now(UTC).strftime('%Y-%m-%d %H:%M')}] LOG: Сетап FVG {fvg_ts} ЗНАЙШОВ/ОНОВИВ F2! Тип={fractal_signal}, Рівень={fractal_level} (Conf @ {fractal_confirmation_ts})")
                    self.f2_details = fractal_data; self.state = "AWAITING_BOS"
                    print(f"[{datetime.now(UTC).strftime('%Y-%m-%d %H:%M')}] LOG: F2 оновлено. Стан -> AWAITING_BOS")

    def check_bos(self, m5_candle: Tuple):
        if self.state != "AWAITING_BOS" or not self.f1_details or not self.f2_details or not self.is_active:
            return
        m5_ts, _, _, _, m5_close, _ = m5_candle
        if m5_ts <= self.f2_details['confirmation_ts']: return
        if m5_ts >= self.mitigation_time:
            print(f"[{datetime.now(UTC).strftime('%Y-%m-%d %H:%M')}] LOG: Сетап FVG {self.fvg_data['ts']} перевіряв BOS, але вже мітигований. Сетап деактивовано.")
            self.is_active = False; return
        fvg_dir = self.fvg_data['dir']; f1_level = self.f1_details['level']
        is_bos_long = fvg_dir == 1 and m5_close > f1_level
        is_bos_short = fvg_dir == -1 and m5_close < f1_level
        if is_bos_long or is_bos_short:
            print(f"[{datetime.now(UTC).strftime('%Y-%m-%d %H:%M')}] LOG: Сетап FVG {self.fvg_data['ts']} ЗНАЙШОВ BOS! Час M5={m5_ts}, Ціна={m5_close}")
            entry_price = self.f1_details['level']; sl_price = self.f2_details['level']; risk = abs(entry_price - sl_price)
            if risk < 1e-9:
                print(f"[{datetime.now(UTC).strftime('%Y-%m-%d %H:%M')}] LOG: Нульовий ризик для сетапу FVG {self.fvg_data['ts']}. Ордер не розміщено.");
                self.is_active = False; return
            tp_price = entry_price + self.strategy.config['rr'] * risk if fvg_dir == 1 else entry_price - self.strategy.config['rr'] * risk
            
            # --- ЗМІНЕНО: Тепер ми викликаємо інший метод ---
            self.strategy.place_and_monitor_order(fvg_dir, entry_price, sl_price, tp_price)
            self.is_active = False # Деактивуємо сетап (його роботу виконано)

# --- КЛАС TradingStrategy (ОНОВЛЕНО) ---
class TradingStrategy:
    def __init__(self, session: HTTP, config: Dict):
        self.session = session; self.config = config; self.symbol = config['symbol']
        self.risk_usd = config['risk_usd']
        
        self.m5_candles: Deque[Tuple] = deque(maxlen=50)
        self.m15_candles: Deque[Tuple] = deque(maxlen=50)
        self.current_m5: Optional[Tuple] = None; self.current_m15: Optional[Tuple] = None
        
        self.active_setups: List[TradingSetup] = []
        self.trades_today: Dict[str, int] = {}
        
        # --- ДОДАНО: Список для відстеження активних лімітних ордерів ---
        self.pending_limit_orders: List[Dict] = []
        
        self.csv_filename = f"live_trades_{self.symbol}.csv"
        self._init_csv_file()
        
        print(f"[{datetime.now(UTC).strftime('%Y-%m-%d %H:%M')}] Стратегія для {self.symbol} ініціалізована."); print(f"Конфігурація: {self.config}")

    def _init_csv_file(self):
        # ... (Код без змін) ...
        headers = [
            'timestamp_utc', 'symbol', 'side', 'qty',
            'entry_price', 'stop_loss', 'take_profit', 'response_order_id'
        ]
        if not os.path.exists(self.csv_filename):
            try:
                with open(self.csv_filename, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f); writer.writerow(headers)
                print(f"[{datetime.now(UTC).strftime('%Y-%m-%d %H:%M')}] LOG: Створено файл журналу угод: {self.csv_filename}")
            except Exception as e:
                print(f"[{datetime.now(UTC).strftime('%Y-%m-%d %H:%M')}] LOG CRITICAL ERROR: Не вдалося створити CSV файл: {e}")

    def _log_trade_to_csv(self, now: datetime, side: str, qty: str, entry: float, sl: float, tp: float, response: Dict):
        # ... (Код без змін) ...
        try:
            order_id = response.get('result', {}).get('orderId', 'N/A')
            row = [
                now.strftime('%Y-%m-%d %H:%M:%S'), self.symbol, side, qty,
                f"{entry:.2f}", f"{sl:.2f}", f"{tp:.2f}", order_id
            ]
            with open(self.csv_filename, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f); writer.writerow(row)
        except Exception as e:
            print(f"[{datetime.now(UTC).strftime('%Y-%m-%d %H:%M')}] LOG ERROR: Не вдалося записати угоду в CSV: {e}")

    def _update_higher_tf(self, candle_m1: Tuple, tf_minutes: int, current_tf_candle: Optional[Tuple], candle_deque: Deque[Tuple]) -> Optional[Tuple]:
        # ... (Код без змін) ...
        timestamp_m1, o, h, l, c, v = candle_m1
        dt_m1 = datetime.fromtimestamp(timestamp_m1 / 1000.0, UTC)
        is_new_interval = dt_m1.minute % tf_minutes == 0
        new_tf_candle = None
        if current_tf_candle is None or is_new_interval:
            if current_tf_candle is not None:
                closed_candle = current_tf_candle
                candle_deque.append(closed_candle)
                if tf_minutes == 15 and len(self.m15_candles) >= 3:
                     self.check_m15_events(closed_candle)
                if tf_minutes == 5:
                     if len(self.m5_candles) >= 3:
                          self.check_m5_events(closed_candle)
            new_ts = int(dt_m1.replace(minute=(dt_m1.minute // tf_minutes) * tf_minutes, second=0, microsecond=0).timestamp() * 1000)
            new_tf_candle = (new_ts, o, h, l, c, v)
        else:
            ts, op, hi, lo, _, vo = current_tf_candle
            new_hi = max(hi, h); new_lo = min(lo, l); new_vo = vo + v
            new_tf_candle = (ts, op, new_hi, new_lo, c, new_vo)
        return new_tf_candle

    def on_m1_candle_close(self, candle_m1_tuple: Tuple):
        """Оновлює ТФ та перевіряє активні ордери на кожній М1 свічці."""
        if not isinstance(candle_m1_tuple, tuple) or len(candle_m1_tuple) < 6:
            print(f"[{datetime.now(UTC).strftime('%Y-%m-%d %H:%M')}] Отримано некоректні дані M1: {candle_m1_tuple}"); return
        
        # 1. Оновлюємо M5 та M15 (це запустить check_m15/m5_events якщо потрібно)
        self.current_m5 = self._update_higher_tf(candle_m1_tuple, 5, self.current_m5, self.m5_candles)
        self.current_m15 = self._update_higher_tf(candle_m1_tuple, 15, self.current_m15, self.m15_candles)
        
        # --- 2. ДОДАНО: Перевірка активних лімітних ордерів ---
        self.check_pending_orders(candle_m1_tuple)

    # --- check_m15_events (З ПОВЕРНУТИМ ФІЛЬТРОМ SMA) ---
    def check_m15_events(self, closed_m15_candle: Tuple):
        """Обробляє події закриття M15: пошук нових FVG (З ФІЛЬТРОМ SMA) та мітигація старих."""
        now_ts_dt = datetime.now(UTC).strftime('%Y-%m-%d %H:%M')
        self.active_setups = [setup for setup in self.active_setups if setup.is_active]
        for setup in self.active_setups:
            setup.check_mitigation(closed_m15_candle)
            
        if len(self.m15_candles) < 3: return
        last_3_m15 = list(self.m15_candles)[-3:]
        fvg_signal, fvg_top, fvg_bottom = Indicators.detect_fvg(last_3_m15)
        fvg_timestamp = last_3_m15[2][0]

        if fvg_signal != 0:
             # --- ПОВЕРНУТО ФІЛЬТР SMA ---
             fast_period = self.config.get('sma_fast_period', 14) 
             slow_period = self.config.get('sma_slow_period', 28)
             if len(self.m15_candles) < slow_period:
                  print(f"[{now_ts_dt}] LOG SMA Filter: Недостатньо M15 ({len(self.m15_candles)}) для SMA({slow_period}).")
                  return 
             m15_list = list(self.m15_candles)
             sma_fast = MovingAverages.calculate_sma(m15_list, fast_period)[-1]
             sma_slow = MovingAverages.calculate_sma(m15_list, slow_period)[-1]
             if sma_fast is None or sma_slow is None:
                  print(f"[{now_ts_dt}] LOG SMA Filter: SMA ще не розраховані.")
                  return 
             sma_trend = 1 if sma_fast > sma_slow else (-1 if sma_fast < sma_slow else 0)
             if fvg_signal != sma_trend:
                  print(f"[{now_ts_dt}] LOG SMA Filter: FVG ({fvg_signal}) не збігається з трендом SMA ({sma_trend}). FVG ігнорується.")
                  return
             print(f"[{now_ts_dt}] LOG SMA Filter: FVG ({fvg_signal}) збігається з трендом SMA ({sma_trend})...")
             # --- КІНЕЦЬ ФІЛЬТРУ SMA ---

             day_name = datetime.fromtimestamp(fvg_timestamp / 1000.0, UTC).strftime('%A')
             forbidden_days = self.config.get('forbidden_entry_days', [])
             if day_name not in forbidden_days:
                  fvg_exists = False
                  for setup in self.active_setups:
                       if setup.fvg_data['ts'] == fvg_timestamp:
                            fvg_exists = True; break
                  if not fvg_exists:
                       print(f"[{now_ts_dt}] LOG NEW FVG FOUND: Новий FVG M15: Сигнал={fvg_signal}, Top={fvg_top}, Bottom={fvg_bottom}, Час={fvg_timestamp}")
                       fvg_data = {'ts': fvg_timestamp, 'top': fvg_top, 'bottom': fvg_bottom, 'dir': fvg_signal}
                       new_setup = TradingSetup(fvg_data, self)
                       self.active_setups.append(new_setup)
                       print(f"[{now_ts_dt}] LOG: Створено новий сетап. Загалом активних сетапів: {len(self.active_setups)}")

    # --- check_m5_events (Без змін) ---
    def check_m5_events(self, closed_m5_candle: Tuple):
        for setup in self.active_setups:
             setup.check_bos(closed_m5_candle)
        if len(self.m5_candles) < 3: return
        last_3_m5 = list(self.m5_candles)[-3:]
        fractal_signal, fractal_level = Indicators.detect_fractal(last_3_m5)
        if fractal_signal != 0:
            fractal_data = {
                'signal': fractal_signal, 'level': fractal_level,
                'formation_ts': last_3_m5[1][0], 'confirmation_ts': last_3_m5[2][0]
            }
            for setup in self.active_setups:
                setup.check_fractal(fractal_data)
        self.active_setups = [setup for setup in self.active_setups if setup.is_active]

    # --- ЗМІНЕНО: Тепер це `place_and_monitor_order` ---
    def place_and_monitor_order(self, direction: int, entry_price: float, sl_price: float, tp_price: float):
        """Розміщує ордер І додає його до списку моніторингу."""
        now = datetime.now(UTC).strftime('%Y-%m-%d %H:%M'); day_name = now.strftime('%A'); hour = now.hour; day_key = now.strftime('%Y-%m-%d')
        forbidden_days = self.config.get('forbidden_entry_days', [])
        forbidden_hours = self.config.get('forbidden_entry_hours', [])
        max_daily_trades = self.config.get('max_trades_per_day', 999)
        
        print(f"[{datetime.now(UTC).strftime('%Y-%m-%d %H:%M')}] LOG: Спроба розмістити ордер. Перевірка фільтрів...")
        if day_name in forbidden_days: print(f"[{now}] LOG: Вхід заборонено: День ({day_name})"); return
        if hour in forbidden_hours: print(f"[{now}] LOG: Вхід заборонено: Година ({hour}:00)"); return
        today_trade_count = self.trades_today.get(day_key, 0)
        if today_trade_count >= max_daily_trades: print(f"[{now}] LOG: Вхід заборонено: Ліміт угод ({day_key}, {today_trade_count}/{max_daily_trades})"); return
        print(f"[{datetime.now(UTC).strftime('%Y-%m-%d %H:%M')}] LOG: Усі фільтри пройдені.")

        qty_str = _qty_from_risk(entry_price, sl_price, self.risk_usd)
        if not qty_str or float(qty_str) <= 0: print(f"[{now}] LOG ERROR: Помилка Qty (entry={entry_price}, sl={sl_price})"); return
        side = "Buy" if direction == 1 else "Sell"
        print(f"[{now}] LOG: ===> РОЗМІЩЕННЯ ОРДЕРА <==="); print(f"LOG: Символ={self.symbol}, Сторона={side}, Qty={qty_str}"); print(f"LOG: Вхід={entry_price:.2f}, SL={sl_price:.2f}, TP={tp_price:.2f}")
        try:
             price_precision = 1; sl_precision = 1; tp_precision = 1
             response = self.session.place_order(
                 category='linear', symbol=self.symbol, side=side, orderType='Limit', qty=qty_str,
                 price=f"{entry_price:.{price_precision}f}", takeProfit=f"{tp_price:.{tp_precision}f}",
                 stopLoss=f"{sl_price:.{sl_precision}f}", timeInForce='GTC'
             )
             print(f"[{now}] LOG: Відповідь біржі: {response}")
             if response.get('retCode') == 0:
                  print(f"[{now}] LOG: Ордер успішно розміщено."); self.trades_today[day_key] = today_trade_count + 1
                  self._log_trade_to_csv(now, side, qty_str, entry_price, sl_price, tp_price, response)
                  
                  # --- ДОДАНО: Реєстрація ордера для моніторингу ---
                  order_id = response.get('result', {}).get('orderId', 'N/A')
                  if order_id != 'N/A':
                      self.pending_limit_orders.append({
                          'orderId': order_id,
                          'symbol': self.symbol,
                          'side': side,
                          'entry_price': entry_price,
                          'tp_price': tp_price,
                          'sl_price': sl_price
                      })
                      print(f"[{now}] LOG: Ордер {order_id} додано до моніторингу.")
                  # --- КІНЕЦЬ ---
                  
             else: print(f"[{now}] LOG ERROR: ПОМИЛКА розміщення ордера: {response.get('retMsg')}")
        except Exception as e: print(f"[{now}] LOG CRITICAL ERROR: КРИТИЧНА ПОМИЛКА при відправці ордера: {e}")

    # --- ДОДАНО: Новий метод для перевірки активних ордерів ---
    def check_pending_orders(self, m1_candle: Tuple):
        """Перевіряє всі активні лімітні ордери на інвалідацію (TP до Entry) або заповнення."""
        if not self.pending_limit_orders:
            return

        # (ts, o, h, l, c, v)
        m1_ts, _, m1_high, m1_low, _, _ = m1_candle
        now = datetime.fromtimestamp(m1_ts / 1000.0, UTC)

        # Ітеруємо по копії, щоб безпечно видаляти елементи з оригіналу
        for order in self.pending_limit_orders.copy():
            order_id = order['orderId']
            entry_price = order['entry_price']
            tp_price = order['tp_price']
            side = order['side']

            is_filled = False
            is_invalidated = False

            if side == "Buy":
                # 1. Перевірка заповнення (ціна торкнулась або пробила вниз вхід)
                if m1_low <= entry_price:
                    is_filled = True
                # 2. Перевірка інвалідації (ціна торкнулась TP *до* того, як торкнулась входу)
                elif m1_high >= tp_price:
                    is_invalidated = True
            
            elif side == "Sell":
                # 1. Перевірка заповнення (ціна торкнулась або пробила вверх вхід)
                if m1_high >= entry_price:
                    is_filled = True
                # 2. Перевірка інвалідації (ціна торкнулась TP *до* того, як торкнулась входу)
                elif m1_low <= tp_price:
                    is_invalidated = True
            
            if is_filled:
                print(f"[{now}] LOG: Лімітний ордер {order_id} ({side} @ {entry_price}) ймовірно ЗАПОВНЕНО (M1 Low={m1_low}, M1 High={m1_high}). Припиняємо відстеження.")
                self.pending_limit_orders.remove(order)
            
            elif is_invalidated:
                print(f"[{now}] LOG: Ордер {order_id} ({side} @ {entry_price}) ІНВАЛІДОВАНО! Ціна досягла TP ({tp_price}) *до* входу (M1 Low={m1_low}, M1 High={m1_high}). СКАСУВАННЯ...")
                self.cancel_limit_order(order)
                self.pending_limit_orders.remove(order)

    # --- ДОДАНО: Новий метод для скасування ордера ---
    def cancel_limit_order(self, order_details: Dict):
        """Відправляє запит на скасування ордера на біржу."""
        try:
            response = self.session.cancel_order(
                category="linear",
                symbol=order_details['symbol'],
                orderId=order_details['orderId']
            )
            print(f"[{datetime.now(UTC).strftime('%Y-%m-%d %H:%M')}] LOG: Відповідь на скасування ордера {order_details['orderId']}: {response}")
            if response.get('retCode') != 0:
                 print(f"[{datetime.now(UTC).strftime('%Y-%m-%d %H:%M')}] LOG ERROR: Не вдалося скасувати ордер: {response.get('retMsg')}")
        except Exception as e:
            print(f"[{datetime.now(UTC).strftime('%Y-%m-%d %H:%M')}] LOG CRITICAL ERROR: Помилка скасування ордера {order_details['orderId']}: {e}")