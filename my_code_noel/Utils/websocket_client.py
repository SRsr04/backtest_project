import json
from websocket import WebSocketApp
from datetime import datetime, timedelta, UTC
import time
from typing import List, Tuple, Optional, Deque, Dict

# Потрібен імпорт класу стратегії
from strategy import TradingStrategy # Припускаємо, що TradingStrategy у файлі strategy.py

class WS_Client:
    def __init__(self, symbol: str, strategy_instance: TradingStrategy):
        self.symbol = symbol
        self.strategy = strategy_instance # Зберігаємо екземпляр стратегії
        self.ws_url = 'wss://stream.bybit.com/v5/public/linear'
        self.subscription_arg = f'kline.1.{self.symbol}' # Підписуємось на M1
        self._ws: Optional[WebSocketApp] = None
        self._connect()

    def _on_open(self, ws):
        print(f"[{datetime.now(UTC).strftime('%Y-%m-%d %H:%M')}] WebSocket підключено. Підписуємось на {self.subscription_arg}...")
        ws.send(json.dumps({'op': 'subscribe', 'args': [self.subscription_arg]}))

    def _on_message(self, ws, message):
        try:
            data = json.loads(message)

            if 'op' in data and data['op'] == 'subscribe':
                print(f"[{datetime.now(UTC).strftime('%Y-%m-%d %H:%M')}] Підписка підтверджена: {data.get('success')}")
                return
            if 'ping' in data:
                 ws.send(json.dumps({'op': 'pong', 'req_id': data['req_id']}))
                 return
            if 'topic' in data and self.subscription_arg in data['topic'] and 'data' in data:
                candle_list = data['data']
                for kline_data in candle_list:
                    if kline_data.get('confirm') is True:
                        try:
                            candle_tuple: Tuple = (
                                int(kline_data['start']), float(kline_data['open']), float(kline_data['high']),
                                float(kline_data['low']), float(kline_data['close']), float(kline_data['volume'])
                            )
                            self.strategy.on_m1_candle_close(candle_tuple)
                        except (KeyError, ValueError, TypeError) as e:
                             print(f"[{datetime.now(UTC).strftime('%Y-%m-%d %H:%M')}] Помилка обробки даних свічки: {kline_data}, Помилка: {e}")

        except json.JSONDecodeError:
            print(f"[{datetime.now(UTC).strftime('%Y-%m-%d %H:%M')}] Помилка декодування JSON: {message}")
        except Exception as e:
            print(f"[{datetime.now(UTC).strftime('%Y-%m-%d %H:%M')}] Неочікувана помилка в on_message: {e}\nПовідомлення: {message}")


    def _on_error(self, ws, error):
        print(f"[{datetime.now(UTC).strftime('%Y-%m-%d %H:%M')}] WebSocket Помилка: {error}")

    def _on_close(self, ws, close_status_code, close_msg):
        print(f"[{datetime.now(UTC).strftime('%Y-%m-%d %H:%M')}] WebSocket Закрито: Code={close_status_code}, Msg={close_msg}")
        time.sleep(5)
        print(f"[{datetime.now(UTC).strftime('%Y-%m-%d %H:%M')}] Спроба перепідключення...")
        self._connect()

    def _connect(self):
         print(f"[{datetime.now(UTC).strftime('%Y-%m-%d %H:%M')}] Підключення до WebSocket...")
         self._ws = WebSocketApp(self.ws_url,
                                 on_open=self._on_open,
                                 on_message=self._on_message,
                                 on_error=self._on_error,
                                 on_close=self._on_close)
         import threading
         wst = threading.Thread(target=self._ws.run_forever)
         wst.daemon = True 
         wst.start()