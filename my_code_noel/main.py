import time
from pybit.unified_trading import HTTP
from datetime import datetime, timezone, UTC
from typing import Tuple

# Імпортуємо необхідні компоненти
from strategy import TradingStrategy, LIVE_CONFIG # Імпортуємо конфіг звідти ж
from Utils.websocket_client import WS_Client
from constants import API_KEY, API_SECRET # Твої ключі API
import pandas as pd
if __name__ == '__main__':
    print(f"[{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')}] Запуск основного скрипту...")

    # --- 1. Ініціалізація HTTP сесії ---
    try:
        session = HTTP(
            testnet=True, 
            api_key=API_KEY,
            api_secret=API_SECRET
        )
        print("HTTP сесія створена (testnet=True).")
    except Exception as e:
        print(f"[{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')}] ПОМИЛКА створення HTTP сесії: {e}")
        exit()

    # --- 2. Створення екземпляру стратегії ---
    try:
        strategy = TradingStrategy(session, LIVE_CONFIG)
    except Exception as e:
        print(f"[{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')}] ПОМИЛКА створення екземпляру стратегії: {e}")
        exit()

    # --- 3. Створення та запуск WebSocket клієнта ---
    try:
        ws_client = WS_Client(symbol=LIVE_CONFIG['symbol'], strategy_instance=strategy)
        print("WebSocket клієнт створено. Очікування повідомлень...")
    except Exception as e:
        print(f"[{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')}] ПОМИЛКА створення WebSocket клієнта: {e}")
        exit()

    # --- 4. Підтримуємо основний потік живим ---
    try:
        last_heartbeat_time = time.time()
        heartbeat_interval = 300 # 5 хвилин

        while True:
            current_time = time.time()
            if current_time - last_heartbeat_time >= heartbeat_interval:
                active_setups_count = len(getattr(strategy, 'active_setups', []))
                print(f"[{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')}] Бот працює... (Активних сетапів: {active_setups_count})")
                last_heartbeat_time = current_time
            
            time.sleep(10) 

    except KeyboardInterrupt:
        print(f"\n[{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')}] Отримано сигнал зупинки (Ctrl+C). Завершення роботи...")
        if ws_client._ws:
             ws_client._ws.close()
        print(f"[{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')}] Роботу завершено.")