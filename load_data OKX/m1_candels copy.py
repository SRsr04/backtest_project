import os
import random
from math import ceil
from datetime import datetime, timedelta
from time import sleep

import pytz
import pandas as pd
from okx.MarketData import MarketAPI

# ======================
# Конфіг
# ======================
inst_id = "BTC-USDT-SWAP"
tf = "1m"

# Максимально дозволено OKX: 300
limit = 300

# Рівно ~5 років (враховуй високосний рік; 1826 = строго 5 календарних)
DAYS_TO_FETCH = 1826

output_file = "btc/m1_candels.csv"

# Рейт-ліміт/стабільність
SAFE_SLEEP_EVERY = 50     # пауза кожні N запитів (для «піддиху»)
SAFE_SLEEP_SEC = 2
REFRESH_CLIENT_EVERY = 1000
RETRY_MAX = 6
RETRY_BASE = 2            # сек (експоненційний бекоф)
RETRY_CAP = 30            # макс пауза між ретрі

# ======================
# Утиліти
# ======================
def from_str_ms(ms_str: str) -> datetime:
    return datetime.fromtimestamp(int(ms_str) / 1000, tz=pytz.UTC)

def get_kyiv_tz():
    try:
        from zoneinfo import ZoneInfo
        return ZoneInfo("Europe/Kyiv")
    except ImportError:
        return pytz.timezone("Europe/Kyiv")

def ensure_dir(path: str):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)

def fetch_with_retry(cl, **kwargs):
    """Виклик OKX з ретрі та експоненційним бекофом."""
    for attempt in range(1, RETRY_MAX + 1):
        try:
            r = cl.get_history_candlesticks(**kwargs)
            return r.get("data", [])
        except Exception as e:
            delay = min(RETRY_BASE * (2 ** (attempt - 1)), RETRY_CAP)
            delay += random.uniform(0, 0.5 * delay)  # джиттер
            print(f"[WARN] {e}. retry {attempt}/{RETRY_MAX} in {delay:.1f}s...")
            sleep(delay)
            # На парних спробах — пересоздаємо клієнт
            if attempt % 2 == 0:
                try:
                    cl = MarketAPI(flag="0", debug=False)
                    print("[INFO] Клієнт пересоздано.")
                except Exception as e2:
                    print(f"[WARN] Не вдалося пересоздати клієнт: {e2}")
    raise RuntimeError("Вичерпано ретрі для get_history_candlesticks")

# ======================
# Основний скрипт (FORWARD)
# ======================
def main():
    print(f"*** OKX {tf} Candles – Last {DAYS_TO_FETCH // 365} Years for {inst_id} ***")

    kyiv_tz = get_kyiv_tz()
    min_dt = datetime.now(tz=pytz.UTC) - timedelta(days=DAYS_TO_FETCH)

    # Скільки сторінок треба при limit=300 (з буфером)
    pages_needed = ceil(DAYS_TO_FETCH * 1440 / limit) + 100

    records = []  # швидше за dict+сортування ключів кожну ітерацію
    cursor = None  # мінімальний ts із попередньої пачки (рух у минуле)

    try:
        cl = MarketAPI(flag="0", debug=False)

        for page in range(1, pages_needed + 1):
            if page % REFRESH_CLIENT_EVERY == 0:
                try:
                    cl = MarketAPI(flag="0", debug=False)
                    print(f"[INFO] REFRESH_CLIENT_EVERY: пересоздано клієнт на сторінці {page}.")
                except Exception as e:
                    print(f"[WARN] Не вдалося пересоздати клієнт: {e}")

            kwargs = dict(instId=inst_id, bar=tf, limit=limit)
            if cursor is not None:
                # Рух у минуле: after = найстаріший ts із попередньої пачки
                kwargs["after"] = str(cursor)

            data = fetch_with_retry(cl, **kwargs)

            if not data:
                print("Даних більше немає. Зупинка.")
                break

            # Обробка пачки
            hit_time_boundary = False
            min_ts_in_batch = None

            for c in data:
                ts = int(c[0])       # мс
                dt = from_str_ms(c[0])

                if min_ts_in_batch is None or ts < min_ts_in_batch:
                    min_ts_in_batch = ts

                if dt < min_dt:
                    hit_time_boundary = True
                    # не додаємо те, що старіше за межу
                    continue

                # Збираємо (дублікати курсора пропустяться природно: ts >= cursor)
                if cursor is None or ts < cursor:
                    records.append({
                        "timestamp_utc": ts,
                        "datetime_utc": dt,      # зручніше конвертнути оптом
                        "open": float(c[1]),
                        "high": float(c[2]),
                        "low": float(c[3]),
                        "close": float(c[4]),
                        "volume": float(c[7]),
                    })

            # Оновлюємо курсор
            if min_ts_in_batch is None or (cursor is not None and min_ts_in_batch >= cursor):
                print("Пагінація не рухається в минуле. Зупинка.")
                break
            cursor = min_ts_in_batch

            if hit_time_boundary:
                print(f"Досягнуто межі {min_dt.date()}. Зупинка.")
                break

            if len(data) < limit:
                print(f"Завершено на сторінці {page} — прийшло менше {limit} свічок.")
                break

            if page % SAFE_SLEEP_EVERY == 0:
                print(f"Сторінка {page} оброблена, коротка пауза...")
                sleep(SAFE_SLEEP_SEC)

        if not records:
            print("Нічого не збережено (порожній результат).")
            return

        # Формуємо DataFrame
        df = pd.DataFrame(records)

        # Конвертація часу: UTC -> Київ; сортування за спаданням
        df["datetime"] = pd.to_datetime(df["datetime_utc"], utc=True).dt.tz_convert(kyiv_tz)
        df = df.drop(columns=["datetime_utc"])
        df = df.sort_values(by="datetime", ascending=False)

        # Колонки як у твоєму файлі
        df = df[["timestamp_utc", "datetime", "open", "high", "low", "close", "volume"]]

        ensure_dir(output_file)
        df.to_csv(output_file, index=False)
        print(f"\n💾 Збережено {len(df)} свічок у файл: {output_file}")

    except KeyboardInterrupt:
        print("Перервано користувачем!")
    except Exception as e:
        print(f"Сталася помилка: {e}")

if __name__ == "__main__":
    main()