import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import logging
from time_utils import to_naive_kyiv

logging.basicConfig(
    filename='trade_search.log',
    filemode='w',                # перезаписувати файл при кожному запуску
    level=logging.INFO,
    format='%(asctime)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def load_data():
    """
    Завантажує CSV із різних таймфреймів, уніфікує часові колонки до наївного київського часу
    та повертає підготовлені DataFrame-и (M5, M15, FVG, BOS, M1, H4) для подальшої обробки.
    """
    m5_df  = pd.read_csv('m5_candels.csv')
    m15_df = pd.read_csv('m15_candels.csv')
    m1_df  = pd.read_csv('m1_candels.csv')
    fvg_df = pd.read_csv('fvg_m15.csv')
    m5_bos_df = pd.read_csv('bos_m5.csv')
    h4_bos_df = pd.read_csv('bos_h4.csv')

    for df in (m5_df, m15_df, m1_df):
        if 'timestamp_utc' in df.columns:
            df.drop(columns=['timestamp_utc'], inplace=True)

    for name, bos_df in (("m5_bos_df", m5_bos_df), ("h4_bos_df", h4_bos_df)):
        # fract_time: може містити рядки з +/-offsets — парсимо з utc=True, конвертуємо в Kyiv, робимо naive
        if 'fract_time' in bos_df.columns:
            bos_df['fract_time'] = pd.to_datetime(bos_df['fract_time'], errors='coerce', utc=True) \
                                    .dt.tz_convert('Europe/Kyiv') \
                                    .dt.tz_localize(None)

        # fract_time_kiev: теж парсимо з utc=True на випадок змішаних форматів, конвертуємо, робимо naive
        if 'fract_time_kiev' in bos_df.columns:
            bos_df['fract_time_kiev'] = pd.to_datetime(bos_df['fract_time_kiev'], errors='coerce', utc=True) \
                                            .dt.tz_convert('Europe/Kyiv') \
                                            .dt.tz_localize(None)

        # Уніфікуємо канонічну колонку BOS-часу → 'bos_time' (Kyiv naive)
        if 'bos_time_kiev' in bos_df.columns:
            bos_df['bos_time'] = to_naive_kyiv(bos_df['bos_time_kiev'], tz_name='Europe/Kyiv')
        elif 'bos_time' in bos_df.columns:
            bos_df['bos_time'] = to_naive_kyiv(bos_df['bos_time'], tz_name='Europe/Kyiv')
        # якщо нічого з вищезгаданого немає — залишимо як є; нижче буде загальна обробка/перевірка

        # записати назад
        if name == "m5_bos_df":
            m5_bos_df = bos_df
        else:
            h4_bos_df = bos_df

    for df, cols in [
        (m5_df,  ['datetime']),
        (m15_df, ['datetime']),
        (m1_df,  ['datetime']),
        (fvg_df, ['time']),
        (m5_bos_df, ['bos_time']),
        (h4_bos_df, ['bos_time']),
    ]:
        for c in cols:
            # якщо колонки немає — пропускаємо
            if c not in df.columns:
                continue

            s = df[c]

            # Інша тактика: спочатку намагаємося розпарсити як UTC (щоб покрити випадки з офсетами).
            parsed = pd.to_datetime(s, utc=True, errors="coerce")

            # Якщо майже всі NaT після utc-парсингу — спробуємо без utc (naive місцевий формат)
            if parsed.isna().all():
                parsed = pd.to_datetime(s, errors="coerce")

            # Якщо щось вдалося розпарсити — привести до Kyiv-wall-time (наївний) і записати
            if not parsed.isna().all():
                try:
                    # якщо parsed має tz -> конвертуємо в Kyiv, потім зробимо naive (drop tz)
                    if getattr(parsed.dt, "tz", None) is not None:
                        df[c] = parsed.dt.tz_convert("Europe/Kyiv").dt.tz_localize(None)
                    else:
                        # parsed без tz — вважаємо що це вже локальний Kyiv time
                        df[c] = parsed.dt.tz_localize(None)
                except Exception:
                    # fallback на твою утиліту, якщо щось не так
                    df[c] = to_naive_kyiv(s, tz_name="Europe/Kyiv")
            else:
                # нічого не розпарсилось — покладаємось на to_naive_kyiv (вона робить більш тонкий підхід)
                df[c] = to_naive_kyiv(s, tz_name="Europe/Kyiv")

            # сортування і дедуп
            try:
                df.sort_values(c, inplace=True)
                df.drop_duplicates(subset=[c], keep='last', inplace=True)
                df.reset_index(drop=True, inplace=True)
            except Exception:
                pass

    # приберемо допоміжну колонку, якщо була
    for df in (m5_bos_df, h4_bos_df):
        if 'bos_time_kiev' in df.columns:
            df.drop(columns=['bos_time_kiev'], inplace=True)

    # Індексами робимо наївний datetime (київський wall-time), сортуємо
    m5_df  = m5_df.set_index('datetime').sort_index()
    m1_df  = m1_df.set_index('datetime').sort_index()
    m15_df = m15_df.set_index('datetime').sort_index()

    m5_bos_df = m5_bos_df.sort_values('bos_time')
    h4_bos_df = h4_bos_df.sort_values('bos_time')
    fvg_df = fvg_df.sort_values('time')

    return m5_df, m15_df, fvg_df, m5_bos_df, m1_df, h4_bos_df
