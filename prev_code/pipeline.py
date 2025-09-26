import os
import sys
import time
from itertools import product
import pandas as pd
import numpy as np
from data_io import load_data
from fractals import find_fractals, fractals_after_fvg
from bos import add_bos_after_f2, blank_h4_columns
from mitigation import find_mitigation_v2
from entries import simulate_entry


if __name__ == "__main__":
    results_dir = "backtest_results"
    os.makedirs(results_dir, exist_ok=True)

    # 1) Дані + свіпи
    m5_df, m15_df, fvg_df, m5_bos_df, m1_df, h4_bos_df = load_data()

    # Обрізаємо всі таймфрейми під M5-інтервал
    m5_min, m5_max = m5_df.index.min(), m5_df.index.max()
    start, end = m5_min, m5_max
    m5_df  = m5_df.loc[start:end]
    m15_df = m15_df.loc[start:end]
    m1_df  = m1_df.loc[start:end]
    M15_INDEX_IS_CLOSE = True

    if m1_df.empty:
        print("WARN: M1 empty → fallback-only M5")

    # --- ПАРАМЕТРИ ГРІДА (винесено наперед, бо MAX_H4_KEEP залежить) ---
    fib_levels           = [0.382, 0.5, 0.618, 0.705, 0.75, 1.0]
    stop_offsets         = [0.0, 5.0, 10.0]
    rrs                  = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]
    be_multipliers       = [0, 1, 1.5, 2]
    max_trades_options   = [1, 2, 3]
    h4_required_options  = [0, 1, 2, 3]

    # Скільки останніх H4-BOS лишати в колонках (напр., 3)
    MAX_H4_KEEP = max(h4_required_options)

    # --- Нормалізація напрямку в H4 BOS ---
    DIR_COL = 'direction' if 'direction' in h4_bos_df.columns else ('type' if 'type' in h4_bos_df.columns else None)
    if DIR_COL is None:
        raise ValueError("bos_h4.csv: немає колонки 'direction' або 'type'")

    def _norm_dir(x: str):
        x = str(x).strip().lower()
        if x in ('bull', 'bullish', 'up', 'long', 'buy'):
            return 'bullish'
        if x in ('bear', 'bearish', 'down', 'short', 'sell'):
            return 'bearish'
        return None

    h4_bos_df[DIR_COL] = h4_bos_df[DIR_COL].apply(_norm_dir)

    # Якщо в CSV 'bos_time' — це open бара, а треба підтвердження після закриття,
    # додаємо зсув у 4 години (для H4). Якщо в твоєму CSV вже стоїть саме close —
    # постав H4_BAR_HOURS = 0.
    H4_BAR_HOURS = 4
    h4_bos_df['confirm_time'] = pd.to_datetime(h4_bos_df['bos_time']) + pd.Timedelta(hours=H4_BAR_HOURS)
    h4_bos_df = h4_bos_df.sort_values('confirm_time').reset_index(drop=True)

    # Для швидкого бісекту (перевірка відсортованості)
    _h4_confirm_np = h4_bos_df['confirm_time'].values.astype('datetime64[ns]')
    assert (np.diff(_h4_confirm_np) >= np.timedelta64(0, 'ns')).all()

    # 2) Формуємо сетапи
    m5_fractals = find_fractals(m5_df)
    fvg_res     = fractals_after_fvg(fvg_df, m5_fractals)
    raw_setups  = fvg_res.to_dict(orient="records")

    # Повертаємо оригінальні межі FVG, якщо є
    fvg_map = {}
    if 'fvg_df' in globals():
        time_col = 'time' if 'time' in fvg_df.columns else ('fvg_time' if 'fvg_time' in fvg_df.columns else None)
        if time_col is not None:
            for _, r in fvg_df.iterrows():
                try:
                    k = pd.to_datetime(r[time_col]).floor('min')
                    fvg_map[k] = {'orig_fvg_min': float(r['min']), 'orig_fvg_max': float(r['max'])}
                except Exception:
                    continue

    for s in raw_setups:
        try:
            k = pd.to_datetime(s.get('fvg_time')).floor('min')
            if k in fvg_map:
                s['orig_fvg_min'] = fvg_map[k]['orig_fvg_min']
                s['orig_fvg_max'] = fvg_map[k]['orig_fvg_max']
            else:
                s['orig_fvg_min'] = s.get('fvg_min', None)
                s['orig_fvg_max'] = s.get('fvg_max', None)
        except Exception:
            s['orig_fvg_min'] = s.get('fvg_min', None)
            s['orig_fvg_max'] = s.get('fvg_max', None)

    setups = add_bos_after_f2(raw_setups, m5_bos_df)

    # Вікно покриття M5
    setups = [
        s for s in setups
        if (m5_min - pd.Timedelta(minutes=5)) <= pd.to_datetime(s["bos_time"]) <= (m5_max - pd.Timedelta(minutes=5))
    ]

    # 2.1) Фільтр мітигацій ДО BOS + дедлайн пошуку BOS
    H_MAX = 8  # годин пошуку після gate
    clean_setups = []

    def _norm_fvg(t):
        if t is None:
            return None
        t = str(t).strip().lower()
        return 'bullish' if t.startswith('bull') else ('bearish' if t.startswith('bear') else None)

    for s in setups:
        # Якщо немає збереженої мітигації — розрахувати
        if not s.get("mitigated", False) or pd.isna(s.get("mit_ts_open")):
            try:
                fvg_time = pd.to_datetime(s.get("fvg_time"))
                fmin     = float(s.get("fvg_min", s.get("f1_price")))
                fmax     = float(s.get("fvg_max", s.get("f2_price")))
                z_low, z_high = sorted([fmin, fmax])
                ftype    = s.get("fvg_type")
                mit2     = find_mitigation_v2(fvg_time, z_high, z_low, ftype, m15_df, eps=0.0, m15_index_is_close=True)
                if mit2:
                    s["mitigated"]          = True
                    s["mit_ts_open"]        = mit2["open"]
                    s["mit_bar_open_m15"]   = mit2["open"]
                    s["mit_ts_close_m15"]   = mit2["close"]
                    s["mit_ts_raw"]         = mit2["close"]
                    s["mit_policy"]         = s.get("mit_policy", "cutoff")
                    s["mit_cutoff"]         = mit2["open"]
                    s["cutoff"]             = mit2["open"]
            except Exception:
                pass

        # Базові поля
        fvg_time    = pd.to_datetime(s["fvg_time"])
        m5_bos_time = pd.to_datetime(s["bos_time"])
        s["m5_bos_time"] = m5_bos_time

        # --- (A) Вибір останніх K H4-BOS до FVG за confirm_time ---
        mask_h4 = h4_bos_df["confirm_time"] <= fvg_time
        if not mask_h4.any():
            # сетап потребує H4-контекст, інакше пропускаємо
            continue

        h4_slice = h4_bos_df.loc[mask_h4].sort_values("confirm_time")
        h4_tail  = h4_slice.tail(MAX_H4_KEEP).reset_index(drop=True)

        # Останній (найсвіжіший) BOS
        row_last   = h4_tail.iloc[-1]
        h4_open    = row_last["bos_time"]
        h4_confirm = row_last["confirm_time"]
        h4_dir     = row_last[DIR_COL]  # 'bullish'/'bearish'

        # FVG має бути після confirm останнього BOS
        if fvg_time < h4_confirm:
            continue

        # Напрям валідності беремо з останнього BOS
        s["fvg_type"] = _norm_fvg(s.get("fvg_type"))
        desired_fvg   = 'bullish' if h4_dir == 'bullish' else 'bearish'
        if s["fvg_type"] != desired_fvg:
            continue

        # --- (B) Обмеження вікна пошуку (gate + мітигація) ---
        gate         = max(h4_confirm, fvg_time, m5_bos_time)
        search_start = gate.floor("5min") + pd.Timedelta(minutes=5)

        z_low  = float(s.get('orig_fvg_min', s.get('fvg_min')))
        z_high = float(s.get('orig_fvg_max', s.get('fvg_max')))
        z_low, z_high = sorted([z_low, z_high])

        mit = find_mitigation_v2(fvg_time, z_high, z_low, s["fvg_type"], m15_df, eps=0.0, m15_index_is_close=M15_INDEX_IS_CLOSE)
        if mit is None:
            search_end       = search_start + pd.Timedelta(hours=H_MAX)
            s["mitigated"]   = False
            s["mit_ts_raw"]  = None
            s["mit_ts_open"] = None
            s["mit_cutoff"]  = None
        else:
            mit_open  = mit["open"]
            mit_close = mit["close"]
            # Мітигація до BOS → невалідно
            if mit_open <= m5_bos_time:
                s["valid"] = False
                continue
            search_end = mit_open
            s["mitigated"]          = True
            s["mit_ts_raw"]         = mit_close
            s["mit_ts_open"]        = mit_open
            s["mit_bar_open_m15"]   = mit_open
            s["mit_ts_close_m15"]   = mit_close
            s["mit_policy"]         = "cutoff"
            s["mit_cutoff"]         = mit_open
            s["cutoff"]             = mit_open

        # Фінальні бар’єри
        search_end = min(search_end, m5_max)
        if search_end <= search_start:
            continue
        if not (m5_bos_time < search_start):
            continue

        # Діагностика BOS після FVG
        delta = m5_bos_time - fvg_time
        secs  = int(delta.total_seconds()); mins = secs // 60
        s["bos_after_fvg_sec"] = secs
        s["bos_after_fvg_min"] = mins
        s["bos_after_fvg_str"] = f"{'-' if secs < 0 else ''}{abs(mins)}m {abs(secs) % 60}s"

        # --- (C) Ланцюжок підрядних BOS того ж напрямку (до FVG) ---
        rev = h4_slice.iloc[::-1]  # від найсвіжішого назад
        first_dir = rev.iloc[0][DIR_COL]
        cnt = 0
        for _, r in rev.iterrows():
            if r[DIR_COL] == first_dir:
                cnt += 1
            else:
                break
        s["h4_chain_count"]       = int(cnt)
        s["h4_chain_dir"]         = first_dir
        s["h4_chain_matches_fvg"] = (first_dir == desired_fvg)

        # --- (D) Розкладаємо останні K BOS у s: h4_1_*, h4_2_*, ... ---
        h4_rev = h4_tail.iloc[::-1].reset_index(drop=True)  # 0 = найсвіжіший
        for j in range(1, MAX_H4_KEEP+1):
            if j-1 < len(h4_rev):
                rj = h4_rev.iloc[j-1]
                s[f"h4_{j}_bos_time"]     = pd.to_datetime(rj.get("bos_time"), errors="coerce")
                s[f"h4_{j}_confirm_time"] = pd.to_datetime(rj.get("confirm_time"), errors="coerce")
                s[f"h4_{j}_dir"]          = rj.get(DIR_COL)
                s[f"h4_{j}_level"]        = rj.get("level", rj.get("h4_level", None))
                s[f"h4_{j}_close"]        = rj.get("close", rj.get("h4_close", None))
                raw_fract                 = rj.get("fract_time_kiev", rj.get("fract_time", None))
                s[f"h4_{j}_fract_time"]   = pd.to_datetime(raw_fract, errors="coerce")
            else:
                s[f"h4_{j}_bos_time"]     = None
                s[f"h4_{j}_confirm_time"] = None
                s[f"h4_{j}_dir"]          = None
                s[f"h4_{j}_level"]        = None
                s[f"h4_{j}_close"]        = None
                s[f"h4_{j}_fract_time"]   = None

        # (E) «Соло»-метадані (з останнього BOS) для сумісності/фільтрів
        s["h4_bos_time"]     = h4_open
        s["h4_confirm_time"] = h4_confirm
        s["h4_bos_dir"]      = h4_dir
        s["allowed_side"]    = 'long' if h4_dir == 'bullish' else 'short'

        def _f(x):
            try:
                return float(x)
            except:
                return None
        s["h4_level"] = _f(row_last.get("level", row_last.get("h4_level", None)))
        s["h4_close"] = _f(row_last.get("close", row_last.get("h4_close", None)))

        raw_fract = row_last.get("fract_time_kiev", row_last.get("fract_time", None))
        s["h4_fract_time"]     = pd.to_datetime(raw_fract, errors='coerce')
        s["h4_fract_time_str"] = s["h4_fract_time"].strftime("%Y-%m-%d %H:%M:%S") if not pd.isna(s["h4_fract_time"]) else None

        s["search_start"] = search_start
        s["search_end"]   = search_end

        # Стабільний ключ для джойну з результатами
        s["setup_id"] = f"{int(pd.Timestamp(fvg_time).value)}-{int(pd.Timestamp(m5_bos_time).value)}"

        assert h4_confirm <= fvg_time <= m5_bos_time
        assert search_start < search_end

        s["valid"] = True
        clean_setups.append(s)

    if not clean_setups:
        print("No setups in M5 coverage window (after H4 confirmation)")
        sys.exit(0)

    balance = 5000

    # Сортуємо для детермінованості
    clean_setups.sort(key=lambda s: pd.to_datetime(s.get("h4_bos_time") or s.get("fvg_time")))

    # Кеш фільтрів за h4_required
    filtered_cache = {}
    for h in h4_required_options:
        filtered_cache[h] = [
            s for s in clean_setups
            if s.get("valid", True)
            and s.get("h4_chain_matches_fvg", False)
            and int(s.get("h4_chain_count", 0)) >= h
        ]

    # Плоский список комбінацій
    param_grid = list(product(
        fib_levels,
        stop_offsets,
        rrs,
        be_multipliers,
        max_trades_options,
        h4_required_options
    ))
    total = len(param_grid)
    master_csv = os.path.join(results_dir, "all_results3.csv")
    start_time = time.time()

    # Пропуск уже зроблених комбо
    done_ids = set()
    if os.path.exists(master_csv):
        try:
            hdr = pd.read_csv(master_csv, nrows=0)
            if '_combo_id' in getattr(hdr, 'columns', []):
                done_ids = set(pd.read_csv(master_csv, usecols=['_combo_id'])['_combo_id'].astype(str).unique())
        except Exception:
            done_ids = set()

    buffered_results = []

    for idx, (fib_level, stop_offset, rr, be_mult, max_trades, h4_required) in enumerate(param_grid, start=1):
        print(f"[{idx}/{total}] fib={fib_level} stop={stop_offset} rr={rr} be={be_mult} maxtr={max_trades} h4_req={h4_required}")

        filtered_setups = filtered_cache.get(h4_required, [])
        if not filtered_setups:
            continue

        combo_id = f"fib{fib_level}_stop{stop_offset}_rr{rr}_be{be_mult}_maxtr{max_trades}_h4{h4_required}"
        if combo_id in done_ids:
            print("  -> SKIP (already done)")
            continue

        # Симуляція для цієї комбінації
        raw_results = simulate_entry(
            filtered_setups,
            m5_df,
            m1_df,
            fib_level=fib_level,
            stop_offset=stop_offset,
            rr=rr,
            balance=balance,
            be_multiplier=be_mult,
        )

        df_run = pd.DataFrame(raw_results or [])
        if df_run.empty:
            continue

        # --- Збирання метаданих H4 (включно з h4_j_*) для мерджу за setup_id ---
        setup_map_id = {}
        for s in filtered_setups:
            if "setup_id" not in s:
                continue
            d = {
                "h4_bos_time":     s.get("h4_bos_time"),
                "h4_confirm_time": s.get("h4_confirm_time"),
                "h4_bos_dir":      s.get("h4_bos_dir"),
                "h4_level":        s.get("h4_level"),
                "h4_close":        s.get("h4_close"),
                "h4_fract_time":   s.get("h4_fract_time"),
            }
            for j in range(1, MAX_H4_KEEP+1):
                d[f"h4_{j}_bos_time"]     = s.get(f"h4_{j}_bos_time")
                d[f"h4_{j}_confirm_time"] = s.get(f"h4_{j}_confirm_time")
                d[f"h4_{j}_dir"]          = s.get(f"h4_{j}_dir")
                d[f"h4_{j}_level"]        = s.get(f"h4_{j}_level")
                d[f"h4_{j}_close"]        = s.get(f"h4_{j}_close")
                d[f"h4_{j}_fract_time"]   = s.get(f"h4_{j}_fract_time")
            setup_map_id[s["setup_id"]] = d

        # Fallback мапа по fvg_time (на випадок відсутності setup_id)
        setup_map_fvg = {}
        for s in filtered_setups:
            try:
                k = pd.to_datetime(s["fvg_time"], errors="coerce").floor("min")
                if pd.notna(k):
                    d = {
                        "h4_bos_time":     s.get("h4_bos_time"),
                        "h4_confirm_time": s.get("h4_confirm_time"),
                        "h4_bos_dir":      s.get("h4_bos_dir"),
                        "h4_level":        s.get("h4_level"),
                        "h4_close":        s.get("h4_close"),
                        "h4_fract_time":   s.get("h4_fract_time"),
                    }
                    for j in range(1, MAX_H4_KEEP+1):
                        d[f"h4_{j}_bos_time"]     = s.get(f"h4_{j}_bos_time")
                        d[f"h4_{j}_confirm_time"] = s.get(f"h4_{j}_confirm_time")
                        d[f"h4_{j}_dir"]          = s.get(f"h4_{j}_dir")
                        d[f"h4_{j}_level"]        = s.get(f"h4_{j}_level")
                        d[f"h4_{j}_close"]        = s.get(f"h4_{j}_close")
                        d[f"h4_{j}_fract_time"]   = s.get(f"h4_{j}_fract_time")
                    setup_map_fvg[k] = d
            except Exception:
                pass

        mapped = False
        if "setup_id" in df_run.columns and setup_map_id:
            df_run["h4_bos_time"]     = df_run["setup_id"].map(lambda k: setup_map_id.get(k, {}).get("h4_bos_time"))
            df_run["h4_confirm_time"] = df_run["setup_id"].map(lambda k: setup_map_id.get(k, {}).get("h4_confirm_time"))
            df_run["h4_bos_dir"]      = df_run["setup_id"].map(lambda k: setup_map_id.get(k, {}).get("h4_bos_dir"))
            df_run["h4_level"]        = df_run["setup_id"].map(lambda k: setup_map_id.get(k, {}).get("h4_level"))
            df_run["h4_close"]        = df_run["setup_id"].map(lambda k: setup_map_id.get(k, {}).get("h4_close"))
            df_run["h4_fract_time"]   = df_run["setup_id"].map(lambda k: setup_map_id.get(k, {}).get("h4_fract_time"))
            for j in range(1, MAX_H4_KEEP+1):
                df_run[f"h4_{j}_bos_time"]     = df_run["setup_id"].map(lambda k, j=j: setup_map_id.get(k, {}).get(f"h4_{j}_bos_time"))
                df_run[f"h4_{j}_confirm_time"] = df_run["setup_id"].map(lambda k, j=j: setup_map_id.get(k, {}).get(f"h4_{j}_confirm_time"))
                df_run[f"h4_{j}_dir"]          = df_run["setup_id"].map(lambda k, j=j: setup_map_id.get(k, {}).get(f"h4_{j}_dir"))
                df_run[f"h4_{j}_level"]        = df_run["setup_id"].map(lambda k, j=j: setup_map_id.get(k, {}).get(f"h4_{j}_level"))
                df_run[f"h4_{j}_close"]        = df_run["setup_id"].map(lambda k, j=j: setup_map_id.get(k, {}).get(f"h4_{j}_close"))
                df_run[f"h4_{j}_fract_time"]   = df_run["setup_id"].map(lambda k, j=j: setup_map_id.get(k, {}).get(f"h4_{j}_fract_time"))
            mapped = True

        if (not mapped) and ("fvg_time" in df_run.columns) and setup_map_fvg:
            df_run["fvg_time"]    = pd.to_datetime(df_run["fvg_time"], errors="coerce")
            df_run["__fvg_key__"] = df_run["fvg_time"].dt.floor("min")
            meta_df = (
                pd.DataFrame.from_dict(setup_map_fvg, orient="index")
                .reset_index().rename(columns={"index": "__fvg_key__"})
            )
            df_run = df_run.merge(meta_df, on="__fvg_key__", how="left")
            df_run.drop(columns="__fvg_key__", inplace=True, errors="ignore")

        # Типи для часу/чисел
        time_cols = ["h4_bos_time", "h4_confirm_time", "h4_fract_time"] + \
                    [f"h4_{j}_bos_time" for j in range(1, MAX_H4_KEEP+1)] + \
                    [f"h4_{j}_confirm_time" for j in range(1, MAX_H4_KEEP+1)] + \
                    [f"h4_{j}_fract_time" for j in range(1, MAX_H4_KEEP+1)]
        for c in time_cols:
            if c in df_run.columns:
                df_run[c] = pd.to_datetime(df_run[c], errors="coerce")

        num_cols = ["h4_level", "h4_close"] + \
                   [f"h4_{j}_level" for j in range(1, MAX_H4_KEEP+1)] + \
                   [f"h4_{j}_close" for j in range(1, MAX_H4_KEEP+1)]
        for c in num_cols:
            if c in df_run.columns:
                df_run[c] = pd.to_numeric(df_run[c], errors="coerce")

        # Додаємо мета-параметри гріда
        meta = {
            "fib_level": fib_level,
            "stop_offset": stop_offset,
            "rr": rr,
            "be_multiplier": be_mult,
            "h4_required": h4_required,
            "max_trades": max_trades,
        }
        for k, v in meta.items():
            df_run[k] = v

        # Нормалізуємо час входу
        time_col = next((c for c in ["entry_time", "entry_time_m1", "entry_time_m5"] if c in df_run.columns), None)
        if time_col is not None:
            df_run[time_col] = pd.to_datetime(df_run[time_col], errors="coerce")
            try:
                df_run["entry_time"] = df_run[time_col].dt.tz_localize(None)
            except Exception:
                df_run["entry_time"] = df_run[time_col]
        
        blank_h4_columns(df_run, keep_k=int(h4_required), max_keep=MAX_H4_KEEP)

        df_run = df_run.sort_values("entry_time", ascending=False, ignore_index=True)

        # Маркуємо комбінацію і час
        df_run['_run_ts']   = pd.Timestamp.utcnow().isoformat()
        df_run['_combo_id'] = combo_id

        buffered_results.append(df_run)
        print(f"  -> buffered {len(df_run)} rows")

    # --- Запис у майстер-файл разом, одним махом ---
    master_csv = os.path.join(results_dir, "sol_all_results.csv")
    if buffered_results:
        new_all = pd.concat(buffered_results, ignore_index=True, sort=False)

        if os.path.exists(master_csv):
            try:
                existing = pd.read_csv(master_csv)
            except Exception:
                existing = pd.DataFrame()
        else:
            existing = pd.DataFrame()

        # Забираємо старі рядки з тими ж _combo_id (якщо є)
        if not existing.empty and '_combo_id' in existing.columns:
            new_ids = set(new_all['_combo_id'].astype(str).unique())
            existing = existing[~existing['_combo_id'].astype(str).isin(new_ids)]

        combined = pd.concat([existing, new_all], ignore_index=True, sort=False)

        # Atomic write
        tmp_csv = master_csv + ".tmp"
        combined.to_csv(tmp_csv, index=False)
        os.replace(tmp_csv, master_csv)

        abs_master = os.path.abspath(master_csv)
        print(f"Wrote {len(new_all)} new rows to {abs_master}; total {len(combined)}")

        # Діагностика розміру
        try:
            sz_bytes = os.path.getsize(abs_master)
            print(f"Master CSV size: {sz_bytes/1024/1024:.2f} MB; rows={len(combined):,}; cols={combined.shape[1]}")
        except Exception as e:
            print(f"Could not stat master CSV: {e}")

        # Parquet-копія (якщо є бекенд)
        try:
            parquet_path = os.path.join(results_dir, "all_results3.parquet")
            combined.to_parquet(parquet_path, index=False)
            print(f"Parquet copy saved to {os.path.abspath(parquet_path)}")
        except Exception as e:
            print(f"Parquet not written (pyarrow/fastparquet missing?): {e}")

        # Невелике прев’ю
        preview_cols = [
            "direction","outcome","entry_time","exit_time","entry_price","exit_price",
            "stop","take","fib_level","rr","be_multiplier","setup_id","fvg_time",
            "h4_bos_dir","h4_bos_time","h4_confirm_time"
        ] + sum(([f"h4_{j}_dir", f"h4_{j}_confirm_time"] for j in range(1, MAX_H4_KEEP+1)), [])
        preview_cols = [c for c in preview_cols if c in combined.columns]

        preview_path = os.path.join(results_dir, "all_results3_preview.csv")
        try:
            combined.tail(1000).loc[:, preview_cols].to_csv(preview_path, index=False)
            print(f"Preview (last 1000 rows) saved to {os.path.abspath(preview_path)}")
        except Exception as e:
            print(f"Could not write preview CSV: {e}")
    else:
        print("No new rows produced; nothing to write.")
