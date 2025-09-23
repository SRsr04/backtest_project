import pandas as pd
import numpy as np

def simulate_entry(setups, m5_df, m1_df, mode=None, *, fib_level, stop_offset, rr, balance, be_multiplier=0, entry_timeout_candles=None, h4_required=None):
    """
    Логіка:
    - Шукаємо перший M5-бар, що торкає entry у вікні [search_start, search_end].
    - Перед входом перевіряємо "take-before-entry" від limit_placed_time (або search_start) ДОКУПИ з баром входу.
      Якщо тейк торкнуто раніше або в ту ж хвилину (на M1), сетап скасовується.
    - У першій M5 після входу порядок подій вирішує M1 (пріоритет STOP → TAKE при конфліктах у хвилині).
    - BE активується лише на закритті М5 і діє з наступного бару.
    """
    results = []
    EPS = 1e-9

    def finish(direction, setup, outcome_, exit_ts, exit_px, entry_time, entry_price, initial_stop, take_price, be_stop, search_start, limit_ts, f1p, f2p):
        # minutes від search_start до entry_time (усічено)
        minutes = None
        try:
            et = pd.to_datetime(entry_time)
            ss = pd.to_datetime(search_start)
            if et.tzinfo is None and getattr(ss, "tzinfo", None) is not None:
                et = et.tz_localize(ss.tzinfo)
            elif ss.tzinfo is None and getattr(et, "tzinfo", None) is not None:
                ss = ss.tz_localize(et.tzinfo)
            if not pd.isna(et) and not pd.isna(ss):
                minutes = int((et - ss).total_seconds() // 60)
                if minutes < 0:
                    minutes = 0
        except Exception:
            minutes = None

        row = {
            'direction': direction,
            'outcome': outcome_,
            'fvg_time': setup.get('fvg_time'),
            'f1_price': f1p,
            'f2_price': f2p,
            'fib_level': fib_level,
            'limit_placed_time': limit_ts,
            'entry_time_m5': entry_time,
            'search_start_m5': search_start,
            'entry_price': entry_price,
            'stop': initial_stop,
            'take': take_price,
            'bos_time': setup.get('bos_time'),
            'exit_time': exit_ts,
            'exit_price': exit_px,
            'be_triggered': (outcome_ == 'be'),
            'be_price': be_stop,
            'time_to_entry': minutes,
        }
        # стабільні ключі + H4/мітигація (якщо присутні в setup)
        if 'setup_id' in setup:       row['setup_id'] = setup['setup_id']
        if 'allowed_side' in setup:   row['allowed_side'] = setup['allowed_side']

        # --- НОВЕ: усе про мітки мітигації та політику ---
        for k in ("mitigated", "mit_ts", "mit_ts_raw", "mit_ts_open",
                  "mit_ts_close_m15", "mit_bar_open_m15",
                  "mit_cutoff", "mit_policy"):
            if k in setup:
                row[k] = setup[k]

        # (Опційно) кінцева межа вікна пошуку, якою реально користувалися при вході
        if '__entry_window_end__' in setup:
            row['entry_window_end'] = setup['__entry_window_end__']

        results.append(row)

    def _infer_mit_open_if_needed(mit_open, mit_close, *, m15_index_is_close=True):
        if pd.isna(mit_open) and pd.notna(mit_close):
            return (mit_close - pd.Timedelta(minutes=15)) if m15_index_is_close else mit_close.floor("15min")
        # випадок коли хтось поклав CLOSE у поле cutoff/open — виправляємо
        if pd.notna(mit_open) and pd.notna(mit_close) and mit_open == mit_close and m15_index_is_close:
            return mit_close - pd.Timedelta(minutes=15)
        return mit_open

    def take_hit_before_entry(pre_start, entry_time, entry_price, take_price, direction, m1_df, m5_df, EPS):
        """
        True → тейк був до/у ту ж хвилину, що і перше торкання entry → сетап скасовуємо.
        """
        # 1) Спроба визначити хвилину входу на M1 (більш точний порядок)
        if m1_df is not None and not m1_df.empty:
            m1_full = m1_df.loc[pre_start: entry_time]  # включно з хвилиною входу
            if not m1_full.empty:
                if direction == 'long':
                    touched_entry = (m1_full['low'] <= entry_price + EPS)
                    if touched_entry.any():
                        entry_min_ts = m1_full.index[int(np.argmax(touched_entry.values))]
                        # тейк доторкнуто до або в ту ж хвилину → скасувати
                        if (m1_full.loc[:entry_min_ts, 'high'] >= take_price - EPS).any():
                            return True
                else:  # short
                    touched_entry = (m1_full['high'] >= entry_price - EPS)
                    if touched_entry.any():
                        entry_min_ts = m1_full.index[int(np.argmax(touched_entry.values))]
                        if (m1_full.loc[:entry_min_ts, 'low'] <= take_price + EPS).any():
                            return True
        # 2) Консервативний фолбек на M5: включаємо бар входу
        m5_pre = m5_df.loc[pre_start: entry_time]
        if m5_pre.empty:
            return False
        if direction == 'long':
            return (m5_pre['high'] >= take_price - EPS).any()
        else:
            return (m5_pre['low']  <= take_price + EPS).any()

    for setup in setups:
        # межі пошуку
        bos_time = pd.to_datetime(setup.get('bos_time'))
        # search_start/search_end пришли з підготовчого етапу; якщо ні — фолбек на bos_time+5m / +8h
        search_start = pd.to_datetime(setup.get('search_start') or (bos_time + pd.Timedelta(minutes=5)))
        search_end   = pd.to_datetime(setup.get('search_end')   or (search_start + pd.Timedelta(hours=8)))
        if pd.isna(search_start) or pd.isna(search_end) or (search_start >= search_end):
            continue

        # сторони: поважаємо allowed_side, якщо задано
        allowed_side = setup.get('allowed_side', None)

        # ціни
        f1p, f2p = float(setup['f1_price']), float(setup['f2_price'])
        if setup['f1_type'] == 'high':
            direction = 'long'
            entry_price   = f2p + (f1p - f2p) * fib_level
            initial_stop  = f2p - stop_offset
        else:
            direction = 'short'
            entry_price   = f2p - (f2p - f1p) * fib_level
            initial_stop  = f2p + stop_offset

        if allowed_side and ((allowed_side == 'long' and direction != 'long') or (allowed_side == 'short' and direction != 'short')):
            continue

        distance = abs(entry_price - initial_stop)
        if distance <= EPS:
            continue

        take_price = entry_price + distance * rr if direction == 'long' else entry_price - distance * rr

        # ризик/комісія → BE-стоп
        risk_usd = balance * 0.01
        position_size = risk_usd / distance if distance > 0 else np.nan
        commission = (position_size * entry_price * 0.000325) if position_size and not np.isnan(position_size) else 0.0
        delta_p = commission / position_size if position_size and position_size != 0 else 0.0

        be_stop = None
        be_activation = None
        if be_multiplier > 0:
            be_activation = entry_price + distance * be_multiplier if direction == 'long' else entry_price - distance * be_multiplier
            be_stop = entry_price + delta_p if direction == 'long' else entry_price - delta_p

        mitigated = bool(setup.get("mitigated", False))
        mit_open  = pd.to_datetime(
            setup.get("mit_bar_open_m15") or setup.get("mit_ts_open") or setup.get("mit_cutoff") or setup.get("cutoff"),
            errors="coerce"
        )
        mit_close = pd.to_datetime(
            setup.get("mit_ts_close_m15") or setup.get("mit_ts_raw"),
            errors="coerce"
        )
        mit_policy = (setup.get("mit_policy") or "cutoff").lower()

        # інфер OPEN із CLOSE, якщо треба (сумісність зі старими/кривими сетапами)
        mit_open = _infer_mit_open_if_needed(mit_open, mit_close, m15_index_is_close=True)
        try:
            if pd.notna(mit_open):  mit_open  = mit_open.tz_localize(None)
            if pd.notna(mit_close): mit_close = mit_close.tz_localize(None)
        except: pass

        # --- формуємо вікно пошуку ---
        if mitigated and mit_policy in ("cutoff","open") and pd.notna(mit_open):
            hard_end = min(search_end, mit_open)
        else:
            hard_end = search_end

        window = m5_df[(m5_df.index >= search_start) & (m5_df.index < hard_end)]
        if window.empty:
            continue

        if entry_timeout_candles is not None:
            window = window.iloc[: entry_timeout_candles + 1]
            if window.empty:
                continue

        # --- шукаємо entry_time ---
        is_long = (direction == 'long')
        arr_low  = window['low' ].to_numpy()
        arr_high = window['high'].to_numpy()
        mask = (arr_low <= entry_price + EPS) if is_long else (arr_high >= entry_price - EPS)
        if not mask.any():
            continue

        idx_entry  = int(np.argmax(mask))
        entry_time = pd.Timestamp(window.index.to_numpy()[idx_entry])

        if mitigated and pd.notna(mit_open):
            # переконаємось, що mit_open без тайзони
            try:
                mit_open_n = pd.to_datetime(mit_open).tz_localize(None)
            except Exception:
                mit_open_n = pd.to_datetime(mit_open, errors="coerce")
            # перевіряємо чи була хоча б якась торкання entry ДО мит-бару (не включає мит-бар)
            pre_mit_start = search_start
            pre_mit_end = mit_open_n - pd.Timedelta(minutes=0)  # exclude mit_open itself

            touched_before_mit = False

            # точніша перевірка на M1, якщо є
            if (m1_df is not None) and (not m1_df.empty):
                try:
                    m1_slice_pre_mit = m1_df.loc[pre_mit_start: pre_mit_end]
                    if not m1_slice_pre_mit.empty:
                        if direction == 'long':
                            touched_before_mit = (m1_slice_pre_mit['low'] <= entry_price + EPS).any()
                        else:
                            touched_before_mit = (m1_slice_pre_mit['high'] >= entry_price - EPS).any()
                except Exception:
                    # у разі проблем зі зрізом — ігноруємо і продовжимо з M5
                    touched_before_mit = False

            # fallback на M5, якщо M1 не показав торкання
            if (not touched_before_mit) and (m5_df is not None) and (not m5_df.empty):
                try:
                    m5_slice_pre_mit = m5_df.loc[pre_mit_start: pre_mit_end]
                    if not m5_slice_pre_mit.empty:
                        if direction == 'long':
                            touched_before_mit = (m5_slice_pre_mit['low'] <= entry_price + EPS).any()
                        else:
                            touched_before_mit = (m5_slice_pre_mit['high'] >= entry_price - EPS).any()
                except Exception:
                    touched_before_mit = False

            if not touched_before_mit:
                print("MIT-KILL: mitigated but entry NOT touched before mitigation -> skip", setup.get("setup_id"), "| mit_open", mit_open_n, "| entry_price", entry_price)
                continue

        # законний assert
        assert search_start <= entry_time < hard_end

        # # --- страхувальна заборона входу після мітигації ---
        # print("MIT-DEBUG",
        #     "id", setup.get("setup_id"),
        #     "| s_start", search_start,
        #     "| s_end", search_end,
        #     "| mit_open", mit_open,
        #     "| policy", mit_policy)

        # # після обчислення entry_time:
        # if mitigated:
        #     if mit_policy in ("cutoff","open") and pd.notna(mit_open) and entry_time >= mit_open:
        #         print("MIT-KILL cutoff", setup.get("setup_id"),
        #             "entry", entry_time, ">= open", mit_open)
        #         continue
        #     if mit_policy in ("strict","close") and pd.notna(mit_close) and entry_time >= mit_close:
        #         print("MIT-KILL strict", setup.get("setup_id"),
        #             "entry", entry_time, ">= close", mit_close)
        #         continue

        # --- take-before-entry ---
        limit_ts  = pd.to_datetime(setup.get('limit_placed_time'), errors='coerce')
        limit_ts  = limit_ts if pd.notna(limit_ts) else search_start
        pre_start = min(max(search_start, limit_ts), entry_time)

        if take_hit_before_entry(pre_start, entry_time, entry_price, take_price, direction, m1_df, m5_df, EPS):
            continue

        # перша M5 після входу — розв'язуємо на M1
        # (вікно рівно 5 хвилин, включно з хвилиною entry_time)
        m1_slice = None
        resolved_on_m1 = False
        if (m1_df is not None) and (not m1_df.empty):
            # узгоджуємо tz, щоб зрізи не ламались
            et = pd.to_datetime(entry_time, errors="coerce")
            if pd.notna(et):
                try:
                    et = et.tz_localize(None)
                except Exception:
                    pass

                m1_slice = m1_df.loc[et : et + pd.Timedelta(minutes=4)]
                if not m1_slice.empty:
                    for ts, row1 in m1_slice.sort_index().iterrows():
                        if direction == "long":
                            # STOP → TAKE
                            if row1["low"]  <= initial_stop + EPS:
                                finish(direction, setup, 'stop', ts, initial_stop, entry_time, entry_price,
                                    initial_stop, take_price, be_stop, search_start, limit_ts, f1p, f2p)
                                resolved_on_m1 = True
                                break
                            if row1["high"] >= take_price  - EPS:
                                finish(direction, setup, 'take', ts, take_price, entry_time, entry_price,
                                    initial_stop, take_price, be_stop, search_start, limit_ts, f1p, f2p)
                                resolved_on_m1 = True
                                break
                        else:
                            # short: STOP → TAKE
                            if row1["high"] >= initial_stop - EPS:
                                finish(direction, setup, 'stop', ts, initial_stop, entry_time, entry_price,
                                    initial_stop, take_price, be_stop, search_start, limit_ts, f1p, f2p)
                                resolved_on_m1 = True
                                break
                            if row1["low"]  <= take_price  + EPS:
                                finish(direction, setup, 'take', ts, take_price, entry_time, entry_price,
                                    initial_stop, take_price, be_stop, search_start, limit_ts, f1p, f2p)
                                resolved_on_m1 = True
                                break

        if resolved_on_m1:
            # уже зафіксували результат на М1 у першій M5 → далі не перевіряємо M5-консолідацію
            continue

        # якщо M1 не визначив вихід у першій M5 — перевіряємо саму M5-консолідовано
        post = m5_df.loc[entry_time:m5_df.index.max()]
        if post.empty:
            continue
        entry_row = post.iloc[0]
        is_long = (direction == 'long')

        # Конфлікт у першій M5: STOP має пріоритет
        hit_take_0 = (entry_row['high'] >= take_price - EPS) if is_long else (entry_row['low']  <= take_price + EPS)
        hit_stop_0 = (entry_row['low']  <= initial_stop + EPS) if is_long else (entry_row['high'] >= initial_stop - EPS)
        if hit_stop_0:
            finish(direction, setup, 'stop', entry_time, initial_stop, entry_time, entry_price,
                   initial_stop, take_price, be_stop, search_start, limit_ts, f1p, f2p)
            continue
        if hit_take_0:
            finish(direction, setup, 'take', entry_time, take_price, entry_time, entry_price,
                   initial_stop, take_price, be_stop, search_start, limit_ts, f1p, f2p)
            continue

        # BE активується ТІЛЬКИ якщо бар закрився за рівнем be_activation; діє з наступного бару
        be_active_from_next_bar = False
        if be_activation is not None:
            be_hit_close = (entry_row['close'] >= be_activation - EPS) if is_long else (entry_row['close'] <= be_activation + EPS)
            if be_hit_close:
                be_active_from_next_bar = True

        # подальші бари
        lows_p, highs_p, closes_p, times_p = (
            post['low'].to_numpy(), post['high'].to_numpy(),
            post['close'].to_numpy(), post.index.to_numpy()
        )
        if len(times_p) <= 1:
            continue

        for i in range(1, len(times_p)):
            current_stop = be_stop if be_active_from_next_bar and (be_stop is not None) else initial_stop
            low_i, high_i, close_i, t_i = lows_p[i], highs_p[i], closes_p[i], times_p[i]

            hit_stop_i = (low_i  <= current_stop + EPS) if is_long else (high_i >= current_stop - EPS)
            hit_take_i = (high_i >= take_price  - EPS) if is_long else (low_i  <= take_price + EPS)

            if hit_stop_i:
                finish(direction, setup, 'be' if (be_active_from_next_bar and be_stop is not None and current_stop == be_stop) else 'stop',
                       pd.Timestamp(t_i), current_stop, entry_time, entry_price,
                       initial_stop, take_price, be_stop, search_start, limit_ts, f1p, f2p)
                break
            if hit_take_i:
                finish(direction, setup, 'take', pd.Timestamp(t_i), take_price, entry_time, entry_price,
                       initial_stop, take_price, be_stop, search_start, limit_ts, f1p, f2p)
                break

            # BE може активуватись на ЗАКРИТТІ цього бару і почати діяти з НАСТУПНОГО
            if (not be_active_from_next_bar) and (be_activation is not None):
                be_hit_close = (close_i >= be_activation - EPS) if is_long else (close_i <= be_activation + EPS)
                if be_hit_close:
                    be_active_from_next_bar = True

    return results  
