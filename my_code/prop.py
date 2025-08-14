#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
from datetime import timedelta

import numpy as np
import pandas as pd

# ==============================
# Configuration
# ==============================
COMMISSION_RATE = 0.000325

RESULTS_DIR = "/Users/synyshyn_04/BTC_backtest_new/backtest_results"
# CSV_FILE = "/Users/synyshyn_04/BTC_backtest_new/backtest_results/0R_NOH1_f0.5_s0.0_r3.5_n1.csv"

INITIAL_BALANCE = 5000
# 0.5%, 0.75%, 1.0%, 1.5%
RISK_PERCENTS = [0.005, 0.0075, 0.01, 0.015]

EXCLUDE_WEEKENDS = True
HOURLY_STATS_RISK = 0.01

PHASES = [
    {"target_pct": 0.08, "max_dd_pct": 0.10, "daily_dd_pct": 0.05},  # Phase 1
    {"target_pct": 0.05, "max_dd_pct": 0.10, "daily_dd_pct": 0.05},  # Phase 2
]

EXCLUDE_PREFIXES = ("summary", "top3", "consensus")
MIN_TRADES = 200


# ==============================
# Helpers
# ==============================
def find_time_column(df: pd.DataFrame) -> str | None:
    for col in ("entry_time", "entry_time_m5", "entry_time_m1"):
        if col in df.columns:
            return col
    return None


def ensure_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def filter_weekdays(df: pd.DataFrame) -> pd.DataFrame:
    if not EXCLUDE_WEEKENDS or df.empty:
        return df
    tcol = find_time_column(df)
    if not tcol:
        return df
    out = df.copy()
    out[tcol] = pd.to_datetime(out[tcol], errors="coerce")
    return out.loc[out[tcol].dt.weekday < 5]


def compute_trade_pnls(df: pd.DataFrame, balance: float, risk_pct: float) -> pd.DataFrame:
    df = filter_weekdays(df.copy())
    req_cols = ["entry_price", "exit_price", "stop", "direction"]
    df = df.dropna(subset=req_cols)
    df = ensure_numeric(df, ["entry_price", "exit_price", "stop"])

    dist = (df["entry_price"] - df["stop"]).abs()
    valid = dist > 0
    risk_usd = balance * risk_pct
    df["position_size"] = np.where(valid, risk_usd / dist, np.nan)

    is_long = df["direction"].astype(str).str.lower().eq("long")
    is_short = df["direction"].astype(str).str.lower().eq("short")

    df["pnl"] = np.nan
    df.loc[is_long, "pnl"] = (df.loc[is_long, "exit_price"] - df.loc[is_long, "entry_price"]) * df.loc[
        is_long, "position_size"
    ]
    df.loc[is_short, "pnl"] = (df.loc[is_short, "entry_price"] - df.loc[is_short, "exit_price"]) * df.loc[
        is_short, "position_size"
    ]

    commission_cost = df["position_size"] * df["entry_price"] * COMMISSION_RATE
    df["pnl"] = df["pnl"] - commission_cost

    df = df.dropna(subset=["pnl"])
    return df


def summarize_pnls(df: pd.DataFrame) -> dict:
    wins = df.loc[df["pnl"] > 0, "pnl"]
    losses = df.loc[df["pnl"] < 0, "pnl"]
    return {
        "avg_win_usd": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss_usd": float(losses.mean()) if len(losses) else 0.0,
        "win_rate": (len(wins) / len(df) * 100.0) if len(df) else 0.0,
    }


def compute_overall_metrics(df_pnl: pd.DataFrame, initial_balance: float) -> dict:
    tcol = find_time_column(df_pnl)
    if tcol:
        df_pnl = df_pnl.copy()
        df_pnl[tcol] = pd.to_datetime(df_pnl[tcol], errors="coerce")
        df_pnl = df_pnl.sort_values(tcol)

    equity = initial_balance + df_pnl["pnl"].cumsum()
    highwater = equity.cummax()
    drawdown_frac = (equity - highwater) / highwater
    max_dd_pct = float(drawdown_frac.min() * 100.0) if len(drawdown_frac) else 0.0

    net_return_pct = float(df_pnl["pnl"].sum() / initial_balance * 100.0) if len(df_pnl) else 0.0
    win_rate = float((df_pnl["pnl"] > 0).mean() * 100.0) if len(df_pnl) else 0.0
    avg_win_pct = float(df_pnl.loc[df_pnl["pnl"] > 0, "pnl"].mean() / initial_balance * 100.0) if (df_pnl["pnl"] > 0).any() else 0.0
    avg_loss_pct = float(df_pnl.loc[df_pnl["pnl"] < 0, "pnl"].mean() / initial_balance * 100.0) if (df_pnl["pnl"] < 0).any() else 0.0

    gross_profit = float(df_pnl.loc[df_pnl["pnl"] > 0, "pnl"].sum())
    gross_loss = float(df_pnl.loc[df_pnl["pnl"] < 0, "pnl"].sum())
    pf = (gross_profit / abs(gross_loss)) if gross_loss != 0 else np.inf

    rd = (net_return_pct / -max_dd_pct) if max_dd_pct < 0 else np.nan
    expectancy_pct = (win_rate / 100.0) * avg_win_pct + (1.0 - win_rate / 100.0) * avg_loss_pct

    return {
        "Net Return %": net_return_pct,
        "Max Drawdown %": max_dd_pct,
        "Return/Drawdown": rd,
        "Win Rate %": win_rate,
        "Avg Win %": avg_win_pct,
        "Avg Loss %": avg_loss_pct,
        "Profit Factor": pf,
        "Expectancy %": expectancy_pct,
        "Trades Count": int(len(df_pnl)),
    }


def simulate_phase(
    df: pd.DataFrame,
    starting_balance: float,
    risk_pct: float,
    target_pct: float,
    max_dd_pct: float,
    daily_dd_pct: float = 0.05,
) -> tuple[bool, float, int, str]:
    if "pnl" not in df.columns:
        raise ValueError("simulate_phase expects df with a 'pnl' column (use compute_trade_pnls first).")

    tcol = find_time_column(df)
    if tcol:
        df = df.copy()
        df[tcol] = pd.to_datetime(df[tcol], errors="coerce")
        df = df.sort_values(tcol)

    balance = starting_balance
    peak = balance
    current_day = None
    day_peak = balance

    for i, row in enumerate(df.itertuples(index=False), start=1):
        if tcol:
            ts = getattr(row, tcol)
            day = pd.Timestamp(ts).date() if pd.notna(ts) else current_day
        else:
            day = current_day

        if current_day is None or (day is not None and day != current_day):
            current_day = day
            day_peak = balance

        pnl = row.pnl
        if not np.isfinite(pnl):
            continue

        balance += pnl
        peak = max(peak, balance)
        day_peak = max(day_peak, balance)

        total_dd = (peak - balance) / peak if peak > 0 else 0.0
        if total_dd > max_dd_pct:
            return False, balance, i, "max_dd"

        daily_dd = (day_peak - balance) / day_peak if day_peak > 0 else 0.0
        if daily_dd > daily_dd_pct:
            return False, balance, i, "daily_dd"

        if balance >= starting_balance * (1.0 + target_pct):
            return True, balance, i, "target"

    return False, balance, len(df), "exhausted"


def out_of_sample_test(df: pd.DataFrame, risk_pct: float = 0.01, last_days: int = 365) -> None:
    tcol = find_time_column(df)
    if not tcol or df.empty:
        print("\n=== Out-of-Sample (last year) Metrics ===")
        print("No time column or empty DF. Skipping OOS.")
        return

    df = df.copy()
    df[tcol] = pd.to_datetime(df[tcol], errors="coerce")
    cutoff = df[tcol].max() - timedelta(days=last_days)
    df_oos = df.loc[df[tcol] >= cutoff].copy()

    if df_oos.empty:
        print("\n=== Out-of-Sample (last year) Metrics ===")
        print("No Out-of-Sample data (last year).")
        return

    df_oos = filter_weekdays(df_oos)
    df_oos = compute_trade_pnls(df_oos, INITIAL_BALANCE, risk_pct)
    metrics = compute_overall_metrics(df_oos, INITIAL_BALANCE)

    print("\n=== Out-of-Sample (last year) Metrics ===")
    print(f"Trades: {len(df_oos)}, Net Return %: {metrics['Net Return %']:.2f}, Win Rate %: {metrics['Win Rate %']:.2f}")


def stress_test_usd(df: pd.DataFrame, initial_balance: float, risk_pct: float = 0.01, drop_frac: float = 0.05) -> None:
    dfp = compute_trade_pnls(filter_weekdays(df.copy()), initial_balance, risk_pct)
    if dfp.empty:
        print("\n=== Stress Test (drop top {:.0f}%) Expectancy %: N/A".format(drop_frac * 100))
        return

    cutoff = dfp["pnl"].quantile(1 - drop_frac)
    stripped = dfp.loc[dfp["pnl"] < cutoff]
    if stripped.empty:
        print("\n=== Stress Test (drop top {:.0f}%) Expectancy %: N/A".format(drop_frac * 100))
        return

    wins = stripped.loc[stripped["pnl"] > 0, "pnl"]
    losses = stripped.loc[stripped["pnl"] < 0, "pnl"]
    w = (len(wins) / len(stripped)) if len(stripped) else 0.0
    aw = float(wins.mean()) if len(wins) else 0.0
    al = float(losses.mean()) if len(losses) else 0.0
    exp_pct = (w * aw + (1 - w) * al) / initial_balance * 100.0

    print("\n=== Stress Test (drop top {:.0f}%) Expectancy %: {:.2f}".format(drop_frac * 100, exp_pct))


# ==============================
# Hourly stats & helpers
# ==============================
def hourly_stats(df: pd.DataFrame, balance: float, risk_pct: float) -> pd.DataFrame:
    df = filter_weekdays(df)
    tcol = find_time_column(df)
    if not tcol or df.empty:
        return pd.DataFrame()

    df = df.copy()
    df[tcol] = pd.to_datetime(df[tcol], errors="coerce")
    df = df.dropna(subset=["exit_price"])
    dfp = compute_trade_pnls(df, balance, risk_pct)
    if dfp.empty:
        return pd.DataFrame()

    dfp["hour"] = pd.to_datetime(dfp[tcol], errors="coerce").dt.hour

    def _agg(g: pd.DataFrame) -> pd.Series:
        trades = len(g)
        wins = int((g["pnl"] > 0).sum())
        losses = int((g["pnl"] < 0).sum())
        gross_profit = float(g.loc[g["pnl"] > 0, "pnl"].sum())
        gross_loss = float(g.loc[g["pnl"] < 0, "pnl"].sum())
        pf = (gross_profit / abs(gross_loss)) if gross_loss != 0 else (np.inf if gross_profit > 0 else np.nan)
        avg_win = float(g.loc[g["pnl"] > 0, "pnl"].mean()) if wins else 0.0
        avg_loss = float(g.loc[g["pnl"] < 0, "pnl"].mean()) if losses else 0.0
        win_rate = (wins / trades * 100.0) if trades else 0.0
        net_ret_pct = float(g["pnl"].sum() / balance * 100.0) if trades else 0.0
        return pd.Series({
            "Trades": trades,
            "Win Rate %": win_rate,
            "Profit Factor": pf,
            "Avg Win $": avg_win,
            "Avg Loss $": avg_loss,
            "Net Return %": net_ret_pct,
        })

    tbl = dfp.groupby("hour", as_index=False).apply(_agg).reset_index(drop=True)

    all_hours = pd.DataFrame({"hour": np.arange(24, dtype=int)})
    tbl = all_hours.merge(tbl, on="hour", how="left").sort_values("hour")

    for c in ["Trades", "Win Rate %", "Avg Win $", "Avg Loss $", "Net Return %"]:
        if c in tbl.columns:
            tbl[c] = tbl[c].fillna(0.0)

    for c in ["Win Rate %", "Profit Factor", "Avg Win $", "Avg Loss $", "Net Return %"]:
        if c in tbl.columns:
            tbl[c] = pd.to_numeric(tbl[c], errors="coerce").round(3)

    return tbl


def best_hours_for_file(file_path: str, min_trades: int = 20, top_k: int = 5) -> pd.DataFrame:
    try:
        df_src = pd.read_csv(file_path).dropna(subset=["exit_price"])
    except Exception as e:
        print(f"Failed to read {os.path.basename(file_path)}: {e}")
        return pd.DataFrame()

    df_src = filter_weekdays(df_src)
    tbl = hourly_stats(df_src, INITIAL_BALANCE, HOURLY_STATS_RISK)
    if tbl.empty:
        return tbl

    good = tbl[(tbl["Trades"] >= min_trades) & (tbl["Profit Factor"] > 1.0) & (tbl["Net Return %"] > 0)]
    if good.empty:
        good = tbl.copy()

    best = good.sort_values(
        by=["Profit Factor", "Net Return %", "Trades"],
        ascending=[False, False, False]
    ).head(top_k)

    return best.reset_index(drop=True)


def filter_to_hours(df: pd.DataFrame, allowed_hours: list[int]) -> pd.DataFrame:
    tcol = find_time_column(df)
    if not tcol or df.empty:
        return df

    out = df.copy()
    out[tcol] = pd.to_datetime(out[tcol], errors="coerce")
    out = out.dropna(subset=[tcol])
    out["hour"] = out[tcol].dt.hour
    out = out.loc[out["hour"].isin(allowed_hours)].drop(columns=["hour"])
    return out


# ==============================
# Оцінка файлів та ранж
# ==============================
def evaluate_file(file_path: str) -> dict | None:
    try:
        hdr = pd.read_csv(file_path, nrows=0).columns
    except Exception as e:
        print(f"Failed to read header {file_path}: {e}")
        return None

    needed = {"entry_price", "exit_price", "stop", "direction"}
    if not needed.issubset(set(hdr)):
        base = os.path.basename(file_path).lower()
        if not base.startswith(EXCLUDE_PREFIXES):
            print(f"Skip {os.path.basename(file_path)}: missing columns {needed - set(hdr)}")
        return None

    time_cols = [c for c in ("entry_time", "entry_time_m5", "entry_time_m1") if c in hdr]
    usecols = list(needed) + time_cols
    try:
        df_raw = pd.read_csv(file_path, usecols=usecols)
    except Exception as e:
        print(f"Failed to read {file_path} (usecols): {e}")
        return None

    df_raw = df_raw.dropna(subset=["exit_price"])  # closed trades only
    if df_raw.empty:
        print(f"Skip {os.path.basename(file_path)}: no closed trades")
        return None

    tcol = find_time_column(df_raw)
    if tcol:
        df_raw[tcol] = pd.to_datetime(df_raw[tcol], errors="coerce")
        if EXCLUDE_WEEKENDS:
            df_raw = df_raw.loc[df_raw[tcol].dt.weekday < 5]
        df_raw = df_raw.sort_values(tcol)

    per_risk_rows = []
    phase_pass_at_1pct = False

    for rp in RISK_PERCENTS:
        df_pnl = compute_trade_pnls(df_raw.copy(), INITIAL_BALANCE, rp)
        if df_pnl.empty:
            continue

        metrics = compute_overall_metrics(df_pnl, INITIAL_BALANCE)

        pass_all = True
        for ph in PHASES:
            ok, _, _, _ = simulate_phase(
                df_pnl[["pnl", tcol]] if tcol else df_pnl[["pnl"]],
                INITIAL_BALANCE,
                rp,
                ph["target_pct"],
                ph["max_dd_pct"],
                daily_dd_pct=ph["daily_dd_pct"],
            )
            if not ok:
                pass_all = False
                break

        if rp == 0.01:
            phase_pass_at_1pct = pass_all

        per_risk_rows.append(
            {
                "file": os.path.basename(file_path),
                "risk_pct": rp,
                "Profit Factor": metrics["Profit Factor"],
                "Return/Drawdown": metrics["Return/Drawdown"],
                "Net Return %": metrics["Net Return %"],
                "Max Drawdown %": metrics["Max Drawdown %"],
                "Trades Count": metrics["Trades Count"],
                "phase_pass": pass_all,
            }
        )

    if not per_risk_rows:
        return None

    tmp = pd.DataFrame(per_risk_rows).replace([np.inf, -np.inf], np.nan)
    tmp = tmp.dropna(subset=["Profit Factor", "Return/Drawdown"])
    if tmp.empty:
        return None

    return {"file": os.path.basename(file_path), "per_risk": tmp, "phase_pass_at_1pct": phase_pass_at_1pct}


def rank_across_files(
    summaries: list[dict],
    min_trades: int = MIN_TRADES,
    require_two_risks: bool = False,
    require_phase1pct: bool = False,
) -> pd.DataFrame:
    long_rows = [s["per_risk"] for s in summaries]
    long_df = pd.concat(long_rows, ignore_index=True)

    # Base gates (no phase by default)
    gated = long_df[
        (long_df["Profit Factor"] > 1.0) &
        (long_df["Return/Drawdown"] > 0.0) &
        (long_df["Trades Count"] >= min_trades)
    ]
    if gated.empty:
        print("\nNo strategies meet PF/RD/min_trades. Falling back to raw ranks.\n")
        gated = long_df.copy()

    # Optional: require phase pass @1% at file level
    if require_phase1pct:
        pass1pct_map = {s["file"]: s.get("phase_pass_at_1pct", False) for s in summaries}
        gated = gated[gated["file"].map(pass1pct_map).fillna(False)]
        if gated.empty:
            print("\nAfter phase@1% gate, nothing left. Disabling this gate.\n")
            gated = long_df.copy()

    # Optional: require >=2 distinct risk levels per file
    if require_two_risks:
        valid_counts = gated.groupby("file")["risk_pct"].nunique()
        keep_files = valid_counts[valid_counts >= 2].index
        if len(keep_files) == 0:
            print("\nAfter '>=2 risks' gate, nothing left. Disabling this gate.\n")
        else:
            gated = gated[gated["file"].isin(keep_files)]

    # Per-risk ranks and consensus rank
    gated = gated.copy()
    gated["pf_rank"] = gated.groupby("risk_pct")["Profit Factor"].rank(ascending=False, method="average")
    gated["rd_rank"] = gated.groupby("risk_pct")["Return/Drawdown"].rank(ascending=False, method="average")
    gated["half_cons"] = (gated["pf_rank"] + gated["rd_rank"]) / 2.0

    cons = gated.groupby("file", as_index=False)["half_cons"].mean().rename(columns={"half_cons": "cons_rank"})

    snap_1 = gated.loc[gated["risk_pct"] == 0.01,
                       ["file", "Profit Factor", "Return/Drawdown", "Net Return %", "Max Drawdown %", "Trades Count"]]
    out = cons.merge(snap_1, on="file", how="left").sort_values("cons_rank", ascending=True)

    pass1pct_map = {s["file"]: s.get("phase_pass_at_1pct", False) for s in summaries}
    if not out.empty:
        out["phase_pass_1pct"] = out["file"].map(pass1pct_map).fillna(False)

    def parse_params(fname: str) -> dict:
        base = os.path.splitext(fname)[0]
        parts = base.split("_")
        params = {"filename": fname}
        try:
            params["be_multiplier"] = parts[0].replace("R", "")
            params["h1_filter"] = "H1F" in parts[1]
            for p in parts[2:]:
                if p.startswith("f"):
                    params["fib_level"] = float(p[1:])
                elif p.startswith("s"):
                    params["stop_offset"] = float(p[1:])
                elif p.startswith("r"):
                    params["rr"] = float(p[1:])
                elif p.startswith("n"):
                    params["max_trades"] = int(float(p[1:]))
        except Exception:
            pass
        return params

    if not out.empty:
        parsed = pd.DataFrame([parse_params(f) for f in out["file"]])
        out = pd.concat([out.reset_index(drop=True), parsed.reset_index(drop=True)], axis=1)

    return out


# ==============================
# Risk comparison for prop passing
# ==============================
def risk_pass_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Для кожного ризику: рахуємо PnL, метрики, та симулюємо Phase 1 і Phase 2.
    Повертаємо таблицю:
      risk_%, pass_p1, trades_p1, reason_p1, pass_p2, trades_p2, reason_p2,
      pass_both, total_trades_if_pass, PF, R/D, NetRet%, Trades.
    """
    rows = []
    tcol = find_time_column(df)
    for rp in RISK_PERCENTS:
        dfp = compute_trade_pnls(df.copy(), INITIAL_BALANCE, rp)
        if dfp.empty:
            rows.append({
                "risk_%": rp * 100,
                "pass_p1": False, "trades_p1": 0, "reason_p1": "no_trades",
                "pass_p2": False, "trades_p2": 0, "reason_p2": "no_trades",
                "pass_both": False, "total_trades": np.nan,
                "Profit Factor": np.nan, "Return/Drawdown": np.nan,
                "Net Return %": np.nan, "Trades Count": 0
            })
            continue

        metrics = compute_overall_metrics(dfp, INITIAL_BALANCE)

        ok1, bal1, used1, r1 = simulate_phase(
            dfp[["pnl", tcol]] if tcol else dfp[["pnl"]],
            INITIAL_BALANCE, rp, PHASES[0]["target_pct"], PHASES[0]["max_dd_pct"], PHASES[0]["daily_dd_pct"]
        )
        ok2, bal2, used2, r2 = simulate_phase(
            dfp[["pnl", tcol]] if tcol else dfp[["pnl"]],
            INITIAL_BALANCE, rp, PHASES[1]["target_pct"], PHASES[1]["max_dd_pct"], PHASES[1]["daily_dd_pct"]
        )

        rows.append({
            "risk_%": rp * 100,
            "pass_p1": ok1, "trades_p1": used1, "reason_p1": r1,
            "pass_p2": ok2, "trades_p2": used2, "reason_p2": r2,
            "pass_both": (ok1 and ok2),
            "total_trades": (used1 + used2) if (ok1 and ok2) else np.nan,
            "Profit Factor": metrics["Profit Factor"],
            "Return/Drawdown": metrics["Return/Drawdown"],
            "Net Return %": metrics["Net Return %"],
            "Trades Count": metrics["Trades Count"],
        })

    tbl = pd.DataFrame(rows)
    # Сортуємо для виводу: спочатку ті, що пройшли обидві фази, далі за total_trades, далі за меншим ризиком
    tbl_sorted = tbl.sort_values(
        by=["pass_both", "total_trades", "risk_%"],
        ascending=[False, True, True],
        kind="mergesort"
    ).reset_index(drop=True)
    return tbl_sorted


def pick_optimal_risk(tbl: pd.DataFrame) -> dict | None:
    """
    Обирає оптимальний ризик:
      — серед pass_both==True обираємо мінімальний total_trades,
      — за рівності — менший risk_%.
      — якщо ніхто не пройшов обидві фази, повертаємо None.
    """
    if tbl.empty:
        return None
    ok = tbl.loc[tbl["pass_both"] == True].copy()
    if ok.empty:
        return None
    best = ok.sort_values(by=["total_trades", "risk_%"], ascending=[True, True]).iloc[0].to_dict()
    return best


# ==============================
# Recompute only with best hours
# ==============================
def recompute_with_hours(file_path: str,
                         allowed_hours: list[int] | None = None,
                         oos_cutoff: str | pd.Timestamp | None = None,
                         oos_last_days: int | None = None,
                         learn_hours_on_train: bool = False,
                         top_k: int = 5,
                         min_trades_hour: int = 20) -> None:
    """Перерахунок метрик по файлу з опційним OOS-розбиттям і фільтром годин.
    Логіка:
      1) Читаємо та нормалізуємо час.
      2) Розбиваємо на train/OOS (якщо задано oos_cutoff або oos_last_days).
      3) Якщо learn_hours_on_train=True і години не задані — вчимо години на train.
      4) Фільтруємо тільки OOS на дозволені години.
      5) Усі метрики, фази, таблиці — по OOS.
    """
    try:
        df = pd.read_csv(file_path).dropna(subset=["exit_price"])  # працюємо лише з закритими угодами
    except Exception as e:
        print(f"Failed to read {os.path.basename(file_path)}: {e}")
        return

    # Будні (за потреби) і нормалізація часу
    df = filter_weekdays(df)
    tcol = find_time_column(df)
    if not tcol or df.empty:
        print("Порожньо: немає часової колонки або угод.")
        return

    df = df.copy()
    df[tcol] = pd.to_datetime(df[tcol], errors="coerce")
    df = df.dropna(subset=[tcol]).sort_values(tcol)

    # --- Train/OOS split ---
    if oos_cutoff is not None:
        cutoff = pd.Timestamp(oos_cutoff)
    elif oos_last_days is not None:
        cutoff = df[tcol].max() - timedelta(days=oos_last_days)
    else:
        cutoff = None

    if cutoff is not None:
        df_train = df.loc[df[tcol] < cutoff].copy()
        df_oos   = df.loc[df[tcol] >= cutoff].copy()
        print(f"OOS cutoff: {cutoff} | train={len(df_train)}, oos={len(df_oos)}")
    else:
        df_train = df.iloc[0:0].copy()
        df_oos   = df.copy()
        print("OOS cutoff: <none> | train=0, oos={}".format(len(df_oos)))

    # --- (Опційно) навчання годин на train ---
    if learn_hours_on_train and not allowed_hours:
        if df_train.empty:
            print("Попередження: train-період порожній, години навчити неможливо — використаємо всі години.")
            learned_hours: list[int] | None = None
        else:
            ht = hourly_stats(df_train, INITIAL_BALANCE, HOURLY_STATS_RISK)
            good = ht[(ht["Trades"] >= min_trades_hour) & (ht["Profit Factor"] > 1.0) & (ht["Net Return %"] > 0)]
            if good.empty:
                good = ht
            learned_hours = (
                good.sort_values(["Profit Factor", "Net Return %", "Trades"], ascending=[False, False, False])
                    .head(top_k)["hour"].astype(int).tolist()
            )
        allowed_hours = learned_hours
        print(f"Навчені години (train): {allowed_hours if allowed_hours else '[усі]'}")

    # --- Застосовуємо фільтр годин до OOS ---
    if allowed_hours:
        df_oos = filter_to_hours(df_oos, allowed_hours)

    print(f"\n=== Перерахунок тільки для годин {allowed_hours if allowed_hours else '[усі]'} ===")
    if df_oos.empty:
        print("Порожньо: у вибраних годинах немає закритих угод.")
        return

    # --- PnL summary by risk (по OOS) ---
    rows = []
    for rp in RISK_PERCENTS:
        df_rp = compute_trade_pnls(df_oos.copy(), INITIAL_BALANCE, rp)
        if df_rp.empty:
            continue
        s = summarize_pnls(df_rp)
        s["risk_pct"] = rp
        rows.append(s)
    if rows:
        stats_df = pd.DataFrame(rows).set_index("risk_pct")
        print("\n=== PnL Summary by Risk % (filtered hours, OOS) ===")
        print(stats_df)

    # --- Overall @1% + проп-фази @1% (по OOS) ---
    df_1 = compute_trade_pnls(df_oos.copy(), INITIAL_BALANCE, 0.01)
    if df_1.empty:
        print("\nNo trades @1% risk after hour filter (OOS).")
        return

    overall = compute_overall_metrics(df_1, INITIAL_BALANCE)
    print("\n=== Overall Performance Metrics @1% (filtered hours, OOS) ===")
    for k, v in overall.items():
        print(f"{k}: {v}")

    print("\n=== Phase Simulation Results @1% (filtered hours, OOS) ===")
    tcol_oos = find_time_column(df_oos)
    for ph in PHASES:
        ok, bal, used, reason = simulate_phase(
            df_1[["pnl", tcol_oos]] if tcol_oos else df_1[["pnl"]],
            INITIAL_BALANCE,
            0.01,
            ph["target_pct"],
            ph["max_dd_pct"],
            daily_dd_pct=ph["daily_dd_pct"],
        )
        print(f"target={ph['target_pct']*100:.0f}%, success={ok}, final_balance={bal:.2f}, trades_used={used}, reason={reason}")

    # === Порівняння ризиків для проходження (по OOS) ===
    tbl = risk_pass_table(df_oos)
    if tbl.empty:
        print("\n(Неможливо порівняти ризики — таблиця порожня по OOS.)")
        return

    print("\n=== Порівняння ризиків для проходження (Phase 1 & Phase 2) — OOS ===")
    pretty = tbl.copy()
    num_cols = ["risk_%", "trades_p1", "trades_p2", "total_trades", "Profit Factor", "Return/Drawdown", "Net Return %", "Trades Count"]
    for c in num_cols:
        if c in pretty.columns:
            pretty[c] = pd.to_numeric(pretty[c], errors="coerce").round(3)
    print(pretty.to_string(index=False))

    best = pick_optimal_risk(tbl)
    if best is None:
        print("\nОптимальний ризик не визначено (жоден рівень не пройшов обидві фази в OOS).")
    else:
        print("\n=== Найоптимальніший ризик для проходження (OOS) ===")
        print(
            f"risk = {best['risk_%']:.2f}% | pass_both = True | total_trades ≈ {int(best['total_trades'])} "
            f"| trades_p1 = {int(best['trades_p1'])}, trades_p2 = {int(best['trades_p2'])}"
        )

# ==============================
# CLI / main
# ==============================
def analyze_single_file(csv_path: str) -> None:
    print(f"\nAnalyzing single file: {os.path.basename(csv_path)}")
    df = pd.read_csv(csv_path).dropna(subset=["exit_price"])
    df = filter_weekdays(df)

    rows = []
    for rp in RISK_PERCENTS:
        df_rp = compute_trade_pnls(df.copy(), INITIAL_BALANCE, rp)
        s = summarize_pnls(df_rp)
        s["risk_pct"] = rp
        rows.append(s)
    stats_df = pd.DataFrame(rows).set_index("risk_pct")
    print("\n=== PnL Summary by Risk % ===")
    print(stats_df)

    df_1 = compute_trade_pnls(df.copy(), INITIAL_BALANCE, 0.01)
    overall = compute_overall_metrics(df_1, INITIAL_BALANCE)
    print("\n=== Overall Performance Metrics (1% risk) ===")
    for k, v in overall.items():
        print(f"{k}: {v}")

    print("\n=== Phase Simulation Results (1% risk) ===")
    tcol = find_time_column(df)
    for ph in PHASES:
        ok, bal, used, reason = simulate_phase(
            df_1[["pnl", tcol]] if tcol else df_1[["pnl"]],
            INITIAL_BALANCE,
            0.01,
            ph["target_pct"],
            ph["max_dd_pct"],
            daily_dd_pct=ph["daily_dd_pct"],
        )
        print(f"target={ph['target_pct']*100:.0f}%, success={ok}, final_balance={bal:.2f}, trades_used={used}, reason={reason}")

    out_of_sample_test(df, risk_pct=0.01)
    stress_test_usd(df, INITIAL_BALANCE, risk_pct=0.01, drop_frac=0.05)

    htbl = hourly_stats(df.copy(), INITIAL_BALANCE, HOURLY_STATS_RISK)
    if not htbl.empty:
        print(f"\n=== Hourly Stats (weekdays only) @ risk {HOURLY_STATS_RISK*100:.2f}% ===")
        print(htbl.to_string(index=False, float_format=lambda x: f"{x:.2f}" if isinstance(x, float) else str(x)))

    # Додатково: таблиця порівняння ризиків для всього файлу (без фільтру годин)
    print("\n=== Порівняння ризиків (без фільтра годин) ===")
    risks_full = risk_pass_table(df)
    if not risks_full.empty:
        pretty = risks_full.copy()
        for c in ["risk_%", "trades_p1", "trades_p2", "total_trades"]:
            if c in pretty.columns:
                pretty[c] = pd.to_numeric(pretty[c], errors="coerce").round(3)
        print(pretty.to_string(index=False))
        best_full = pick_optimal_risk(risks_full)
        if best_full:
            print(f"\nНайоптимальніший ризик (без фільтра годин): {best_full['risk_%']:.2f}% "
                  f"(total_trades ≈ {int(best_full['total_trades'])})")


def analyze_folder(results_dir: str, min_trades: int = MIN_TRADES, require_two_risks: bool = False, require_phase1pct: bool = False) -> None:
    # Recursively collect CSVs from nested backtest_results structure, e.g.
    # backtest_results/BE_{be_mult}/{h1_filter|no_h1_filter}/*.csv
    pattern = os.path.join(results_dir, "**", "*.csv")
    csvs_all = sorted(glob.glob(pattern, recursive=True))

    # Exclude summary-like files by basename
    csvs = [fp for fp in csvs_all if not os.path.basename(fp).lower().startswith(EXCLUDE_PREFIXES)]
    if not csvs:
        print(
            f"No CSV files found under {results_dir} (after excluding summaries). "
            "Make sure your files are in nested folders like BE_*/h1_filter or no_h1_filter."
        )
        return

    # Keep a mapping from basename -> full path (filenames are unique due to be/h1 tags)
    base_to_full = {os.path.basename(fp): fp for fp in csvs}

    summaries = []
    total_trades_all = 0
    for fp in csvs:
        s = evaluate_file(fp)
        if s is not None:
            summaries.append(s)
            try:
                hdr = pd.read_csv(fp, nrows=0).columns
                if "exit_price" in hdr:
                    df_tmp = pd.read_csv(fp, usecols=["exit_price"])  # count closed trades only
                    total_trades_all += int(df_tmp["exit_price"].notna().sum())
            except Exception:
                pass

    print(f"\nПісля фільтрації залишилось {total_trades_all} угод для аналізу.")

    if not summaries:
        print("No evaluable files found.")
        return

    ranked = rank_across_files(
        summaries,
        min_trades=min_trades,
        require_two_risks=require_two_risks,
        require_phase1pct=require_phase1pct,
    )
    if ranked.empty:
        print("\nNo ranked strategies after gating. Try relaxing MIN_TRADES or gates.")
        return

    # === Рівно ОДИН найкращий сетап ===
    best = ranked.head(1).copy()
    row = best.iloc[0]
    fname = row["file"]
    fpath = base_to_full.get(fname, os.path.join(results_dir, fname))

    print("\n=== Найкращий сетап для проп (враховано проп-фази @1%) ===")
    cols = [
        "cons_rank", "file", "Profit Factor", "Return/Drawdown",
        "Net Return %", "Max Drawdown %", "Trades Count",
        "be_multiplier", "h1_filter", "fib_level", "stop_offset", "rr", "max_trades"
    ]
    cols = [c for c in cols if c in best.columns]
    print(best[cols].to_string(index=False))

    # === Найкращі години для цього сетапу ===
    hours_tbl = best_hours_for_file(fpath, min_trades=20, top_k=5)
    if hours_tbl.empty:
        print(f"\n(Немає по-годинної статистики для {fname} або даних замало.)")
        # Все одно покажемо оптимальний ризик без фільтра годин:
        try:
            df_src = pd.read_csv(fpath).dropna(subset=["exit_price"])
            df_src = filter_weekdays(df_src)
            risks_full = risk_pass_table(df_src)
            if not risks_full.empty:
                print("\n=== Порівняння ризиків (без фільтра годин) ===")
                print(risks_full.to_string(index=False))
                best_full = pick_optimal_risk(risks_full)
                if best_full:
                    print(
                        f"\nНайоптимальніший ризик: {best_full['risk_%']:.2f}% "
                        f"(total_trades ≈ {int(best_full['total_trades'])})"
                    )
        except Exception as e:
            print(f"Failed risk comparison for {fname}: {e}")
        return

    print(f"\n=== Найкращі години для {fname} (будні) @ risk {HOURLY_STATS_RISK*100:.2f}% ===")
    print(hours_tbl.to_string(index=False))

    # === Перерахунок ВСІЄЇ статистики лише по цих годинах + визначення оптимального ризику ===
    allowed_hours = hours_tbl["hour"].astype(int).tolist()
    recompute_with_hours(fpath, allowed_hours)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Backtest results analyzer with prop-firm constraints (best setup + best hours + optimal risk).")
    parser.add_argument("--dir", default=RESULTS_DIR, help="Directory with CSV results to rank.")
    parser.add_argument("--file", default=None, help="Analyze a single CSV instead.")
    parser.add_argument("--min-trades", type=int, default=MIN_TRADES, help="Мінімальна кількість угод для ранжування (per risk level).")
    parser.add_argument("--require-two-risks", action="store_true", help="Фільтр: залишати лише файли з ≥2 рівнями ризику.")
    parser.add_argument("--require-phase1pct", action="store_true", help="Фільтр: вимагати проходження обох фаз @1% ризику.")
    args = parser.parse_args()

    if args.file:
        analyze_single_file(args.file)
    else:
        analyze_folder(
            args.dir,
            min_trades=args.min_trades,
            require_two_risks=args.require_two_risks,
            require_phase1pct=args.require_phase1pct,
        )


if __name__ == "__main__":

    main()

# ==============================
# Scan & analyze all CSVs in RESULTS_DIR
# ==============================
def list_result_csvs(root: str) -> list[str]:
    """
    Повертає всі CSV у підпапках root, окрім тих, що починаються на EXCLUDE_PREFIXES.
    """
    pattern = os.path.join(root, "**", "*.csv")
    files = glob.glob(pattern, recursive=True)
    out = []
    for fp in files:
        base = os.path.basename(fp).lower()
        if base.startswith(EXCLUDE_PREFIXES):
            continue
        out.append(fp)
    return sorted(out)


def analyze_results_dir(root: str = RESULTS_DIR, save_rank: bool = True) -> None:
    """
    Пробігає всі результати у root, оцінює кожен файл і ранжує стратегії між файлами.
    """
    print(f"\n=== Scanning results in: {root} ===")
    files = list_result_csvs(root)
    if not files:
        print("Немає csv-файлів для аналізу.")
        return

    summaries = []
    for fp in files:
        s = evaluate_file(fp)
        if s is not None:
            summaries.append(s)

    if not summaries:
        print("Жоден файл не пройшов базову валідацію (немає закритих угод або відсутні колонки).")
        return

    ranked = rank_across_files(summaries)
    if ranked.empty:
        print("Ранжування порожнє після фільтрів.")
        return

    # Друк компактної таблиці
    print("\n=== Ranked strategies across files ===")
    cols = ["file", "cons_rank", "Profit Factor", "Return/Drawdown", "Net Return %", "Max Drawdown %", "Trades Count"]
    show = ranked[cols].copy()
    # Округлення для друку
    for c in ["cons_rank", "Profit Factor", "Return/Drawdown", "Net Return %", "Max Drawdown %"]:
        show[c] = pd.to_numeric(show[c], errors="coerce").round(3)
    print(show.to_string(index=False))

    if save_rank:
        out_path = os.path.join(root, "summary_rank.csv")
        ranked.to_csv(out_path, index=False)
        print(f"\n→ Saved ranking to {out_path}")

    # Підказка найкращих годин (опційно: лише для топ-3 файлів у cons_rank)
    try:
        top_files = ranked.nsmallest(3, "cons_rank")["file"].tolist()
        if top_files:
            print("\n=== Best hours suggestions (top-3 files) ===")
            for fname in top_files:
                fp = next((f for f in files if os.path.basename(f) == fname), None)
                if not fp:
                    continue
                best = best_hours_for_file(fp, min_trades=20, top_k=5)
                if best.empty:
                    print(f"{fname}: немає достатньо даних для годин або все слабко.")
                else:
                    print(f"\n{fname}:")
                    print(best.to_string(index=False))
    except Exception as e:
        print(f"\n(Не вдалося порахувати найкращі години: {e})")


