from data_loader import find_fractals
import pandas as pd
import numpy as np

def sweep(df):
    df_fractal = find_fractals(df)

    def calc_directional_sweeps(fractal_df, price_df):
        # Ensure datetime index
        if 'time' in price_df.columns:
            price_df = price_df.set_index('time')
        price_df = price_df.sort_index()

        # Normalize index to tz-naive DatetimeIndex
        price_df.index = pd.to_datetime(price_df.index, errors='coerce')
        if isinstance(price_df.index, pd.DatetimeIndex) and price_df.index.tz is not None:
            price_df.index = price_df.index.tz_convert(None)
        # After this, index is tz-naive (or already was)

        # Drop duplicates in index
        price_df = price_df[~price_df.index.duplicated(keep='first')]

        # Prepare containers
        l2h_time, l2h_price, h2l_time, h2l_price = [], [], [], []
        l2h_events, h2l_events = [], []

        # Convert to numpy arrays for speed
        f_types = fractal_df['type'].to_numpy()
        f_times = pd.to_datetime(fractal_df['time'], errors='coerce')
        # Make fractal times tz-naive to match price_df index
        if getattr(f_times.dt, 'tz', None) is not None:
            f_times = f_times.dt.tz_convert(None)
        f_times = f_times.to_pydatetime().tolist()

        # Pre-fetch highs and lows as dict for O(1) lookup
        highs = price_df['high'].to_dict()
        lows = price_df['low'].to_dict()

        last_type = None
        last_price = None
        last_ts = None

        for ts, f_type in zip(f_times, f_types):
            if ts not in highs or ts not in lows:
                continue
            high = highs[ts]
            low = lows[ts]

            if last_type is None:
                last_type = f_type
                last_price = high if f_type == 'high' else low
                last_ts = ts
                continue

            if f_type == 'high' and last_type == 'low':
                time_diff = (ts - last_ts).total_seconds() / 60
                price_diff = abs(high - last_price)
                l2h_time.append(time_diff)
                l2h_price.append(price_diff)
                l2h_events.append({"start": last_ts, "end": ts, "time_diff": time_diff, "price_diff": price_diff})
                last_type, last_price, last_ts = 'high', high, ts
            elif f_type == 'low' and last_type == 'high':
                time_diff = (ts - last_ts).total_seconds() / 60
                price_diff = abs(low - last_price)
                h2l_time.append(time_diff)
                h2l_price.append(price_diff)
                h2l_events.append({"start": last_ts, "end": ts, "time_diff": time_diff, "price_diff": price_diff})
                last_type, last_price, last_ts = 'low', low, ts
            elif f_type == 'high' and last_type == 'high':
                if high > last_price:
                    last_price, last_ts = high, ts
            elif f_type == 'low' and last_type == 'low':
                if low < last_price:
                    last_price, last_ts = low, ts

        return {
            "low_to_high": {"time": l2h_time, "price": l2h_price, "raw_events": l2h_events},
            "high_to_low": {"time": h2l_time, "price": h2l_price, "raw_events": h2l_events},
        }

    def stats(arr):
        if not arr:
            return {"mean": np.nan, "median": np.nan, "max": np.nan, "min": np.nan,
                    "count": 0, "std": np.nan, "p25": np.nan, "p75": np.nan}
        arr = np.array(arr)
        return {
            "mean": arr.mean(),
            "median": np.median(arr),
            "max": arr.max(),
            "min": arr.min(),
            "count": len(arr),
            "std": arr.std(),
            "p25": np.percentile(arr, 25),
            "p75": np.percentile(arr, 75)
        }

    # Lower timeframe
    lower_dirs = calc_directional_sweeps(df_fractal, df)

    # Combine both low_to_high and high_to_low stats under "lower_tf" and "higher_tf"
    return {
            "low_to_high": {
                "time": stats(lower_dirs["low_to_high"]["time"]),
                "price": stats(lower_dirs["low_to_high"]["price"]),
                "raw_events": lower_dirs["low_to_high"]["raw_events"]
            },
            "high_to_low": {
                "time": stats(lower_dirs["high_to_low"]["time"]),
                "price": stats(lower_dirs["high_to_low"]["price"]),
                "raw_events": lower_dirs["high_to_low"]["raw_events"]
            }}