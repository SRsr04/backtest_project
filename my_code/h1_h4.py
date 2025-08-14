import pandas as pd
from data_loader import find_fractals, strip_offset_keep_wall_time
import numpy as np

def load_data(h1_path, h4_path):
    h1_df = pd.read_csv(h1_path)
    h4_df = pd.read_csv(h4_path)

    for name, df in (('h1', h1_df), ('h4', h4_df)):
        df['datetime'] = strip_offset_keep_wall_time(df['datetime'])
        df.set_index('datetime', inplace=True)
        df.sort_index(inplace=True)
        if name == 'h1':
            h1_df = df
        else:
            h4_df = df

    return h1_df, h4_df


def sweep(df, higher_df):
    """
    Calculate statistics for low→high and high→low sweep sequences separately.
    Tracks time and price differences for each direction.
    """
    import numpy as np
    df_fractal = find_fractals(df)
    higher_fractal = find_fractals(higher_df)

    def calc_directional_sweeps(fractal_df, price_df):
        # Two directions: low→high and high→low
        l2h_time = []
        l2h_price = []
        h2l_time = []
        h2l_price = []

        last_sweep_type = None  # "low" or "high"
        last_sweep_price = None
        last_sweep_ts = None

        for _, row in fractal_df.iterrows():
            ts = row['time']
            row_candle = price_df.loc[ts]
            if isinstance(row_candle, pd.DataFrame):
                row_candle = row_candle.iloc[0]
            high = row_candle['high']
            low = row_candle['low']

            # Determine sweep type
            # If this fractal is a new high
            if last_sweep_type is None:
                # Initialize with first sweep type
                if row['type'] == 'high':
                    last_sweep_type = "high"
                    last_sweep_price = high
                    last_sweep_ts = ts
                elif row['type'] == 'low':
                    last_sweep_type = "low"
                    last_sweep_price = low
                    last_sweep_ts = ts
                continue

            # Only record when the type switches
            if row['type'] == 'high' and last_sweep_type == "low":
                # low→high
                time_diff = (ts - last_sweep_ts).total_seconds() / 60
                price_diff = abs(high - last_sweep_price)
                l2h_time.append(time_diff)
                l2h_price.append(price_diff)
                last_sweep_type = "high"
                last_sweep_price = high
                last_sweep_ts = ts
            elif row['type'] == 'low' and last_sweep_type == "high":
                # high→low
                time_diff = (ts - last_sweep_ts).total_seconds() / 60
                price_diff = abs(low - last_sweep_price)
                h2l_time.append(time_diff)
                h2l_price.append(price_diff)
                last_sweep_type = "low"
                last_sweep_price = low
                last_sweep_ts = ts
            # If same type, update last_sweep if new extremum
            elif row['type'] == 'high' and last_sweep_type == "high":
                if high > last_sweep_price:
                    last_sweep_price = high
                    last_sweep_ts = ts
            elif row['type'] == 'low' and last_sweep_type == "low":
                if low < last_sweep_price:
                    last_sweep_price = low
                    last_sweep_ts = ts

        return {
            "low_to_high": {"time": l2h_time, "price": l2h_price},
            "high_to_low": {"time": h2l_time, "price": h2l_price},
        }

    def stats(arr):
        if not arr:
            return {"mean": None, "median": None, "max": None, "min": None, "count": 0, "std": None, "p25": None, "p75": None}
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
    # Higher timeframe
    higher_dirs = calc_directional_sweeps(higher_fractal, higher_df)

    # Combine both low_to_high and high_to_low stats under "lower_tf" and "higher_tf"
    return {
        "lower_tf": {
            "low_to_high": {
                "time": stats(lower_dirs["low_to_high"]["time"]),
                "price": stats(lower_dirs["low_to_high"]["price"])
            },
            "high_to_low": {
                "time": stats(lower_dirs["high_to_low"]["time"]),
                "price": stats(lower_dirs["high_to_low"]["price"])
            }
        },
        "higher_tf": {
            "low_to_high": {
                "time": stats(higher_dirs["low_to_high"]["time"]),
                "price": stats(higher_dirs["low_to_high"]["price"])
            },
            "high_to_low": {
                "time": stats(higher_dirs["high_to_low"]["time"]),
                "price": stats(higher_dirs["high_to_low"]["price"])
            }
        }
    }
    

if __name__ == "__main__":
    h1_df, h4_df = load_data("h1_candels.csv", "h4_candels.csv")

    # sweep на основі H1 і H4
    results = sweep(h1_df, h4_df)
    print("Sweep Statistics (5 Years):\n")
    for tf, metrics in results.items():
        print(f"{tf}:")
        for direction, direction_metrics in metrics.items():
            print(f"  {direction}:")
            for category, stats_dict in direction_metrics.items():
                print(f"    {category}:")
                for stat_name, stat_value in stats_dict.items():
                    if stat_name == "count":
                        print(f"      {stat_name}: {stat_value}")
                    else:
                        if stat_value is None:
                            print(f"      {stat_name}: None")
                        else:
                            print(f"      {stat_name}: {stat_value:.2f}")
                print()
            print()
        print()

    # Recommended Filter Parameters for Backtest
    # For lower_tf, use low_to_high as representative (could be averaged if desired)
    lower_tf_time_window = (
        results['lower_tf']['low_to_high']['time']['p25'],
        results['lower_tf']['low_to_high']['time']['p75']
    )
    lower_tf_price_window = (
        results['lower_tf']['low_to_high']['price']['p25'],
        results['lower_tf']['low_to_high']['price']['p75']
    )
    higher_tf_time_window = (
        results['higher_tf']['low_to_high']['time']['p25'],
        results['higher_tf']['low_to_high']['time']['p75']
    )
    higher_tf_price_window = (
        results['higher_tf']['low_to_high']['price']['p25'],
        results['higher_tf']['low_to_high']['price']['p75']
    )
    lower_tf_median_time = results['lower_tf']['low_to_high']['time']['median']
    lower_tf_median_price = results['lower_tf']['low_to_high']['price']['median']
    higher_tf_median_time = results['higher_tf']['low_to_high']['time']['median']
    higher_tf_median_price = results['higher_tf']['low_to_high']['price']['median']

    print("Recommended Filter Parameters for Backtest:\n")
    print("lower_tf:")
    print(f"  time_window (p25–p75): {lower_tf_time_window[0]:.2f} – {lower_tf_time_window[1]:.2f} minutes")
    print(f"  price_window (p25–p75): {lower_tf_price_window[0]:.2f} – {lower_tf_price_window[1]:.2f} points")
    print(f"  median_time: {lower_tf_median_time:.2f} minutes")
    print(f"  median_price: {lower_tf_median_price:.2f} points")
    print()
    print("higher_tf:")
    print(f"  time_window (p25–p75): {higher_tf_time_window[0]:.2f} – {higher_tf_time_window[1]:.2f} minutes")
    print(f"  price_window (p25–p75): {higher_tf_price_window[0]:.2f} – {higher_tf_price_window[1]:.2f} points")
    print(f"  median_time: {higher_tf_median_time:.2f} minutes")
    print(f"  median_price: {higher_tf_median_price:.2f} points")