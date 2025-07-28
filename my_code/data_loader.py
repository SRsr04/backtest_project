import pandas as pd
import numpy as np
import sys

RED = "\033[91m"
GREEN = "\033[92m"
RESET = "\033[0m"

def load_data():
    # 1. Прочитати всі 4 файли
    m5_df = pd.read_csv('m5_candels.csv')
    m15_df = pd.read_csv('m15_candels.csv')
    fvg_df = pd.read_csv('fvg_m15.csv')
    bos_df = pd.read_csv('bos_m5.csv')
    # 2. Конвертувати дат
    def to_kiev(series):
        return pd.to_datetime(series, errors='coerce', utc = True).dt.tz_convert('Europe/Kiev').dt.tz_localize(None)
    m5_df['datetime'] = to_kiev(m5_df.datetime)
    m15_df['datetime'] = to_kiev(m15_df.datetime)
    fvg_df['time'] = to_kiev(fvg_df.time)
    bos_df['bos_time'] = to_kiev(bos_df.bos_time)
    bos_df['bos_time_kiev'] = to_kiev(bos_df.bos_time_kiev)
    # 3. Поставити індекси для свічок і відсортувати
    m5_df = m5_df.set_index('datetime').sort_index(ascending=True)
    m15_df = m15_df.set_index('datetime').sort_index(ascending=True)
    # 4. Прибрати зайві колонки
    m15_df.drop(columns=['timestamp_utc'], inplace=True, errors='ignore')
    # 5. Повернути DataFrame
    return m5_df, m15_df, fvg_df, bos_df


def find_fractals(df):
    fractals = []
    highs = df['high'].values
    lows = df['low'].values
    idx = df.index

    for i in range(1, len(df) - 1):
        # 1. Перевірити верхній фрактал
        if highs[i] > highs[i - 1] and highs[i] > highs[i + 1]:
            fractals.append({'time': idx[i], 'price': highs[i], 'type': 'high'})
        elif lows[i] < lows[i - 1] and lows[i] < lows[i + 1]:
            fractals.append({'time': idx[i], 'price': lows[i], 'type': 'low'})
    return pd.DataFrame(fractals)

def fractals_after_fvg(fvg_df, fractals_df):
    results = []
    # масив часів фракталів
    fractals_times = fractals_df['time']
    
    for _, fvg in fvg_df.iterrows():
        fvg_time = pd.Timestamp(fvg['time']+ pd.Timedelta(minutes=15))
        
        # шукаємо позицію першого фрактала після FVG
        idx = fractals_times.searchsorted(fvg_time)
        
        # перевіряємо, що є хоча б два фрактали після FVG
        if idx + 1 >= len(fractals_times):
            continue
        
        f1 = fractals_df.iloc[idx]
        f2 = fractals_df.iloc[idx + 1]
        
        results.append({
            "fvg_time": pd.Timestamp(fvg['time']) + pd.Timedelta(minutes=15),
            "fvg_type": fvg['type'],
            "f1_time": f1['time'], "f1_price": f1['price'], "f1_type": f1['type'],
            "f2_time": f2['time'], "f2_price": f2['price'], "f2_type": f2['type'],
            "fvg_min": fvg['min'],
            "fvg_max": fvg['max'],
        })
    
    return results

def add_bos_after_f2(fvg_f1_f2_list, bos_df, debug=False):
    results = []
    bos_df = bos_df.sort_values('bos_time_kiev').reset_index(drop=True)
    bos_times = bos_df['bos_time_kiev'].values
    bos_closes = bos_df['close'].values
    bos_types = bos_df['type'].values
    bos_levels = bos_df['level'].values
    

    for n, item in enumerate(fvg_f1_f2_list):
        f2_time = np.datetime64(item['f2_time'])
        f1_price = item['f1_price']
        f1_type = item['f1_type']
        fvg_min = item['fvg_min']
        fvg_max = item['fvg_max']
        f2_price = item['f2_price']
        fvg_type = item['fvg_type']

        if fvg_type == 'bullish':
            if not (f1_price > fvg_max and fvg_min <= f2_price <= fvg_max):
                if debug:
                    print(f"{RED}❌ F1/F2 не відповідають правилам для лонга{RESET}")
                continue

        # Логіка для шорта
        elif fvg_type == 'bearish':
            if not (f1_price < fvg_min and fvg_min <= f2_price <= fvg_max):
                if debug:
                    print(f"{RED}❌ F1/F2 не відповідають правилам для шорта{RESET}")
                continue

        if not (fvg_min <= f2_price <= fvg_max):
            if debug:
                print(f"{RED}❌ F2 ({f2_price}) не в зоні FVG ({fvg_min} - {fvg_max}){RESET}")
                continue

        idx = np.searchsorted(bos_times, f2_time)

        if idx >= len(bos_times):
            # if debug:
            #     print(f"[{n}] f2_time {item['f2_time']} → BOS not found (за межами)")
            continue

        bos_deadline = item['fvg_time'] + np.timedelta64(2, 'h')
        if bos_times[idx] > bos_deadline:
          if debug:
           print(f"{RED}❌ BOS запізнився: {bos_times[idx]} (FVG: {item['fvg_time']}){RESET}")
           continue

        if debug:
            print(f"\n[{n}] f2_time: {item['f2_time']} | f1_price: {f1_price} ({f1_type})")
            print("bos candidate at idx:")
            print(bos_df.iloc[idx][['bos_time_kiev','close','level','type']])

        found = False
        while idx < len(bos_times):
            if bos_times[idx] > bos_deadline:
                if debug:
                    print(f"{RED}❌ BOS запізнився: {bos_times[idx]} (FVG: {item['fvg_time']}){RESET}")
                    break
            if f1_type == 'high' and bos_closes[idx] > f1_price:
                # if debug:
                #     print(f"{GREEN}✔ BOS знайдений: {bos_times[idx]} > {f1_price}{RESET}")
                item.update({
                    "bos_time": bos_times[idx],
                    "bos_type": bos_types[idx],
                    "bos_price": bos_levels[idx]
                })
                results.append(item)
                break
            elif f1_type == 'low' and bos_closes[idx] < f1_price:
                # if debug:
                #     print(f"{GREEN}✔ BOS знайдений: {bos_times[idx]} < {f1_price}{RESET}")
                item.update({
                    "bos_time": bos_times[idx],
                    "bos_type": bos_types[idx],
                    "bos_price": bos_levels[idx]
                })
                results.append(item)
                break
            idx += 1
        if debug and not found:
            print(f"→ No BOS found after f2_time for this FVG")
    return results


def format_results(results):
    formatted = []
    for r in results:
        formatted.append({
            "fvg_time": pd.Timestamp(r['fvg_time']).strftime("%Y-%m-%d %H:%M"),
            "f1_time": pd.Timestamp(r['f1_time']).strftime("%Y-%m-%d %H:%M"),
            "f2_time": pd.Timestamp(r['f2_time']).strftime("%Y-%m-%d %H:%M"),
            "bos_time": pd.Timestamp(r['bos_time']).strftime("%Y-%m-%d %H:%M"),
            "f1_price": float(r['f1_price']),
            "f2_price": float(r['f2_price']),
            "bos_price": float(r['bos_price']),
            "fvg_min": float(r['fvg_min']),
            "fvg_max": float(r['fvg_max']),
            "fvg_type": r['fvg_type'],
            "bos_type": r['bos_type'],
        })
    return formatted

if __name__ == "__main__":
    m5_df, m15_df, fvg_df, bos_df = load_data()
    m5_fractals = find_fractals(m5_df)
    m15_fractals = find_fractals(m15_df)
    fvg_df['time'] = fvg_df['time'].dt.tz_localize(None)
    m5_fractals['time'] = m5_fractals['time'].dt.tz_localize(None)
    fvg_fractals = fractals_after_fvg(fvg_df, m5_fractals)
    bos_df = bos_df.dropna(subset=['bos_time_kiev']).sort_values('bos_time_kiev').reset_index(drop=True)

    fvg_f1_f2_bos = add_bos_after_f2(fvg_fractals, bos_df, debug=True)

clean_results = format_results(fvg_f1_f2_bos)
for r in clean_results[:5]:
    print(r)

