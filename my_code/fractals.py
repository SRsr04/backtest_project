import pandas as pd

# Завантаження CSV
df = pd.read_csv('backtest_results/2R_H1F_f0.75_s0.0_r3.2_n2.csv')

df['entry_time'] = pd.to_datetime(df['entry_time'])
df['exit_time']  = pd.to_datetime(df['exit_time'])
df = df[df['entry_time'].dt.weekday < 5]  

# Зберегти назад (за бажанням)
df.to_csv('backtest_results/2R_H1F_f0.75_s0.0_r3.2_n2.csv', index=False)

print(df.head())