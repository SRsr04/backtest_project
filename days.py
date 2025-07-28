import pandas as pd

# Завантаження CSV-файлу
df = pd.read_csv("btc_backtest_results/backtest_m15_h1_be.csv")

# Перетворення колонки з часом входу в datetime
df['entry_time'] = pd.to_datetime(df['entry_time'])

# Додавання колонки з днем тижня
df['weekday'] = df['entry_time'].dt.day_name()

# Підрахунок статистики по днях
summary_by_day = df.groupby('weekday')['outcome'].value_counts().unstack().fillna(0)

# Додавання підсумку угод і winrate
summary_by_day['total'] = summary_by_day.sum(axis=1)
summary_by_day['winrate_%'] = (summary_by_day.get('win', 0) / summary_by_day['total']) * 100

# Сортуємо дні тижня в правильному порядку
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
summary_by_day = summary_by_day.reindex(day_order)

# Вивід результату
print(summary_by_day)
