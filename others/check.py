import pandas as pd

# Список всіх файлів та їх колонок з датами для перевірки
FILES_TO_CHECK = [
    {'name': 'm5_candels.csv', 'date_cols': ['datetime']},
    {'name': 'fvg_m15.csv', 'date_cols': ['datetime']}, # Припускаємо, що колонка вже перейменована на 'datetime'
    {'name': 'bos_m5.csv', 'date_cols': ['bos_time_kiev', 'fract_time_kiev']},
    {'name': 'h1_candels.csv', 'date_cols': ['datetime']},
    {'name': 'm15_candels.csv', 'date_cols': ['datetime']},
]

def diagnose_file(filename, date_columns):
    """Функція для діагностики одного файлу."""
    print(f"\n{'='*20}\nАНАЛІЗУЄМО ФАЙЛ: {filename}\n{'='*20}")
    
    try:
        df = pd.read_csv(filename)
    except FileNotFoundError:
        print(f"ПОМИЛКА: Файл '{filename}' не знайдено. Перевірте назву та розташування.")
        return

    # Перейменовуємо колонку 'time' в 'datetime' для FVG файлу для уніфікації
    if filename == 'fvg_m15.csv' and 'time' in df.columns:
        df.rename(columns={'time': 'datetime'}, inplace=True)

    for col_name in date_columns:
        if col_name not in df.columns:
            print(f"ПОПЕРЕДЖЕННЯ: У файлі '{filename}' відсутня колонка '{col_name}'.")
            continue

        print(f"\n--- Перевірка колонки: '{col_name}' ---")
        
        # Спроба перетворення
        converted_col = pd.to_datetime(df[col_name], errors='coerce')
        
        # Пошук проблемних рядків
        problem_rows = df[converted_col.isna() & df[col_name].notna()]
        
        if not problem_rows.empty:
            print(f"🔥 ЗНАЙДЕНО {len(problem_rows)} ПРОБЛЕМНИХ РЯДКІВ у колонці '{col_name}'!")
            print("Ось перші 5 з них:")
            print(problem_rows[[col_name]].head())
        else:
            print(f"✅ Проблем з форматом дати в колонці '{col_name}' не виявлено.")

# --- Головний цикл діагностики ---
for file_info in FILES_TO_CHECK:
    diagnose_file(file_info['name'], file_info['date_cols'])

print(f"\n{'='*20}\nДіагностику завершено.\n{'='*20}")