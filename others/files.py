import pandas as pd

def перевірити_типи_даних(файл_csv):
    """
    Завантажує CSV-файл та виводить типи даних у кожній колонці.

    Аргументи:
        файл_csv (str): Шлях до CSV-файлу.
    """
    try:
        df = pd.read_csv(файл_csv)
        print(df.dtypes)
    except FileNotFoundError:
        print(f"Помилка: Файл {файл_csv} не знайдено.")
    except Exception as e:
        print(f"Сталася помилка: {e}")

# Замість "your_file.csv" вкажіть шлях до вашого файлу
перевірити_типи_даних("bos_m5.csv")
перевірити_типи_даних("fvg_m15.csv")
перевірити_типи_даних("m5_candels.csv")
перевірити_типи_даних("m15_candels.csv")