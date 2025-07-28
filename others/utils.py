import pandas as pd
import pytz
import logging

def custom_date_parser(date_str):
    """
    Більш надійний парсер дат, що обробляє різні формати.
    """
    if pd.isna(date_str):
        return pd.NaT
    # Просто дозволяємо pandas автоматично визначити формат
    return pd.to_datetime(date_str, errors='coerce')

def is_trading_hour(current_time: pd.Timestamp) -> bool:
    """
    Перевіряє, чи поточний час знаходиться в межах торгових годин (Пн-Пт, 08:00-22:59 Київського часу).
    """
    if current_time.tzinfo is None:
        kiev_tz = pytz.timezone('Europe/Kiev')
        current_time = current_time.tz_localize(kiev_tz)
    else:
        current_time = current_time.tz_convert(pytz.timezone('Europe/Kiev'))
    
    if current_time.dayofweek >= 5:
        return False
        
    if 8 <= current_time.hour < 23:
        return True
    else:
        return False

def calculate_drawdown(equity_curve, initial_balance):
    """
    Розраховує максимальну просадку від початкового балансу (для проп-фірм).
    """
    if not equity_curve:
        return 0.0, 0.0
    
    equity_series = pd.Series(equity_curve)
    
    # Знаходимо мінімальне значення балансу за весь період
    min_equity = equity_series.min()
    
    # Просадка рахується тільки якщо баланс впав нижче початкового
    if min_equity < initial_balance:
        absolute_drawdown = initial_balance - min_equity
        percent_drawdown = (absolute_drawdown / initial_balance) * 100
    else:
        absolute_drawdown = 0.0
        percent_drawdown = 0.0
        
    return absolute_drawdown, percent_drawdown

def breakeven_info(entry_price, position_size, trade_type):
    """Розраховує комісію та ціну беззбитку."""
    commission_percent = 0.04
    commission = entry_price * position_size * (commission_percent / 100) * 2 # Вхід + вихід
    
    commission_per_unit = commission / position_size
    
    if trade_type == 'long':
        be_price = entry_price + commission_per_unit
    else: # short
        be_price = entry_price - commission_per_unit
        
    return commission, be_price

def calculate_max_losing_streak(outcomes):
    """Розраховує максимальну кількість збиткових угод поспіль."""
    if outcomes.empty:
        return 0
    max_streak = 0
    current_streak = 0
    for outcome in outcomes:
        if outcome == 'loss':
            current_streak += 1
        else:
            if current_streak > max_streak:
                max_streak = current_streak
            current_streak = 0
    # Фінальна перевірка, якщо серія закінчується останньою угодою
    if current_streak > max_streak:
        max_streak = current_streak
    return max_streak