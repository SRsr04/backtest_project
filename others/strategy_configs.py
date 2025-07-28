# strategy_configs.py

# Список словників, де кожен словник - це одна торгова стратегія.
# Починаємо з сетапу M5/M115.

STRATEGIES = [
    {
        'name': 'm5m15',
        'fvg_tf': 'm15',
        'entry_tf': 'm5',
        'mitigation_tf': 'm15', # FVG з M15 мітигується на M15
        'use_breakeven': False,
        'htf_filter': 'h1', 
        'time_limit_hours': 2, # Часовий ліміт 2 години для пошуку сетапу
        'files': {
            'candles': '/Users/synyshyn_04/BTC_backtest_new/m5_candels.csv',
            'fvg': '/Users/synyshyn_04/BTC_backtest_new/fvg_m15.csv',
            'bos': '/Users/synyshyn_04/BTC_backtest_new/bos_m5.csv',
            'm15_candles': '/Users/synyshyn_04/BTC_backtest_new/m15_candels.csv', # Потрібен для мітигації
            'h1_candles': '/Users/synyshyn_04/BTC_backtest_new/h1_candels.csv',   # Поки не використовується, але потрібен для структури
            'h4_candles': '/Users/synyshyn_04/BTC_backtest_new/h4_candels.csv'    # Поки не використовується, але потрібен для структури
        }
    },
    # {
    #     'name': 'm5m15_be',
    #     'fvg_tf': 'm15',
    #     'entry_tf': 'm5',
    #     'mitigation_tf': 'm15',
    #     'use_breakeven': True, # <--- Змінено
    #     'htf_filter': None, 
    #     'time_limit_hours': 2,
    #     'files': {
    #         'candles': 'm5_candels.csv',
    #         'fvg': 'fvg_m15.csv',
    #         'bos': 'bos_m5.csv',
    #         'm15_candles': 'm15_candels.csv',
    #         'h1_candles': 'h1_candels.csv',
    #         'h4_candles': 'h4_candels.csv'
    #     }
    # },
]