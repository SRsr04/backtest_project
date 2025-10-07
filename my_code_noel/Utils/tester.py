import pandas as pd
def run_backtest(df):

    LONG = 1
    SHORT = -1
    
    current_trades = {}
    trades_data = {}

    trade_open = False

    if len(df) > 0 and len(df[df['signal'] != 0]) > 0:

        for i in range(len(df) - 1):

            cc = df.iloc[i]
            nc = df.iloc[i+1]
            

            if cc['signal'] and not trade_open:

                direction = LONG if cc['signal'] == LONG else SHORT

                current_trades[len(current_trades)] = {
                    'direction': direction,
                    'entry_level': nc['open'],
                    'stop_loss': cc['stop_loss'],
                    'take_profit': cc['take_profit'],
                    'entry_index': i
                    }
                
                trade_open = True

            if trade_open:

                trades_to_remove = []

                for n_of_trade, current_trade in current_trades.items():

                    direction = current_trade['direction']
                    entry =  current_trade['entry_level']
                    sl = current_trade['stop_loss']
                    tp = current_trade['take_profit']
                    profit = None

                    if direction == LONG:

                        if (nc['low'] <= sl and nc['high'] >= tp or 
                            nc['low'] <= sl):

                            profit = sl - entry

                        elif nc['high'] > tp:

                            profit = tp - entry
                    
                    elif direction == SHORT:

                        if (nc['high'] >= sl and nc['low'] <= tp or 
                            nc['high'] >= sl):

                            profit = entry - sl

                        elif nc['low'] < tp:
                            profit = entry - tp

                    if profit is not None:
                        trades_data[len(trades_data)] = {
                            'direction': direction,
                            'entry_level': entry,
                            'stop_loss': sl,
                            'take_profit': tp,
                            'profit': profit,
                            'entry_index': current_trade['entry_index'],
                            'exit_index': i + 1
                            }
                        
                        trade_open = False
                        trades_to_remove.append(n_of_trade)
                
                if len(trades_to_remove) > 0:
                    for trade in trades_to_remove:
                        current_trades.pop(trade, None)

    return pd.DataFrame(trades_data).transpose()


