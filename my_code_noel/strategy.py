def get_signal(current_candle):

    signal = 0
    stop_loss = 0
    take_profit = 0

    if current_candle['fvg'] == 1:
        signal = 1
        stop_loss = current_candle['fvg_bottom']
        take_profit = current_candle['close'] + [current_candle['close'] - stop_loss] * 2
    elif current_candle['fvg'] == -1:
        signal = -1
        stop_loss = current_candle['fvg_top']
        take_profit = current_candle['close'] - (abs(current_candle['close'] - stop_loss)) * 2
    return signal, stop_loss, take_profit