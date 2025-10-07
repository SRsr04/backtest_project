from Utils.websocket_client import WS_Client
from technicals import detect_fvg
from strategy import get_signal
from constants import API_KEY, API_SECRET #SYMBOL
from pybit.unified_trading import HTTP

session = HTTP(demo=True, api_key=API_KEY, api_secret=API_SECRET)

RISK_USD = 1000.0

def qty_from_risk(entry_price, stop_loss, risk_usd = RISK_USD):

    dist = abs(entry_price - stop_loss)
    if dist <= 0:
        return '0'
    qty = risk_usd / dist
    return round(qty, 3)


def on_candle(ohlc):

    if len(ohlc) <= 2:

        return

    last_3_candels = ohlc.iloc[-3:]
    fvg, fvg_top, fvg_bottom = detect_fvg(last_3_candels)

    current_candle = last_3_candels.iloc[-1].copy()
    current_candle['fvg'] = fvg
    current_candle['fvg_top'] = fvg_top
    current_candle['fvg_bottom'] = fvg_bottom

    signal, stop_loss, take_profit = get_signal(current_candle=current_candle)

    if signal not in (1, -1):
        return 
    
    qty = qty_from_risk(entry_price=current_candle['close'], stop_loss=stop_loss)
    if float(qty) <= 0:
        print('SL/Entry < 0')
        return
    
    side = 'Buy' if signal == 1 else 'Sell'
    session.place_order(
        category='linear', 
        # symbol=SYMBOL, 
        side=side, 
        order_type='Market', 
        qty=1, 
        time_in_force='GoodTillCancel', 
        takeProfit=str(take_profit), 
        stopLoss=str(stop_loss))


if __name__ == '__main__':
    client = WS_Client(on_candle=on_candle)
    client.run() 