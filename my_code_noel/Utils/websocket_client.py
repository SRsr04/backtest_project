import json
from websocket import WebSocketApp
from datetime import datetime, timezone
import pandas as pd

WS = 'wss://stream.bybit.com/v5/public/linear'
ARG = 'kline.1.BTCUSDT'

class WS_Client:

    def __init__(self, on_candle):
        self.df = pd.DataFrame(columns=['time', 'open', 'high', 'close', 'low', 'volume'])
        self.on_candle = on_candle   # тут приймаємо колбек

    def on_open(self, ws):
        ws.send(json.dumps({'op': 'subscribe', 'args': [ARG]}))

    def on_message(self, ws, m):
        d = json.loads(m)
        if not d.get('data'):
            return

        items = d['data'] if isinstance(d['data'], list) else [d['data']]
        for k in items:
            if k.get('confirm'):
                candle = {
                    'time': datetime.fromtimestamp(k['start'] // 1000, tz=timezone.utc),
                    'open': float(k['open']),
                    'high': float(k['high']),
                    'close': float(k['close']),
                    'low': float(k['low']),
                    'volume': float(k['volume'])
                }
                candle = pd.DataFrame([candle])
                self.df = pd.concat([self.df, candle], axis=0)
                self.df = self.df.drop_duplicates(subset='time')
                self.df = self.df.sort_values(by='time').reset_index(drop=True)

                print(self.df)
                self.on_candle(self.df)   # виклик переданого колбеку

    def run(self):
        ws = WebSocketApp(WS, on_open=self.on_open, on_message=self.on_message)
        ws.run_forever()