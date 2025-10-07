import pandas as pd
from constants import API_KEY, API_SECRET
from pybit.unified_trading import HTTP
from my_code_noel.Utils.data_proccessing import get_historical_ohlc
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple, List

@dataclass
class Candle:
    o: float; h: float; l: float; c: float

class RollingSMA:

    def __init__(self, period: int):

        self.period = period
        self.buf = deque(maxlen=period)
        self.sum = 0.0
        self.value: Optional[float] = None

    def update(self, x: float) -> Optional[float]:

        if len(self.buf) == self.period:

            self.sum -= self.buf[0]
            
        self.buf.append(x)
        self.sum += x

        if len(self.buf) == self.period:
            self.value = self.sum / self.period

        else:
            self.value = None

        return self.value

class FractalDetector:
    """Подтверждение на закрытии третьей свечи."""
    def __init__(self):
        self.last_confirmed_idx: Optional[int] = None
        self.last_level: Optional[float] = None

        self.last_bu_confirmed_idx: Optional[int] = None
        self.last_bu_level: Optional[float] = None
        self.last_be_confirmed_idx: Optional[int] = None
        self.last_be_level: Optional[float] = None

        self._w = deque(maxlen=3)

    def update(self, idx: int, cndl: Candle):
        self._w.append(cndl)

        if len(self._w) < 3: 

            return None, None  # (type, level)
        
        a, b, c = self._w[0], self._w[1], self._w[2]

        # Бычий фрактал: максимум середины выше соседей
        if b.h > a.h and b.h > c.h:
            self.last_confirmed_idx = idx - 1  # индекс средней свечи
            self.last_level = b.h
            self.last_bu_confirmed_idx = idx - 1
            self.last_bu_level = b.h
            return "bull", b.h
        
        # Медвежий фрактал: минимум середины ниже соседей
        if b.l < a.l and b.l < c.l:
            self.last_confirmed_idx = idx - 1
            self.last_level = b.l
            self.last_be_confirmed_idx = idx - 1
            self.last_be_level = b.l

            return "bear", b.l
        
        return None, None

class FVGDetector:
    
    def __init__(self):
        self._w = deque(maxlen=3)
        self.zones: List[Tuple[int, str, float, float]] = []  # (idx, type, top, bottom)

    def update(self, idx: int, cndl: Candle):
        self._w.append(cndl)
        if len(self._w) < 3:
            return None
        a, b, c = self._w[0], self._w[1], self._w[2]
        # bull
        if c.l > a.h:
            top = c.l; bottom = a.h
            self.zones.append((idx, "bull", top, bottom))
            return ("bull", top, bottom)
        # bear
        if c.h < a.l:
            top = a.l; bottom = c.h
            self.zones.append((idx, "bear", top, bottom))
            return ("bear", top, bottom)
        return None

    def last_between(self, start_idx: int, end_idx: int) -> Optional[Tuple[int, str, float, float]]:
        # Последний FVG, сформированный в (start_idx, end_idx]
        for z in reversed(self.zones):
            zi, *_ = z
            if start_idx < zi <= end_idx:
                return z
        return None

class Strategy:
    def __init__(self, s_sma_per=28, f_sma_per=14):
        self.slow = RollingSMA(s_sma_per)
        self.fast = RollingSMA(f_sma_per)
        self.fract = FractalDetector()
        self.fvg = FVGDetector()
        self.idx = -1
        self.prev_cross: Optional[int] = None  # -1 (bear), +1 (bull)
        # Полезно хранить последние 3 свечи для логики входа «от зоны»
        self.last_candles = deque(maxlen=3)

        # Хранилище сигналов (для быстроты — список dict)
        self._signals: List[dict] = []

    def on_bar(self, cndl: Candle):
        """Вызывай на КАЖДОЕ ЗАКРЫТИЕ свечи."""
        self.idx += 1
        self.last_candles.append(cndl)

        s = self.slow.update(cndl.c)
        f = self.fast.update(cndl.c)

        # Обновляем детекторы
        fract_type, fract_level = self.fract.update(self.idx, cndl)
        fvg_new = self.fvg.update(self.idx, cndl)

        signal = None
        entry = None
        sl = None
        tp = None

        # Логику запускаем только когда обе SMA существуют
        if s is not None and f is not None:
            cross_now = 1 if f > s else (-1 if f < s else 0)
            # Ищем смену знака (пересечение)
            crossed = (self.prev_cross is not None and cross_now != 0 and cross_now != self.prev_cross)
            if self.prev_cross is None and cross_now != 0:
                crossed = False  # самое первое определение тренда
            if crossed:
                # Берём последний подтверждённый фрактал
                last_fr_idx = self.fract.last_confirmed_idx
                if last_fr_idx is None:
                    pass
                else:
                    # Ищем последний FVG после этого фрактала и до текущего бара
                    last_fvg = self.fvg.last_between(start_idx=last_fr_idx, end_idx=self.idx)
                    if last_fvg:
                        zi, ztype, top, bottom = last_fvg
                        if cross_now == 1 and ztype == "bull":
                            # вход «от FVG»: лимитом на верх/низ зоны (на выбор)
                            signal = "long"
                            entry = bottom  # ретест зоны
                            sl = self.fract.last_be_level
                            tp = entry + (entry - sl)
                        elif cross_now == -1 and ztype == "bear":
                            signal = "short"
                            entry = bottom
                            sl = self.fract.last_bu_level
                            tp = entry - (sl - entry)

                        # если сформирован сигнал — сохраняем компактную запись
                        if signal is not None:
                            self._signals.append({
                                "idx": self.idx,
                                "entry": float(entry),
                                "tp": float(tp),
                                "sl": float(sl),
                            })

            self.prev_cross = cross_now if cross_now != 0 else self.prev_cross

        return {
            "idx": self.idx,
            "sma_fast": self.fast.value,
            "sma_slow": self.slow.value,
            "fract_type": fract_type,
            "fract_level": fract_level,
            "fvg_new": fvg_new,            # если на этом баре родился FVG
            "signal": signal,              # "long"/"short" или None
            "entry": entry,
            "stop_loss": sl,
            "take_profit": tp,
        }
    
    def signals_df(self) -> pd.DataFrame:
        """Возвращает DataFrame с колонками: idx, entry, tp, sl."""
        if not self._signals:
            return pd.DataFrame(columns=["idx", "entry", "tp", "sl"])
        return pd.DataFrame(self._signals, columns=["idx", "entry", "tp", "sl"])

# Example
st = Strategy(s_sma_per=28, f_sma_per=14)



session = HTTP(demo=True, api_key=API_KEY, api_secret=API_SECRET)
ohlc = get_historical_ohlc(session=session, symbol='BTCUSDT', interval='1')

for b in range(len(ohlc)): 
    bar = ohlc.iloc[b]
    res = st.on_bar(Candle(bar.open, bar.high, bar.low, bar.close))
    #if res["signal"]:
    #    print("SIG:", res)
        # тут ставишь лимитку 
df_signals = st.signals_df()
print(df_signals)