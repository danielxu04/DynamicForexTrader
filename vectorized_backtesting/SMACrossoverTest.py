import numpy as np
import pandas as pd
import Vectorized

class SMACrossoverTest(Vectorized.Vectorized):
    def __init__(self, symbol, start, end, tc, Fast_SMA, Slow_SMA, granularity):
        self.Fast_SMA = Fast_SMA
        self.Slow_SMA = Slow_SMA
        super().__init__(symbol, start, end, tc, granularity=granularity)

    def test_strategy(self):
        df = self._data.copy().dropna()
        df['Returns'] = np.log(df.price.div(df.price.shift(1)))
        df['Fast SMA'] = df.price.rolling(self.Fast_SMA).mean()
        df['Slow SMA'] = df.price.rolling(self.Slow_SMA).mean()
        df.dropna(inplace=True)
        # When Fast SMA 'crosses over' the Slow SMA, go long; otherwise, go short
        df['position'] = np.where(df['Fast SMA'] > df['Slow SMA'], 1, -1)
        df['strategy'] = df['position'].shift(1) * df['Returns']
        # determine the number of trades in each bar
        df['trades'] = df['position'].diff().fillna(0).abs()
        df['hits'] = np.sign(df['Returns']) * np.sign(df['position'])
        # subtract transaction/trading costs from pre-cost return
        df['strategy'] = df['strategy'] - (df['trades'] * self.tc)
        df.dropna(inplace=True)
        # Cumulative performance columns
        df['Standard Cumulative Returns'] = df['Returns'].cumsum().apply(np.exp)
        df['Strategy Cumulative Returns'] = df['strategy'].cumsum().apply(np.exp)
        self.results = df
        # Absolute performance of strategy and out/under-performance of strategy
        performance = df['Strategy Cumulative Returns'].iloc[-1]
        outperformance = df['Standard Cumulative Returns'].iloc[-1]
        return round(performance, 6), round(outperformance, 6)