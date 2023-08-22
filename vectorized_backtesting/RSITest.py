import numpy as np
import pandas as pd
import Vectorized

class RSITest(Vectorized.Vectorized):
    def __init__(self, symbol, start, end, tc, window, buy_threshold, sell_threshold, granularity):
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.window = window
        super().__init__(symbol, start, end, tc, granularity=granularity)

    def test_strategy(self):
        df = self._data.copy().dropna()
        df['Returns'] = np.log(df.price.div(df.price.shift(1)))
        df.dropna(inplace=True)
        # Compute RSI Columns regarding gain/loss
        df['Change'] = df['price'].diff()
        df['Gain'] = df['Change'].mask(df['Change'] < 0, 0.0)
        df['Loss'] = -df['Change'].mask(df['Change'] > 0, -0.0)
        df['Average Gain'] = df['Gain'].rolling(self.window).mean()
        df['Average Loss'] = df['Loss'].rolling(self.window).mean()
        df['Relative Strength'] = df['Average Gain'] / df['Average Loss']
        df['RSI'] = 100 - 100 / (1 + df['Relative Strength'])
        # If RSI exceeds the buy threshold, go long
        df['position'] = np.where(df['RSI'] > self.buy_threshold, 1, np.nan)
        # If RSI exceeds the sell threshold, go short
        df['position'] = np.where(df['RSI'] < self.sell_threshold, -1, df['position'])
        # Assume a neutral hold otherwise
        df['position'] = df['position'].ffill().fillna(0)
        # Shift strategy by 1 to get rid of first value
        df['strategy'] = df['position'].shift(1) * df['Returns']
        # determine the number of trades in each bar
        df['trades'] = df['position'].diff().fillna(0).abs()
        df['hits'] = np.sign(df['Returns']) * np.sign(df['position'])
        # subtract transaction/trading costs (# of trades * half spread) from pre-cost return
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