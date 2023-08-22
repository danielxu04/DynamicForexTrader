import numpy as np
import pandas as pd
import Vectorized

class BollingerBandsTest(Vectorized.Vectorized):
    def __init__(self, symbol, start, end, tc, SMA, standard_deviations, granularity):
        self.SMA = SMA
        self.standard_deviations = standard_deviations
        super().__init__(symbol, start, end, tc, granularity=granularity)

    def test_strategy(self):
        df = self._data.copy().dropna()
        df['Returns'] = np.log(df.price.div(df.price.shift(1)))
        df.dropna(inplace=True)
        # Compute Mean Reversion Bollinger Bands columns
        df['SMA'] = df['price'].rolling(self.SMA).mean()
        df['Lower Band'] = df['SMA'] - (df['price'].rolling(self.SMA).std() * self.standard_deviations)
        df['Upper Band'] = df['SMA'] + (df['price'].rolling(self.SMA).std() * self.standard_deviations)
        df.dropna(inplace=True)
        # Create helper column, 'distance' - the distance between price and SMA
        df['Distance'] = df['price'] - df['SMA']
        # If oversold (Price < Lower Band), then we go long
        df['position'] = np.where(df['price'] < df['Lower Band'], 1, np.nan)
        # If overbought (Price > Upper Band), go short
        df['position'] = np.where(df['price'] > df['Upper Band'], -1, df['position'])
        # If the product between current distance and previous distance is negative, it implies that 
        #   we have crossed the SMA; thus, go neutral
        df['position'] = np.where(df['Distance'] * df['Distance'].shift(1) < 0, 0, df['position'])
        # Otherwise, we hold the previous position, and we can achieve this with a forward fill (ffil), 
        #   except for the first position (NaN), where we will fill with 0
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