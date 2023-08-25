import numpy as np
import Trader


class SMACrossover(Trader.Trader):
    def __init__(self, conf_file, instrument, bar_length, units, duration, Slow_MA=200, Fast_MA=50):
        self.Slow_MA = Slow_MA
        self.Fast_MA = Fast_MA
        super().__init__(conf_file, instrument, bar_length, units, duration)

    def define_strategy(self):
        df = self.raw_data.copy()
        df['Slow MA'] = df[self.instrument].rolling(self.Slow_MA).mean()
        df['Fast MA'] = df[self.instrument].rolling(self.Fast_MA).mean()
        df['position'] = np.where(df['Fast MA'] > df['Slow MA'], 1, -1)
        self.data = df.copy()