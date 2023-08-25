import numpy as np
import Trader
from stocktrends import Renko
import statsmodels.api as sm
import copy

class RenkoMACD(Trader.Trader):
    def __init__(
        self, 
        conf_file, 
        instrument, 
        bar_length,
        units,
        window,
        duration,
        fast_MA = 12,
        slow_MA = 26,
        sig = 9
    ):
        self.fast_MA = fast_MA
        self.slow_MA = slow_MA
        self.sig = sig
        super().__init__(conf_file, instrument, bar_length, units, duration, window)

    def MACD(self):
        df = self.raw_data.copy()
        df['Fast MA'] = df[self.instrument].ewm(span=self.fast_MA, adjust=False).mean()
        df['Slow MA'] = df[self.instrument].ewm(span=self.slow_MA, adjust=False).mean()
        df['MACD'] = df['Fast MA'] - df['Slow MA']
        df['Signal'] = df['MACD'].ewm(span=self.sig, adjust=False).mean()
        return (df['MACD'], df['Signal'])
    
    def ATR(self, n):
        df = self.raw_data.copy().dropna(how='any')
        df['H-L'] = abs(df['High'] - df['Low'])
        df['H-PC'] = abs(df['High'] - df[self.instrument].shift(1))
        df['L-PC'] = abs(df['Low'] - df[self.instrument].shift(1))
        df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1, skipna=False)
        df['ATR'] = df['TR'].rolling(n).mean()
        returned_df = df.drop(['H-L', 'H-PC', 'L-PC'], axis=1)
        return returned_df
    
    # slope - function to calculate the slope of n consecutive points on a plot
    def slope(ser, n):
        slopes = [i*0 for i in range(n-1)]
        for i in range(n,len(ser)+1):
            y = ser[i-n:i]
            x = np.array(range(n))
            y_scaled = (y - y.min())/(y.max() - y.min())
            x_scaled = (x - x.min())/(x.max() - x.min())
            x_scaled = sm.add_constant(x_scaled)
            model = sm.OLS(y_scaled,x_scaled)
            results = model.fit()
            slopes.append(results.params[-1])
        slope_angle = (np.rad2deg(np.arctan(np.array(slopes))))
        return np.array(slope_angle)

    def RENKO(self, DF):
        df = DF.copy()
        df.reset_index(inplace=True)
        df = df.iloc[:, [0, 1, 2, 3]]
        df.columns = ['Date', 'Close', 'High', 'Low']
        df2 = Renko(df)
        df2.brick_size = round(self.ATR(n=120)['ATR'][-1], 4)
        renko_df = df2.get_bricks()
        renko_df['Bar Num'] = np.where(renko_df['uptrend'] == True, 1, np.where(renko_df['uptrend'] == False, -1, 0))
        for i in range(1, len(renko_df['Bar Num'])):
            if renko_df['Bar Num'][i] > 0 and renko_df['Bar Num'][i - 1] > 0:
                renko_df['Bar Num'][i] += renko_df['Bar Num'][i - 1]
            elif renko_df['Bar Num'][i] < 0 and renko_df['Bar Num'][i - 1] < 0:
                renko_df['Bar Num'][i] += renko_df['Bar Num'][i - 1]
        renko_df.drop_duplicates(subset='date', keep='last', inplace=True)
        return renko_df
    
    def define_strategy(self):
        data = self.raw_data.copy()
        data["Date"] = data.index
        renko = self.RENKO(data)
        renko.columns = ["Date",'Close', 'High', 'Low', 'uptrend', 'Bar Num']
        merged_df = df.merge(renko.loc[:,["Date","Bar Num"]],how="outer",on="Date")
        merged_df['bar_num'].fillna(method='ffill',inplace=True)
        merged_df['macd']= self.MACD(merged_df, 12, 26, 9)[0]
        merged_df['macd_sig']= self.MACD(merged_df, 12, 26, 9)[1]
        merged_df['macd_slope'] = self.slope(merged_df["macd"], 5)
        merged_df['macd_sig_slope'] = self.slope(merged_df["macd_sig"], 5)

        df = copy.deepcopy(merged_df)
        df['position'] = np.where(df['bar_num'] >= 2 and df['macd'] > df['macd_sig'] and df['macd_slope'] > df['macd_sig_slope'], 1, np.nan)
        df['position'] = np.where(df['bar_num'] <= -2 and df['macd'] < df['macd_sig'] and df['macd_slope'] < df['macd_sig_slope'], -1, df['position'])


        