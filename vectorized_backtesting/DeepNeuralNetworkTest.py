import keras
import pickle
import numpy as np
import Vectorized

class DeepNeuralNetworkTest(Vectorized.Vectorized):
    def __init__(self, symbol, start, end, tc, granularity='1d', window=50, Fast_SMA=75, Slow_SMA=150, Fast_EMA=12, Slow_EMA=26, signal=9, rsi_window=14, model=None, pkl=None, lags=8):
        self.window = window
        self.Fast_SMA = Fast_SMA
        self.Slow_SMA = Slow_SMA
        self.Fast_EMA = Fast_EMA
        self.Slow_EMA = Slow_EMA
        self.signal = signal
        self.rsi_window = rsi_window
        self.model = model
        self.pkl = pkl
        self.lags = lags
        super().__init__(symbol, start, end, tc, granularity=granularity)

    def load_model(self, model_path, pkl_path):
        self.model = keras.models.load_model(model_path)
        parameters = pickle.load(open(pkl_path, 'rb'))
        self.mean = parameters['mu']
        self.std = parameters['sigma']

    def test_strategy(self):
        data = self._data.copy().dropna()
        df = data.rename(columns={'price': 'Price'})
        # Log Returns
        df['Returns'] = np.log(df['Price'] / df['Price'].shift(1))
        # Direction for class weight balancing to eliminate buy bias
        df['Direction'] = np.where(df['Returns'] > 0, 1, 0)
        # MACD Growth Indicator
        macd = df['Price'].ewm(span=self.Fast_EMA, adjust=False).mean() - df['Price'].ewm(span=self.Slow_EMA, adjust=False).mean()
        signal = macd.ewm(span=self.signal, adjust=False).mean()
        df['MACD'] = macd - signal
        # SMA Crossover with 75-150 Fast SMA-Slow SMA split
        df['SMA Crossover'] = df['Price'].rolling(self.Fast_SMA).mean() - df['Price'].rolling(self.Slow_SMA).mean()
        # Mean Reversion (similar to Bollinger Bands) with 50-period window
        df['Mean Reversion'] = (df['Price'] - df['Price'].rolling(self.window).mean()) / df['Price'].rolling(self.window).std()
        # Rolling Min/Max normalization with current price
        df['Rolling Min'] = (df['Price'].rolling(self.window).min() / df['Price']) - 1
        df['Rolling Max'] = (df['Price'].rolling(self.window).max() / df['Price']) - 1
        # Momentum
        df['Momentum'] = df['Returns'].rolling(self.window).mean()
        change = df['Price'].diff()
        df['RSI'] = 100 - (100 / (1 + (change.mask(change < 0, 0.0).rolling(self.rsi_window).mean() / -change.mask(change > 0, -0.0).rolling(self.rsi_window).mean())))
        # Volatility
        df['Volatility'] = df['Returns'].rolling(self.window).std()
        df.dropna(inplace=True)

        # ------------------------------------ MODEL PREDICTION ---------------------------------------
        columns = []
        features = ['Returns', 'Direction', 'MACD', 'SMA Crossover', 'Mean Reversion', 'Rolling Min', 'Rolling Max', 'Momentum', 'RSI', 'Volatility']

        for feature in features:
            for lag in range(1, self.lags + 1):
                column = '{}_lag_{}'.format(feature, lag)
                df[column] = df[feature].shift(lag)
                columns.append(column)
        df.dropna(inplace=True)

        standardized_df = (df - self.mean) / self.std

        df['Probability'] = self.model.predict(standardized_df[columns])
        df['Probability'] = df['Probability'].rolling(50).mean()

        # If probability < 0.48, go short
        df['position'] = np.where(df['Probability'] < 0.475, -1, np.nan)
        # If probability > 0.52, go long
        df['position'] = np.where(df['Probability'] > 0.535, 1, df['position'])
        # Otherwise, either forward fill or go neutral
        df['position'] = df['position'].ffill().fillna(0)

        # Calculate strategy returns based on position df
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