import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf


class Instrument:
    def __init__(self, ticker, start, end, source_file=None, start_time=None, end_time=None, granularity="1d"):
        self._ticker = ticker
        self._start = start
        self._end = end
        self._data = None
        self.source_file = source_file
        self.start_time=start_time
        self.end_time=end_time
        self.granularity=granularity
        self.get_data()
        self.log_returns()

    def __repr__(self):
        return "Instrument(ticker={}, start={}, end={})".format(self._ticker, self._start, self._end)
    
    def get_start(self):
        return self._start
    
    def set_start(self, start):
        self._start = start
        self.get_data()

    def get_end(self):
        return self._end
    
    def set_end(self, end):
        self._end = end
        self.get_data()

    def get_ticker(self):
        return self._ticker
    
    def set_ticker(self, ticker):
        self._ticker = ticker
        self.get_data()

    def get_data(self):
        if self.source_file is None:
            data = yf.download(self._ticker, self._start, self._end, interval=self.granularity, progress=False).Close.to_frame()
            data.rename(columns={"Close": "price"}, inplace=True)
        else:
            data = pd.read_csv(self.source_file, parse_dates=["time"], index_col="time")
            if self.start_time is not None and self.end_time is not None:
                data = data.loc[(data.index.hour > self.start_time) & (data.index.hour < self.end_time)]
            if self.granularity is not None:
                data = data.resample(self.granularity, label="right").last().dropna().iloc[:-1]
        self._data = data
        return self._data.copy()

    def log_returns(self):
        self._data["log_returns"] = np.log(self._data.price / self._data.price.shift(1))

    def plot_prices(self):
        self._data.price.plot(figsize=(12, 8))
        plt.title("Price Chart: {}".format(self._ticker), fontsize=13)

    def plot_returns(self, kind="ts"):
        if kind == "ts":
            self._data.log_returns.plot(figsize=(12, 8))
            plt.title("Returns: {}".format(self._ticker), fontsize=15)
        elif kind == "hist":
            self._data.log_returns.hist(
                figsize=(12, 8), bins=int(np.sqrt(len(self._data)))
            )
            plt.title("Frequency of Returns: {}".format(self._ticker), fontsize=15)

    def mean_return(self, freq=None):
        if freq is None:
            return self._data.log_returns.mean()
        else:
            resampled_price = self._data.price.resample(freq).last()
            resampled_returns = np.log(resampled_price / resampled_price.shift(1))
            return resampled_returns.mean()

    def std_returns(self, freq=None):
        if freq is None:
            return self._data.log_returns.std()
        else:
            resampled_price = self._data.price.resample(freq).last()
            resampled_returns = np.log(resampled_price / resampled_price.shift(1))
            return resampled_returns.std()

    def annualized_performance(self):
        mean_return = round(self._data.log_returns.mean() * 252, 3)
        risk = round(self._data.log_returns.std() * np.sqrt(252), 3)
        print("Return: {} | Risk: {}".format(mean_return, risk))