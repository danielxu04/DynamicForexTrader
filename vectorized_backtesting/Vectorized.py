import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(1, '../utilities')
import Instrument
Instrument = Instrument.Instrument

plt.style.use("seaborn-v0_8")

# VECTORIZED BACKTESTING CLASS
        # Parameters
        # ----------
        # symbol: str
        #     ticker symbol (instrument) to be backtested
        # start: str
        #     start date for data import
        # end: str
        #     end date for data import
        # tc: float
        #     proportional transaction/trading costs per trade
class Vectorized:
    def __init__(self, symbol, start, end, tc, granularity='1d'):
        self.results_overview = None
        self.tc = tc
        self.results = None
        self._instrument = Instrument(symbol, start, end, granularity=granularity)
        self._data = self._instrument.get_data()

    @classmethod
    def from_instrument(cls, instrument, tc):
        instance =  cls(
            instrument.get_ticker(), instrument.get_start(), instrument.get_end(), tc
        )
        instance._instrument = instrument
        instance._data = instance._instrument.get_data()
        return instance

    def __repr__(self):
        return "VectorizedBacktester(symbol={}, start={}, end={})".format(
            self._instrument.get_ticker(),
            self._instrument.get_start(),
            self._instrument.get_end(),
        )

    # Overriden method
    def test_strategy(self):
        # data = self._data.copy().dropna()
        # data["log_returns"] = np.log(data.price / data.price.shift(1))
        # data["position"] = np.sign(data["log_returns"].rolling(1).mean()).mul(-1)
        # data["strategy"] = data["position"].shift(1) * data["log_returns"]
        # data.dropna(inplace=True)

        # # determine the number of trades in each bar
        # data["trades"] = data.position.diff().fillna(0).abs()

        # # subtract transaction/trading costs from pre-cost return
        # data.strategy = data.strategy - data.trades * self.tc

        # data["creturns"] = data["log_returns"].cumsum().apply(np.exp)
        # data["cstrategy"] = data["strategy"].cumsum().apply(np.exp)
        # self.results = data

        # perf = data["cstrategy"].iloc[-1]  # absolute performance of the strategy
        # outperf = perf - data["creturns"].iloc[-1]  # out-/underperformance of strategy

        # return round(perf, 6), round(outperf, 6)
        return

    # PLOT_RESULTS - Plots results of strategy, compares to buy/hold
    def plot_results(self):
        if self.results is None:
            print("Run test_strategy() first.")
        else:
            title = "{} Returns with TC = {}".format(self._instrument.get_ticker(), self.tc)
            self.results[["Strategy Cumulative Returns", "Standard Cumulative Returns"]].plot(title=title, figsize=(12, 8))
    
    # HIT_RATIO - returns proportion of profitable trades
    def hit_ratio(self):
        if self.results is not None and self.results.hits is not None:
            value_count = self.results.hits.value_counts()
            return value_count[1] / (value_count[0] + value_count[-1] + value_count[1])
        return "No hit ratio avaliable. Test strategy before calling this method."
    
    # DETAILED_METRICS - returns detailed performance metrics (MR, CAGR, Sharpe)
    def detailed_metrics(self):
        if self.results is not None:
            mean_return = round(self.results.strategy.mean() * 252, 3)
            risk = round(self.results.strategy.std() * np.sqrt(252), 3)
            print("Annualized Return: {} | Annualized Risk: {}".format(mean_return, risk))

            cagr = (self.results.cstrategy.iloc[-1] / self.results.cstrategy.iloc[0]) ** (365 / (self.results.index[-1] - self.results.index[0]).days) - 1
            print("CAGR: {}".format(cagr))

            sharpe = (cagr - 0.039) / risk
            print("SHARPE: {}".format(sharpe))
        else:
            return "No data avaliable. Test strategy before calling detailed_metrics()"