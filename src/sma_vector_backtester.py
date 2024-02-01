import numpy as np
import pandas as pd
import yfinance as yf
import warnings
import time

warnings.filterwarnings("ignore")
from scipy.optimize import brute


class SMAVectorBacktester:
    def __init__(self, symbol, SMA1, SMA2, start, end):
        if not isinstance(SMA1, int) or not isinstance(SMA2, int):
            raise ValueError("SMA1 and SMA2 should be integers.")
        if SMA1 >= SMA2:
            raise ValueError("SMA1 should be less than SMA2.")
        try:
            pd.to_datetime(start)
            pd.to_datetime(end)
        except ValueError:
            raise ValueError("Invalid start or end date.")
        self.symbol = symbol
        self.SMA1 = SMA1
        self.SMA2 = SMA2
        self.start = start
        self.end = end
        self.results = None
        self.data = self.get_data()

    def get_data(self, period="1d"):
        try:
            ticker = yf.Ticker(self.symbol)
            ticker_data = (
                ticker.history(period=period, start=self.start, end=self.end)
                .reset_index()
                .assign(Symbol=self.symbol)
                .rename(columns={"Close": "Price"})
                .assign(Return=lambda x: np.log(x.Price / x.Price.shift(1)))
                .assign(SMA1=lambda x: x.Price.rolling(self.SMA1).mean())
                .assign(SMA2=lambda x: x.Price.rolling(self.SMA2).mean())
            )
            if ticker_data.empty:
                raise ValueError("Invalid ticker symbol.")
            return ticker_data
        except Exception as e:
            print(f"Error getting data: {e}")
            return None

    def update_SMA_parameters(self, SMA1=None, SMA2=None):
        if SMA1 is not None:
            if not isinstance(SMA1, int):
                raise ValueError("SMA1 should be an integer.")
            self.SMA1 = SMA1
        if SMA2 is not None:
            if not isinstance(SMA2, int):
                raise ValueError("SMA2 should be an integer.")
            self.SMA2 = SMA2
        self.data = self.get_data()

    def backtest_strategy(self):
        temp_data = self.data.copy().dropna()
        temp_data = temp_data.assign(
            Position=np.where(temp_data["SMA1"] > temp_data["SMA2"], 1, -1)
        )
        temp_data = temp_data.assign(
            Strategy=temp_data["Position"].shift(1) * temp_data["Return"]
        )
        temp_data = (
            temp_data.dropna()
            .assign(Cumulative_Returns=lambda x: np.exp(x["Return"].cumsum()))
            .assign(Cumulative_Strategy=lambda x: np.exp(x["Strategy"].cumsum()))
        )

        self.results = temp_data  # its a dataframe

        strategy_performance = temp_data["Cumulative_Strategy"].iloc[-1]

        strategy_outperformance = (
            strategy_performance - temp_data["Cumulative_Returns"].iloc[-1]
        )
        return round(strategy_performance, 2), round(strategy_outperformance, 2)

    def plot_results(self):
        if self.results is None:
            print("No results to plot yet. Run a strategy.")
        title = "%s | SMA1=%d, SMA2=%d" % (self.symbol, self.SMA1, self.SMA2)
        self.results[["Cumulative_Returns", "Cumulative_Strategy"]].plot(
            title=title, figsize=(10, 6)
        )

    def update_parameters_and_backtest(self, SMA):
        self.update_SMA_parameters(int(SMA[0]), int(SMA[1]))
        return -self.backtest_strategy()[0]

    def optimize_parameters(self, SMA1_range, SMA2_range):
        # read up on brute function
        opt = brute(
            self.update_parameters_and_backtest, (SMA1_range, SMA2_range), finish=None
        )
        return opt, -self.update_parameters_and_backtest(opt)


if __name__ == "__main__":
    smabt = SMAVectorBacktester("AAPL", 42, 252, "2010-1-1", "2020-12-31")
    print(smabt.backtest_strategy())
    smabt.update_SMA_parameters(SMA1=20, SMA2=100)
    print(smabt.backtest_strategy())
    start_time = time.time()
    print(smabt.optimize_parameters((42, 56, 1), (252, 300, 1)))
    end_time = time.time()
    print(f"Time taken to optimize: {end_time - start_time} seconds")
