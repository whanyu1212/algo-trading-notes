import yfinance as yf
import pandas as pd
import numpy as np
import warnings
from typing import Union, List, Tuple
from scipy.optimize import minimize_scalar

warnings.filterwarnings("ignore")


class MeanReversionVectorBacktestor:
    def __init__(
        self,
        symbol: str,
        start: str,
        end: str,
        amount: float,
        transaction_cost: float,
        threshold: float,
        window: int,
        interval: str = "1d",
    ):
        """Initializes the MeanReversionVectorBacktestor class with the given parameters.

        Args:
            symbol (str): ticker symbol for the stock to be tested
            start (str): start date for the period to be tested
            end (str): end date for the period to be tested
            amount (float): initial amount of money to be invested
            transaction_cost (float): the proportional cost per transaction
            window (int): window period for the calculation of the mean and standard deviation
            interval (str, optional): frequency of the data to be fetched. Defaults to "1d".

        Raises:
            ValueError: _description_
        """
        try:
            pd.to_datetime(start)
            pd.to_datetime(end)
        except ValueError:
            raise ValueError("Invalid start or end date.")
        self.symbol = symbol
        self.start = start
        self.end = end
        self.amount = amount
        self.threshold = threshold
        self.transaction_cost = transaction_cost
        self.window = window
        self.interval = interval
        self.results = None
        self.data = self.get_data()

    def get_data(self) -> Union[pd.DataFrame, None]:
        """Fetches the historical data for the ticker symbol
        from Yahoo Finance using the yfinance library.

        Raises:
            ValueError: if the dataframe is empty

        Returns:
            Union[pd.DataFrame, None]: either the dataframe with the historical data
            or None if an error occurred
        """
        try:
            ticker_data = (
                yf.download(
                    self.symbol, start=self.start, end=self.end, interval=self.interval
                )
                .rename(columns={"Close": "Price"})
                .assign(Return=lambda x: np.log(x.Price / x.Price.shift(1)))
            )
            if ticker_data.empty:
                raise ValueError("Invalid ticker symbol.")
            return ticker_data

        except Exception as e:
            print(f"Error getting data: {e}")
            return None

    def assign_SMA(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculates the Simple Moving Average (SMA) of the stock price.

        Args:
            data (pd.DataFrame): Initial dataframe without the SMA column

        Returns:
            pd.DataFrame: dataframe with the SMA column added
        """
        data["SMA"] = data["Price"].rolling(self.window).mean()
        return data

    def assign_distance(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculates the distance of the stock price from the SMA.

        Args:
            data (pd.DataFrame): Initial dataframe without the Distance column

        Returns:
            pd.DataFrame: dataframe with the Distance column added
        """
        data["Distance"] = data["Price"] - data["SMA"]
        return data

    def calculate_position(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculates the position of the strategy based on the distance
        of the stock's price from the SMA. Whether it is buying or selling.
        If the distance is greater than the threshold, the position is -1,
        if it is less than the negative threshold, the position is 1,
        otherwise it is 0 (no position).

        Args:
            data (pd.DataFrame): Initial dataframe without the position column

        Returns:
            pd.DataFrame: dataframe with the position column added
        """
        try:

            conditions = [
                data["Distance"] > self.threshold,  # sell signals
                data["Distance"] < -self.threshold,  # buy signals
                data["Distance"] * data["Distance"].shift(1)
                < 0,  # crossing of current price and SMA (zero distance)
            ]

            choices = [-1, 1, 0]

            data["position"] = np.select(conditions, choices, default=np.nan)
            data["position"] = data["position"].ffill().fillna(0)
            data["position"] = data["position"].astype(int)
        except Exception as e:
            print(f"Error calculating position: {e}")

        return data

    def apply_strategy(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create a new column in the dataframe called Strategy
        which contains the returns of the strategy. It is calculated
        by multiplying the position by the returns of the stock.

        Returns:
            pd.DataFrame: dataframe with the Strategy column added
        """
        data["Strategy"] = data["position"].shift(1) * data["Return"]
        trades = data["position"].diff().fillna(0) != 0
        data.loc[trades, "Strategy"] -= (
            self.transaction_cost * data.loc[trades, "Strategy"]
        )
        return data

    def calculate_cumulative_returns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculates the cumulative returns of the stock and the strategy
        and unlogarithmizes them.

        Args:
            data (pd.DataFrame): initial dataframe without the cumulative returns columns

        Returns:
            pd.DataFrame: dataframe with the cumulative returns columns added
        """
        data["cumulative_returns"] = self.amount * data["Return"].cumsum().apply(np.exp)
        data["cumulative_strategy"] = self.amount * data["Strategy"].cumsum().apply(
            np.exp
        )
        return data

    def calculate_strategy_performance(self, data: pd.DataFrame) -> Tuple[float, float]:
        """Calculates the performance of the strategy by subtracting the
        cumulative returns of the stock from the cumulative returns of the strategy.

        Args:
            data (pd.DataFrame): dataframe with all the columns added

        Returns:
            Tuple[float, float]: absolute performance and outperformance of the strategy
        """
        aperf = data["cumulative_strategy"].iloc[-1]
        operf = aperf - data["cumulative_returns"].iloc[-1]
        return round(aperf, 2), round(operf, 2)

    def backtest_strategy(self) -> Tuple[float, float]:
        """Backtests the mean reversion trading strategy by
        calculating the position, applying the strategy and
        calculating the cumulative returns and the performance.

        Returns:
            Tuple[float, float]: absolute performance and outperformance of the strategy
        """
        temp_data = self.data.dropna()
        self.assign_SMA(temp_data)
        self.assign_distance(temp_data)
        temp_data.dropna(inplace=True)
        self.calculate_position(temp_data)
        self.apply_strategy(temp_data)
        self.calculate_cumulative_returns(temp_data)
        return self.calculate_strategy_performance(temp_data)

    def plot_results(self):
        """plots the cumulative returns of the stock and the strategy"""

        if self.results is None:
            print("No results to plot yet. Run a strategy.")
        title = "%s | TC = %.4f" % (self.symbol, self.transaction_cost)
        self.results[["cumulative_returns", "cumulative_strategy"]].plot(
            title=title, figsize=(10, 6)
        )

    def update_parameters_and_backtest(self, threshold: float) -> float:
        """Updates the threshold and backtests the strategy.

        Args:
            threshold (float): new threshold to be used

        Returns:
            float: outperformance of the strategy
        """
        self.threshold = threshold
        return -self.backtest_strategy()[0]

    def optimize_parameters(self, threshold_range: List[float]) -> float:
        """Optimizes the threshold parameter by running the strategy
        with different values and finding the one that maximizes the outperformance.

        Args:
            threshold_range (List[float]): range of threshold values to be tested

        Returns:
            float: the optimal threshold value
        """
        opt = minimize_scalar(
            self.update_parameters_and_backtest,
            bounds=threshold_range,
            method="bounded",
        )
        return opt.x, -self.update_parameters_and_backtest(opt.x)


if __name__ == "__main__":
    backtestor = MeanReversionVectorBacktestor(
        symbol="AAPL",
        start="2010-01-01",
        end="2020-01-01",
        amount=10000,
        transaction_cost=0.001,
        threshold=10,
        window=20,
    )
    print(backtestor.backtest_strategy())
    # backtestor.plot_results()
    print(backtestor.optimize_parameters([5, 20]))
