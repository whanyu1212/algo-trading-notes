import pandas as pd
import numpy as np
import yfinance as yf
import warnings

warnings.filterwarnings("ignore")
from typing import Union, Tuple
from scipy.optimize import brute
from datetime import datetime, timedelta


class MomemtumVectorTester:
    """
    This class is responsible for backtesting a momentum trading strategy on a given stock symbol
    over a specified time period.

    Attributes:
    symbol (str): The ticker symbol for the stock to be tested.
    start (str): The start date for the period to be tested.
    end (str): The end date for the period to be tested.
    interval (str): The interval for the data to be fetched. Default is '1d'.
    amount (float): The initial amount of money to be invested.
    momentum (int): The momentum period to be used in the strategy.
    transaction_cost (float): The cost per transaction.
    results (DataFrame): The results of the backtest.
    data (DataFrame): The historical data for the stock.

    Methods:
    get_data(): Fetches the historical data for the stock.
    backtest_strategy(): Backtests the momentum trading strategy on the data.
    plot_results(): Plots the cumulative returns of the strategy and the stock.
    update_momentum_and_backtest(momentum): Updates the momentum period and backtests the strategy.
    optimize_momentum(range): Optimizes the momentum period to maximize the strategy's returns.
    """

    def __init__(
        self,
        symbol: str,
        start: str,
        end: str,
        amount: float,
        transaction_cost: float,
        momentum: int,
        interval: str = "1d",
    ):
        """Initializes the MomentumVectorTester class with the given parameters

        Args:
            symbol (str): the ticker symbol for the stock to be tested
            start (str): the start date for the period to be tested
            end (str): the end date for the period to be tested
            amount (float): the initial amount of money to be invested
            transaction_cost (float): the proportional cost per transaction
            momentum (int): the window period for the momentum calculation
            interval (str, optional): interval of data that we are fetching.
            Defaults to "1d".

        Raises:
            ValueError: if the dates are not valid
            ValueError: if the interval parameter is 1d, we have to limit
            the data to last 30 days only
        """
        try:
            start_date = pd.to_datetime(start)
            end_date = pd.to_datetime(end)
        except ValueError:
            raise ValueError("Invalid start or end date.")

        if interval == "1m":
            thirty_days_ago = datetime.now() - timedelta(days=30)
            if start_date < thirty_days_ago or end_date < thirty_days_ago:
                raise ValueError(
                    "For '1m' interval, data can only be fetched for the past 30 days."
                )

        self.symbol = symbol
        self.start = start
        self.end = end
        self.interval = interval
        self.amount = amount
        self.momentum = momentum
        self.transaction_cost = transaction_cost
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

    def calculate_position(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculates the position of the strategy based on the momentum
        of the stock's returns. Whether it is buying or selling.
        If the momentum is positive, the position is 1, otherwise -1.

        Args:
            data (pd.DataFrame): dataframe without the position column

        Returns:
            pd.DataFrame: dataframe with the position column added
        """
        data["position"] = np.sign(data["Return"].rolling(self.momentum).mean())
        return data

    def apply_strategy(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create a new column in the dataframe called strategy
        which contains the returns of the strategy. It is calculated
        by multiplying the position by the returns of the stock.

        Args:
            data (pd.DataFrame): initial dataframe without the strategy column

        Returns:
            pd.DataFrame: dataframe with the strategy column added
        """
        data["strategy"] = data["position"].shift(1) * data["Return"]
        trades = data["position"].diff().fillna(0) != 0
        data.loc[trades, "strategy"] -= (
            self.transaction_cost * data.loc[trades, "strategy"]
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
        data["cumulative_strategy"] = self.amount * data["strategy"].cumsum().apply(
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
        """Backtests the momentum trading strategy by
        execute the methods in the correct order.

        Returns:
            Tuple[float, float]: absolute performance and outperformance of the strategy
        """
        temp_data = self.data.copy().dropna()
        temp_data = self.calculate_position(temp_data)
        temp_data = self.apply_strategy(temp_data)
        temp_data = self.calculate_cumulative_returns(temp_data)
        self.results = temp_data
        return self.calculate_strategy_performance(temp_data)

    def plot_results(self):
        """plots the cumulative returns of the stock and the strategy"""

        if self.results is None:
            print("No results to plot yet. Run a strategy.")
        title = "%s | TC = %.4f" % (self.symbol, self.transaction_cost)
        self.results[["cumulative_returns", "cumulative_strategy"]].plot(
            title=title, figsize=(10, 6)
        )

    def update_momentum_and_backtest(self, momentum: np.array) -> float:
        """Updates the momentum period and backtests the strategy.

        Args:
            momentum (int): the new momentum period to be used

        Returns:
            float: the negative absolute performance of the strategy that
            will be minimized by the optimization function
        """
        momentum = int(momentum[0])  # extract integer from numpy array
        self.momentum = momentum
        return -self.backtest_strategy()[0]

    def optimize_momentum(self, range: tuple) -> Tuple[float, float]:
        """Optimizes the momentum period to maximize the strategy's returns.

        Args:
            range (tuple): the range of values and step size to be tested
            for the momentum period

        Returns:
            Tuple[float, float]: the optimal momentum period and the strategy's
            absolute performance at that period
        """
        range = (range[0], range[1], 1)
        # by default the brute function hanles 2 dimensional search
        # we need to pass a tuple of ranges for each parameter
        # however, we only have one parameter to optimize
        # so we need to use indexing to extract the first element
        opt = brute(self.update_momentum_and_backtest, (range,))
        return opt, -self.update_momentum_and_backtest(opt)


if __name__ == "__main__":
    tester = MomemtumVectorTester(
        symbol="AAPL",
        start="2023-01-01",
        end="2024-01-01",
        amount=1000,
        transaction_cost=0.0,
        momentum=3,
    )
    print(tester.optimize_momentum((1, 21, 1)))
