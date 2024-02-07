import time
import yfinance as yf
import pandas as pd
import warnings
import numpy as np
from typing import List, Optional
from loguru import logger

warnings.filterwarnings("ignore")


class BASS:
    """
    A class function that calculates Beta, Alpha, Standard Deviation, and Sharpe Ratio for a list of tickers
    """

    def __init__(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        benchmark_ticker: str = "^STI",
        interval: str = "1d",
        risk_free_rate: float = 0.0,
    ):
        """Initializes the BASS class

        Args:
            tickers (List[str]): A list of tickers to calculate BASS for
            start_date (str): left bound of the date range (inclusive)
            end_date (str): right bound of the date range (exclusive)
            benchmark_ticker (str, optional): Benchmark ticker that is used to represent the market. Defaults to "^STI".
            interval (str, optional): Interval of the pricing data. Defaults to "1d".
            risk_free_rate (float, optional): risk free rate for this calculation. Defaults to 0.0.
        """

        # Input validation

        # Check that tickers is a list of strings
        if not isinstance(tickers, list) or not all(
            isinstance(ticker, str) for ticker in tickers
        ):
            raise ValueError("tickers must be a list of strings")

        # Check that start_date and end_date are strings and can be parsed into datetime objects
        try:
            pd.to_datetime(start_date)
            pd.to_datetime(end_date)
        except ValueError:
            raise ValueError("start_date and end_date must be valid date strings")

        # Check that benchmark_ticker is a string
        if not isinstance(benchmark_ticker, str):
            raise ValueError("benchmark_ticker must be a string")

        # Check that interval is a string and is one of the allowed values
        if not isinstance(interval, str) or interval not in ["1d", "1wk", "1mo"]:
            raise ValueError("interval must be a string and one of '1d', '1wk', '1mo'")

        # Check that risk_free_rate is a float
        if not isinstance(risk_free_rate, float):
            raise ValueError("risk_free_rate must be a float")

        self.tickers = tickers
        self.benchmark_ticker = benchmark_ticker
        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval
        self.risk_free_rate = risk_free_rate
        logger.info(
            f"Initiliazing BASS calculation for {self.tickers} "
            f"from {self.start_date} to {self.end_date}"
        )
        self.data = self.get_data()
        self.returns = self.calculate_daily_returns()

    def get_data(self, retries=3, delay=5) -> pd.DataFrame:
        """Download data using the yfinance library
        Args:
            retries (int, optional): retry mechanism. Defaults to 3.
            delay (int, optional): time delay between consecutive retries. Defaults to 5.

        Returns:
            pd.DataFrame: Dataframe that contains each ticker's closing price
        """
        for i in range(retries):
            try:
                logger.info(
                    "Downloading data for tickers: {}",
                    self.tickers + [self.benchmark_ticker],
                )
                data = yf.download(
                    self.tickers + [self.benchmark_ticker],
                    start=self.start_date,
                    end=self.end_date,
                    interval=self.interval,
                )["Close"]
                # one of the tickers have been delisted
                # drop the column
                data = data.dropna(axis=1, how="all")

                # update ticker list
                # skip the dropped ticker
                self.tickers = [
                    ticker
                    for ticker in data.columns.tolist()
                    if ticker != self.benchmark_ticker
                ]
                if not data.empty:
                    logger.info("Data downloaded successfully")
                else:
                    logger.error("Data download failed.")
                return data
            except Exception as e:
                logger.error("An error occurred during data download: %s", str(e))
                if i < retries - 1:  # i is zero indexed
                    time.sleep(delay)  # wait before trying again
                    continue
                else:
                    raise  # re-raise the last exception if all retries fail

    def calculate_daily_returns(self) -> Optional[pd.DataFrame]:
        if self.data is None or self.data.empty:
            logger.error("No data available for calculating returns")
            return None

        try:
            logger.info("Calculating daily returns for tickers: {}", self.tickers)
            return_data = self.data.pct_change().dropna()
            return return_data
        except Exception as e:
            logger.error("An error occurred during return calculation: %s", str(e))
            raise

    def calculate_annualized_return(self, returns: pd.Series, ticker: str) -> float:
        """Calculate the annualized return for a given ticker using the CAGR formula,
            which considers compounding effect over a period of time.

        Args:
            returns (pd.Series): column of daily returns for a given ticker
            ticker (str): the name of the ticker

        Returns:
            float: a single value that represents the annualized return
        """
        try:
            logger.info("Calculating annualized return for %s", ticker)
            return np.prod(1 + returns) ** (252 / len(returns)) - 1
        except Exception as e:
            logger.error(
                "An error occurred during annualized return calculation: %s", e
            )
            raise

    def calculate_beta(self, ticker: str) -> float:
        """_summary_

        Args:
            ticker (str): ticker symbol that we want to calculate beta for

        Raises:
            KeyError: if ticker column not found
            KeyError: if the benchmark ticker column not found
            ValueError: cases of singular covariance matrix

        Returns:
            float: beta value
        """
        if ticker not in self.returns:
            raise KeyError(f"Ticker {ticker} not found in returns.")
        if self.benchmark_ticker not in self.returns:
            raise KeyError(
                f"Benchmark ticker {self.benchmark_ticker} not found in returns."
            )

        logger.info("Calculating beta for ticker: {}", ticker)
        cov = np.cov(self.returns[ticker], self.returns[self.benchmark_ticker])

        # if the covariance matrix is singular, the determinant will be zero
        # that means the stock is perfectly correlated with the market
        # in this case, the beta will be 1
        # which will not be meaningful for the calculation
        if np.linalg.det(cov) == 0:
            raise ValueError("Covariance matrix is singular.")

        return cov[0][1] / cov[1][1]

    def calculate_alpha(self, ticker: str) -> float:
        """Calculate the alpha for a given ticker

        Args:
            ticker (str): ticker symbol that we want to calculate alpha for

        Returns:
            float: alpha value
        """
        logger.info("Calculating alpha for ticker: {}", ticker)
        try:
            try:
                beta = self.calculate_beta(ticker)
            except ValueError as e:
                logger.error(
                    "Failed to calculate beta for ticker %s: %s", ticker, str(e)
                )
                return None
            annualized_return = self.calculate_annualized_return(
                self.returns[ticker], ticker
            )
            annualized_benchmark_return = self.calculate_annualized_return(
                self.returns[self.benchmark_ticker], ticker
            )
            alpha = annualized_return - (
                self.risk_free_rate
                + beta * (annualized_benchmark_return - self.risk_free_rate)
            )
            return alpha
        except ZeroDivisionError:
            logger.error(
                "ZeroDivisionError occurred during alpha calculation for ticker: %s",
                ticker,
            )
            return None

    def calculate_sharpe(self, ticker: str) -> float:
        """Calculate the Sharpe ratio for a given ticker

        Args:
            ticker (str): ticker symbol that we want to calculate Sharpe ratio for

        Returns:
            float: Sharpe ratio value
        """
        logger.info("Calculating sharpe ratio for ticker: {}", ticker)
        try:
            annualized_return = self.calculate_annualized_return(
                self.returns[ticker], ticker
            )
            individual_tick_std = self.returns[ticker].std() * np.sqrt(252)
            return (annualized_return - self.risk_free_rate) / individual_tick_std
        except ZeroDivisionError:
            return None

    def tabulate_BASS(self) -> pd.DataFrame:
        """Tabulate the BASS metrics for each ticker
           Loop through each ticker and append values
           to the respective columns in the dataframe

        Returns:
            pd.DataFrame: output dataframe
        """
        metrics = pd.DataFrame(index=self.tickers)

        for ticker in self.tickers:
            try:
                metrics.loc[ticker, "Return"] = self.calculate_annualized_return(
                    self.returns[ticker], ticker
                )
                metrics.loc[ticker, "Standard Deviation"] = self.returns[
                    ticker
                ].std() * np.sqrt(252)
                metrics.loc[ticker, "Covariance with Market"] = np.cov(
                    self.returns[ticker], self.returns[self.benchmark_ticker]
                )[0][1]
                try:
                    metrics.loc[ticker, "Beta"] = self.calculate_beta(ticker)
                except ValueError as e:
                    logger.error(
                        "Failed to calculate beta for ticker %s: %s", ticker, str(e)
                    )
                    continue
                metrics.loc[ticker, "Alpha"] = self.calculate_alpha(ticker)
                metrics.loc[ticker, "Sharpe Ratio"] = self.calculate_sharpe(ticker)
            except ZeroDivisionError as e:
                logger.error(f"ZeroDivisionError for {ticker} - {e}")
                continue
            except Exception as e:
                logger.error(f"Unexpected error for {ticker} - {e}")
                continue
        return metrics


if __name__ == "__main__":
    # store the tickers in cfg file if possible
    all_stocks = [
        "C52.SI",
        "T39.SI",
        "S68.SI",
        "G13.SI",
        "V03.SI",
        "U11.SI",
        "C07.SI",
        "D05.SI",
        "Z74.SI",
        "D01.SI",
        "O39.SI",
        "S63.SI",
        "A17U.SI",
        "BN4.SI",
        "BS6.SI",
        "M44U.SI",
        "H78.SI",
        "Y92.SI",
        "C38U.SI",
        "U14.SI",
        "N2IU.SI",
        "F34.SI",
        "C09.SI",
        "J36.SI",
        "S58.SI",
        "C6L.SI",
        "U96.SI",
        "1810.HK",
        "9999.HK",
        "7500.HK",
        "9618.HK",
        "1024.HK",
        "3690.HK",
        "6618.HK",
    ]

    # store the dates in cfg file if possible
    start_date = "2023-01-01"
    end_date = "2024-01-01"
    bass = BASS(all_stocks, start_date, end_date)
    results = bass.tabulate_BASS()
    print(results)
