import time
import yfinance as yf
import pandas as pd
import warnings
import numpy as np
from typing import List
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

    def get_data(self, retries=3, delay=5):
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

    def calculate_daily_returns(self):
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

    def calculate_annualized_return(self, returns: pd.Series, ticker: str):
        """
        This method calculates the annualized return given a series of returns
        """
        try:
            logger.info("Calculating annualized return for %s", ticker)
            return np.prod(1 + returns) ** (252 / len(returns)) - 1
        except Exception as e:
            logger.error(
                "An error occurred during annualized return calculation: %s", e
            )
            raise

    def calculate_beta(self, ticker: str):
        if ticker not in self.returns:
            raise KeyError(f"Ticker {ticker} not found in returns.")
        if self.benchmark_ticker not in self.returns:
            raise KeyError(
                f"Benchmark ticker {self.benchmark_ticker} not found in returns."
            )

        logger.info("Calculating beta for ticker: {}", ticker)
        cov = np.cov(self.returns[ticker], self.returns[self.benchmark_ticker])

        if np.linalg.det(cov) == 0:
            raise ValueError("Covariance matrix is singular.")

        return cov[0][1] / cov[1][1]

    def calculate_alpha(self, ticker: str):
        logger.info("Calculating alpha for ticker: {}", ticker)
        try:
            beta = self.calculate_beta(ticker)
            annualized_return = self.calculate_annualized_return(self.returns[ticker])
            annualized_benchmark_return = self.calculate_annualized_return(
                self.returns[self.benchmark_ticker]
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

    def calculate_sharpe(self, ticker: str):
        logger.info("Calculating sharpe ratio for ticker: {}", ticker)
        try:
            annualized_return = self.calculate_annualized_return(self.returns[ticker])
            individual_tick_std = self.returns[ticker].std() * np.sqrt(252)
            return (annualized_return - self.risk_free_rate) / individual_tick_std
        except ZeroDivisionError:
            return None

    def tabulate_BASS(self):
        metrics = pd.DataFrame(index=self.tickers)
        for ticker in self.tickers:
            try:
                metrics.loc[ticker, "Return"] = self.calculate_annualized_return(
                    self.returns[ticker]
                )
                metrics.loc[ticker, "Standard Deviation"] = self.returns[
                    ticker
                ].std() * np.sqrt(252)
                metrics.loc[ticker, "Covariance with Market"] = np.cov(
                    self.returns[ticker], self.returns[self.benchmark_ticker]
                )[0][1]
                metrics.loc[ticker, "Beta"] = self.calculate_beta(ticker)
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

    start_date = "2023-01-01"
    end_date = "2024-01-01"
    bass = BASS(all_stocks, start_date, end_date)
    results = bass.tabulate_BASS()
    print(results)
