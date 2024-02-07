import unittest
from src.BASS import BASS
import pandas as pd
import numpy as np


class TestBASS(unittest.TestCase):
    def setUp(self):
        self.tickers = ["AAPL", "GOOG"]
        self.start_date = "2020-01-01"
        self.end_date = "2021-01-01"
        self.bass = BASS(self.tickers, self.start_date, self.end_date)

    def test_get_data(self):
        data = self.bass.get_data()
        self.assertIsNotNone(data)
        self.assertEqual(
            list(data.columns), self.tickers + [self.bass.benchmark_ticker]
        )
        self.assertTrue(isinstance(data.index, pd.DatetimeIndex))

    def test_calculate_daily_returns(self):
        returns = self.bass.calculate_daily_returns()
        self.assertIsNotNone(returns)
        self.assertEqual(
            list(returns.columns), self.tickers + [self.bass.benchmark_ticker]
        )
        self.assertTrue(isinstance(returns.index, pd.DatetimeIndex))
        self.assertTrue(
            (returns <= 1).all().all()
        )  # returns should be less than or equal to 1
        self.assertTrue(
            (returns >= -1).all().all()
        )  # returns should be greater than or equal to -1

    def test_calculate_annualized_return(self):
        for ticker in self.tickers:
            returns = self.bass.calculate_daily_returns()[ticker]
            annualized_return = self.bass.calculate_annualized_return(returns, ticker)
            self.assertIsNotNone(annualized_return)
            self.assertTrue(isinstance(annualized_return, float))
            self.assertTrue(
                annualized_return <= np.prod(1 + returns) ** (252 / len(returns))
            )  # should not be greater than geometric mean

    def test_calculate_beta(self):
        for ticker in self.tickers:
            beta = self.bass.calculate_beta(ticker)
            self.assertIsNotNone(beta)
            self.assertTrue(isinstance(beta, float))

    def test_calculate_alpha(self):
        for ticker in self.tickers:
            alpha = self.bass.calculate_alpha(ticker)
            self.assertIsNotNone(alpha)
            self.assertTrue(isinstance(alpha, float))

    def test_calculate_sharpe(self):
        for ticker in self.tickers:
            sharpe = self.bass.calculate_sharpe(ticker)
            self.assertIsNotNone(sharpe)
            self.assertTrue(isinstance(sharpe, float))

    def test_tabulate_BASS(self):
        metrics = self.bass.tabulate_BASS()
        self.assertIsNotNone(metrics)
        self.assertEqual(list(metrics.index), self.tickers)
        self.assertEqual(
            set(metrics.columns),
            set(
                [
                    "Return",
                    "Covariance with Market",
                    "Standard Deviation",
                    "Beta",
                    "Alpha",
                    "Sharpe Ratio",
                ]
            ),
        )


if __name__ == "__main__":
    unittest.main()
