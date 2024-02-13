import yfinance as yf
import pandas as pd
import numpy as np
from typing import Union, List, Tuple


class TechnicalAnalysisIndicators:
    def __init__(
        self, symbol: str, start: str, end: str, interval: str = "1d", window: int = 14
    ):
        """Initializes the TechnicalAnalysisIndicators class with the given parameters.

        Args:
            symbol (str): The ticker symbol that we are interested in
            start (str): start date for the period to be tested
            end (str): end date for the period to be tested
            interval (str, optional): frequency of the data fetched. Defaults to "1d".
            window (int, optional): window period for the calculation of indicators. Defaults to 14.
        """
        try:
            pd.to_datetime(start)
            pd.to_datetime(end)
        except ValueError:
            raise ValueError("Invalid start or end date.")

        if start > end:
            raise ValueError("Start date should be before end date.")

        self.symbol = symbol
        self.start = start
        self.end = end
        self.interval = interval
        self.window = window
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
            ticker_data = yf.download(
                self.symbol, start=self.start, end=self.end, interval=self.interval
            )
            if ticker_data.empty:
                raise ValueError("Invalid ticker symbol.")
            return ticker_data

        except Exception as e:
            print(f"Error getting data: {e}")
            return None

    def calculate_price_change(self) -> pd.DataFrame:
        """Calculate changes in price from the previous day.

        Returns:
            pd.DataFrame: dataframe with the added column
        """
        self.data["Price Change"] = self.data["Close"].diff(1)
        return self.data

    def calculate_RSI(self) -> pd.DataFrame:
        """Calculate the RSI given the window period.

        Returns:
            pd.DataFrame: dataframe with the added column
        """
        gain = self.data["Price Change"].mask(self.data["Price Change"] < 0, 0)
        loss = -self.data["Price Change"].mask(self.data["Price Change"] > 0, 0)

        avg_gain = gain.rolling(window=self.window).mean()
        avg_loss = loss.rolling(window=self.window).mean()

        RS = avg_gain / avg_loss
        self.data["RSI"] = 100 - (100 / (1 + RS))
        return self.data

    def calculate_SMA(self) -> pd.DataFrame:
        """Calculate the simple moving average given the window period.

        Returns:
            pd.DataFrame: dataframe with the added column
        """
        self.data = self.data.assign(SMA=lambda x: x.Close.rolling(self.window).mean())
        return self.data

    def calculate_ATR(self) -> pd.DataFrame:
        """Calculate the average true range given the window period.

        Returns:
            pd.DataFrame: dataframe with the added columns
        """
        self.data = self.data.assign(
            HLC=(self.data["High"] - self.data["Low"]),
            HL=(self.data["High"] - self.data["Close"].shift(1)).abs(),
            LC=(self.data["Low"] - self.data["Close"].shift(1)).abs(),
        )
        self.data["TR"] = self.data[["HLC", "HL", "LC"]].max(axis=1)
        self.data["ATR"] = self.data["TR"].rolling(self.window).mean()
        return self.data

    def calculation_flow(self) -> pd.DataFrame:
        """Orchestrate the calculation of the technical analysis indicators.

        Returns:
            pd.DataFrame: dataframe with the added columns
        """
        self.data = self.calculate_price_change()
        self.data = self.calculate_RSI()
        self.data = self.calculate_SMA()
        self.data = self.calculate_ATR()

        return self.data


if __name__ == "__main__":
    symbol = "AAPL"
    start = "2023-01-01"
    end = "2024-01-01"
    interval = "1d"
    window = 14
    technical_analysis = TechnicalAnalysisIndicators(
        symbol, start, end, interval, window
    )
    output = technical_analysis.calculation_flow()
    print(output.filter(items=["Close", "Price Change", "RSI", "SMA", "ATR"]))
