import logging
import yfinance as yf
import pandas as pd
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import os

from src.config.settings import CACHE_DIR, LOG_DIR

# Set up logging
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(LOG_DIR, 'data_aggregation.log'),
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataAggregationAgent:
    def __init__(self):
        """Initialize the Data Aggregation Agent."""
        self.cache_dir = CACHE_DIR
        os.makedirs(self.cache_dir, exist_ok=True)
        logger.info("Data Aggregation Agent initialized")

    def fetch_stock_data(
        self,
        ticker: str,
        period: str = "1y",
        interval: str = "1d",
        force_refresh: bool = False
    ) -> pd.DataFrame:
        """
        Fetch historical stock data using yfinance.
        
        Args:
            ticker: Stock ticker symbol
            period: Time period to fetch (e.g., "1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max")
            interval: Data interval (e.g., "1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo")
            force_refresh: Whether to force a refresh of the data instead of using cached data
            
        Returns:
            DataFrame containing the stock data
        """
        cache_file = os.path.join(self.cache_dir, f"{ticker}_{period}_{interval}.csv")
        
        if not force_refresh and os.path.exists(cache_file):
            try:
                data = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                logger.info(f"Loaded cached data for {ticker}")
                return data
            except Exception as e:
                logger.warning(f"Error loading cached data for {ticker}: {str(e)}")
        
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(period=period, interval=interval)
            
            if data.empty:
                logger.warning(f"No data found for {ticker}")
                return pd.DataFrame()
            
            # Save to cache
            data.to_csv(cache_file)
            logger.info(f"Fetched and cached data for {ticker}")
            
            return data
        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {str(e)}")
            return pd.DataFrame()

    def fetch_multiple_stocks(
        self,
        tickers: List[str],
        period: str = "1y",
        interval: str = "1d"
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple stocks.
        
        Args:
            tickers: List of stock ticker symbols
            period: Time period to fetch
            interval: Data interval
            
        Returns:
            Dictionary mapping tickers to their respective DataFrames
        """
        results = {}
        for ticker in tickers:
            data = self.fetch_stock_data(ticker, period, interval)
            if not data.empty:
                results[ticker] = data
        return results

    def get_stock_info(self, ticker: str) -> Dict:
        """
        Get detailed information about a stock.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary containing stock information
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            logger.info(f"Retrieved info for {ticker}")
            return info
        except Exception as e:
            logger.error(f"Error getting info for {ticker}: {str(e)}")
            return {}

    def get_dividends(self, ticker: str) -> pd.DataFrame:
        """
        Get dividend history for a stock.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            DataFrame containing dividend history
        """
        try:
            stock = yf.Ticker(ticker)
            dividends = stock.dividends
            logger.info(f"Retrieved dividends for {ticker}")
            return dividends
        except Exception as e:
            logger.error(f"Error getting dividends for {ticker}: {str(e)}")
            return pd.DataFrame()

if __name__ == "__main__":
    # Example usage
    agent = DataAggregationAgent()
    
    # Fetch data for a single stock
    aapl_data = agent.fetch_stock_data("AAPL")
    print("AAPL Data:")
    print(aapl_data.head())
    
    # Fetch data for multiple stocks
    stocks = ["AAPL", "MSFT", "GOOGL"]
    stock_data = agent.fetch_multiple_stocks(stocks)
    print("\nMultiple Stocks Data:")
    for ticker, data in stock_data.items():
        print(f"\n{ticker} Data:")
        print(data.head())
    
    # Get stock info
    aapl_info = agent.get_stock_info("AAPL")
    print("\nAAPL Info:")
    print(aapl_info)
    
    # Get dividends
    aapl_dividends = agent.get_dividends("AAPL")
    print("\nAAPL Dividends:")
    print(aapl_dividends.head()) 