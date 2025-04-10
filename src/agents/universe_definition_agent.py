import logging
import pandas as pd
from typing import List, Dict, Optional
import os

from src.config.settings import CACHE_DIR, LOG_DIR
from src.agents.data_aggregation_agent import DataAggregationAgent

# Set up logging
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(LOG_DIR, 'universe_definition.log'),
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class UniverseDefinitionAgent:
    def __init__(self):
        """Initialize the Universe Definition Agent."""
        self.data_agent = DataAggregationAgent()
        self.cache_dir = CACHE_DIR
        os.makedirs(self.cache_dir, exist_ok=True)
        logger.info("Universe Definition Agent initialized")

    def filter_stocks(
        self,
        stock_data: pd.DataFrame,
        min_volume: float = 1000000,
        max_pe: float = 30,
        min_price: float = 5.0,
        max_price: float = 1000.0
    ) -> pd.DataFrame:
        """
        Filter stocks based on various criteria.
        
        Args:
            stock_data: DataFrame containing stock data
            min_volume: Minimum average daily volume
            max_pe: Maximum P/E ratio
            min_price: Minimum stock price
            max_price: Maximum stock price
            
        Returns:
            Filtered DataFrame
        """
        try:
            # Ensure required columns exist
            required_columns = ['Volume', 'PE_Ratio', 'Close']
            missing_columns = [col for col in required_columns if col not in stock_data.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")

            # Apply filters
            filtered = stock_data[
                (stock_data['Volume'] >= min_volume) &
                (stock_data['PE_Ratio'] <= max_pe) &
                (stock_data['Close'] >= min_price) &
                (stock_data['Close'] <= max_price)
            ]
            
            logger.info(f"Filtered stocks: {len(filtered)} out of {len(stock_data)}")
            return filtered
        except Exception as e:
            logger.error(f"Error filtering stocks: {str(e)}")
            return pd.DataFrame()

    def create_momentum_universe(
        self,
        tickers: List[str],
        min_volume: float = 1000000,
        min_price: float = 5.0,
        lookback_period: int = 20
    ) -> Dict[str, pd.DataFrame]:
        """
        Create a universe of momentum stocks.
        
        Args:
            tickers: List of stock tickers to consider
            min_volume: Minimum average daily volume
            min_price: Minimum stock price
            lookback_period: Number of days to look back for momentum calculation
            
        Returns:
            Dictionary mapping tickers to their data
        """
        try:
            # Fetch data for all tickers
            stock_data = self.data_agent.fetch_multiple_stocks(tickers)
            
            momentum_universe = {}
            for ticker, data in stock_data.items():
                if len(data) < lookback_period:
                    continue
                    
                # Calculate momentum metrics
                data['Returns'] = data['Close'].pct_change()
                data['Momentum'] = data['Close'].pct_change(periods=lookback_period)
                data['Volume_MA'] = data['Volume'].rolling(window=lookback_period).mean()
                
                # Filter based on criteria
                filtered = data[
                    (data['Volume_MA'] >= min_volume) &
                    (data['Close'] >= min_price) &
                    (data['Momentum'] > 0)  # Positive momentum
                ]
                
                if not filtered.empty:
                    momentum_universe[ticker] = filtered
            
            logger.info(f"Created momentum universe with {len(momentum_universe)} stocks")
            return momentum_universe
        except Exception as e:
            logger.error(f"Error creating momentum universe: {str(e)}")
            return {}

    def create_value_universe(
        self,
        tickers: List[str],
        max_pe: float = 30,
        max_pb: float = 3.0,
        min_dividend_yield: float = 0.02
    ) -> Dict[str, pd.DataFrame]:
        """
        Create a universe of value stocks.
        
        Args:
            tickers: List of stock tickers to consider
            max_pe: Maximum P/E ratio
            max_pb: Maximum Price-to-Book ratio
            min_dividend_yield: Minimum dividend yield
            
        Returns:
            Dictionary mapping tickers to their data
        """
        try:
            value_universe = {}
            for ticker in tickers:
                # Get stock info
                info = self.data_agent.get_stock_info(ticker)
                
                if not info:
                    continue
                
                # Check if required metrics are available
                if all(metric in info for metric in ['trailingPE', 'priceToBook', 'dividendYield']):
                    pe = info.get('trailingPE', float('inf'))
                    pb = info.get('priceToBook', float('inf'))
                    div_yield = info.get('dividendYield', 0)
                    
                    # Apply value criteria
                    if (pe <= max_pe and 
                        pb <= max_pb and 
                        div_yield >= min_dividend_yield):
                        
                        # Get historical data
                        data = self.data_agent.fetch_stock_data(ticker)
                        if not data.empty:
                            value_universe[ticker] = data
            
            logger.info(f"Created value universe with {len(value_universe)} stocks")
            return value_universe
        except Exception as e:
            logger.error(f"Error creating value universe: {str(e)}")
            return {}

    def create_growth_universe(
        self,
        tickers: List[str],
        min_revenue_growth: float = 0.15,
        min_earnings_growth: float = 0.10
    ) -> Dict[str, pd.DataFrame]:
        """
        Create a universe of growth stocks.
        
        Args:
            tickers: List of stock tickers to consider
            min_revenue_growth: Minimum revenue growth rate
            min_earnings_growth: Minimum earnings growth rate
            
        Returns:
            Dictionary mapping tickers to their data
        """
        try:
            growth_universe = {}
            for ticker in tickers:
                # Get stock info
                info = self.data_agent.get_stock_info(ticker)
                
                if not info:
                    continue
                
                # Check if required metrics are available
                if all(metric in info for metric in ['revenueGrowth', 'earningsGrowth']):
                    revenue_growth = info.get('revenueGrowth', 0)
                    earnings_growth = info.get('earningsGrowth', 0)
                    
                    # Apply growth criteria
                    if (revenue_growth >= min_revenue_growth and 
                        earnings_growth >= min_earnings_growth):
                        
                        # Get historical data
                        data = self.data_agent.fetch_stock_data(ticker)
                        if not data.empty:
                            growth_universe[ticker] = data
            
            logger.info(f"Created growth universe with {len(growth_universe)} stocks")
            return growth_universe
        except Exception as e:
            logger.error(f"Error creating growth universe: {str(e)}")
            return {}

if __name__ == "__main__":
    # Example usage
    agent = UniverseDefinitionAgent()
    
    # Example tickers
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "JPM", "V", "WMT"]
    
    # Create different universes
    momentum_universe = agent.create_momentum_universe(tickers)
    print("\nMomentum Universe:")
    for ticker in momentum_universe:
        print(f"- {ticker}")
    
    value_universe = agent.create_value_universe(tickers)
    print("\nValue Universe:")
    for ticker in value_universe:
        print(f"- {ticker}")
    
    growth_universe = agent.create_growth_universe(tickers)
    print("\nGrowth Universe:")
    for ticker in growth_universe:
        print(f"- {ticker}") 