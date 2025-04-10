import logging
import pandas as pd
from typing import List, Dict, Optional, Any
import os

from src.config.settings import CACHE_DIR, LOG_DIR
from src.agents.data_aggregation_agent import DataAggregationAgent
from src.llm.base_llm_client import BaseLLMClient

# Set up logging
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(LOG_DIR, 'universe_definition_agent.log'),
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class UniverseDefinitionAgent:
    def __init__(self):
        """Initialize the universe definition agent with LLM client."""
        self.data_agent = DataAggregationAgent()
        self.cache_dir = CACHE_DIR
        os.makedirs(self.cache_dir, exist_ok=True)
        self.llm_client = BaseLLMClient("universe_agent")
        logger.info("Universe definition agent initialized with LLM client")

    def update_llm_config(self, new_config: Dict[str, Any]) -> None:
        """Update the LLM configuration for the universe definition agent."""
        self.llm_client.update_config(new_config)
        logger.info(f"Updated LLM configuration: {new_config}")

    async def define_universe(self, stock_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Define a universe of stocks based on the given data."""
        try:
            # Prepare the prompt with stock data
            prompt = self._prepare_universe_prompt(stock_data)
            
            # Generate the universe definition using LLM
            definition = await self.llm_client.generate(prompt)
            
            # Parse and structure the definition
            structured_definition = self._parse_universe_definition(definition)
            
            logger.info("Successfully defined universe")
            return structured_definition
            
        except Exception as e:
            logger.error(f"Error defining universe: {str(e)}")
            raise

    def _prepare_universe_prompt(self, stock_data: Dict[str, pd.DataFrame]) -> str:
        """Prepare the universe definition prompt with stock data."""
        # Extract key metrics from the stock data
        metrics = self._extract_stock_metrics(stock_data)
        
        prompt = f"""
        Define a universe of stocks based on the following data:
        
        Stock Metrics:
        {metrics}
        
        Please provide:
        1. Universe criteria and filters
        2. Rationale for inclusion/exclusion
        3. Expected universe characteristics
        4. Risk considerations
        5. Monitoring parameters
        """
        
        return prompt

    def _extract_stock_metrics(self, stock_data: Dict[str, pd.DataFrame]) -> str:
        """Extract key metrics from the stock data."""
        metrics = []
        for ticker, data in stock_data.items():
            metrics.append(f"\n{ticker}:")
            metrics.append(f"  - Current Price: {data['Close'].iloc[-1]:.2f}")
            metrics.append(f"  - Market Cap: {data.get('MarketCap', 'N/A')}")
            metrics.append(f"  - P/E Ratio: {data.get('PE_Ratio', 'N/A')}")
            metrics.append(f"  - Volume: {data['Volume'].mean():.2f}")
        
        return "\n".join(metrics)

    def _parse_universe_definition(self, definition: str) -> Dict[str, Any]:
        """Parse the LLM-generated universe definition into a structured format."""
        sections = {
            "criteria": "",
            "rationale": "",
            "characteristics": "",
            "risk_considerations": "",
            "monitoring": ""
        }
        
        current_section = None
        for line in definition.split("\n"):
            line = line.strip()
            if not line:
                continue
                
            if "1." in line:
                current_section = "criteria"
            elif "2." in line:
                current_section = "rationale"
            elif "3." in line:
                current_section = "characteristics"
            elif "4." in line:
                current_section = "risk_considerations"
            elif "5." in line:
                current_section = "monitoring"
            elif current_section:
                sections[current_section] += line + "\n"
        
        return sections

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