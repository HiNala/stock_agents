import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import os
from datetime import datetime, timedelta

from src.config.settings import CACHE_DIR, LOG_DIR
from src.agents.data_aggregation_agent import DataAggregationAgent
from src.agents.universe_definition_agent import UniverseDefinitionAgent

# Set up logging
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(LOG_DIR, 'research_agent.log'),
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ResearchAgent:
    def __init__(self):
        """Initialize the Research Agent."""
        self.data_agent = DataAggregationAgent()
        self.universe_agent = UniverseDefinitionAgent()
        self.cache_dir = CACHE_DIR
        os.makedirs(self.cache_dir, exist_ok=True)
        logger.info("Research Agent initialized")

    def analyze_universe(
        self,
        universe_data: Dict[str, pd.DataFrame],
        lookback_period: int = 252  # 1 year of trading days
    ) -> Dict[str, Dict]:
        """
        Analyze a universe of stocks and generate a research report.
        
        Args:
            universe_data: Dictionary mapping tickers to their data
            lookback_period: Number of days to look back for analysis
            
        Returns:
            Dictionary containing analysis results for each stock
        """
        try:
            analysis_results = {}
            
            for ticker, data in universe_data.items():
                if len(data) < lookback_period:
                    logger.warning(f"Insufficient data for {ticker}")
                    continue
                
                # Calculate key metrics
                returns = data['Close'].pct_change()
                volatility = returns.std() * np.sqrt(252)  # Annualized volatility
                sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)
                
                # Calculate drawdown
                rolling_max = data['Close'].rolling(window=lookback_period, min_periods=1).max()
                drawdown = (data['Close'] - rolling_max) / rolling_max
                max_drawdown = drawdown.min()
                
                # Calculate volume metrics
                avg_volume = data['Volume'].mean()
                volume_std = data['Volume'].std()
                
                # Get additional info
                info = self.data_agent.get_stock_info(ticker)
                
                analysis_results[ticker] = {
                    'returns': {
                        'daily_mean': returns.mean(),
                        'daily_std': returns.std(),
                        'annualized_volatility': volatility,
                        'sharpe_ratio': sharpe_ratio
                    },
                    'risk_metrics': {
                        'max_drawdown': max_drawdown,
                        'avg_volume': avg_volume,
                        'volume_std': volume_std
                    },
                    'fundamentals': {
                        'pe_ratio': info.get('trailingPE', None),
                        'market_cap': info.get('marketCap', None),
                        'dividend_yield': info.get('dividendYield', None)
                    }
                }
            
            logger.info(f"Completed analysis for {len(analysis_results)} stocks")
            return analysis_results
        except Exception as e:
            logger.error(f"Error analyzing universe: {str(e)}")
            return {}

    def generate_sector_analysis(
        self,
        universe_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Dict]:
        """
        Generate sector-level analysis for the universe.
        
        Args:
            universe_data: Dictionary mapping tickers to their data
            
        Returns:
            Dictionary containing sector analysis results
        """
        try:
            sector_analysis = {}
            
            for ticker, data in universe_data.items():
                # Get sector information
                info = self.data_agent.get_stock_info(ticker)
                sector = info.get('sector', 'Unknown')
                
                if sector not in sector_analysis:
                    sector_analysis[sector] = {
                        'stocks': [],
                        'total_market_cap': 0,
                        'avg_pe_ratio': 0,
                        'avg_volume': 0
                    }
                
                # Update sector metrics
                sector_analysis[sector]['stocks'].append(ticker)
                sector_analysis[sector]['total_market_cap'] += info.get('marketCap', 0)
                sector_analysis[sector]['avg_pe_ratio'] += info.get('trailingPE', 0)
                sector_analysis[sector]['avg_volume'] += data['Volume'].mean()
            
            # Calculate averages
            for sector in sector_analysis:
                num_stocks = len(sector_analysis[sector]['stocks'])
                if num_stocks > 0:
                    sector_analysis[sector]['avg_pe_ratio'] /= num_stocks
                    sector_analysis[sector]['avg_volume'] /= num_stocks
            
            logger.info(f"Completed sector analysis for {len(sector_analysis)} sectors")
            return sector_analysis
        except Exception as e:
            logger.error(f"Error generating sector analysis: {str(e)}")
            return {}

    def generate_correlation_analysis(
        self,
        universe_data: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Generate correlation analysis for the universe.
        
        Args:
            universe_data: Dictionary mapping tickers to their data
            
        Returns:
            DataFrame containing correlation matrix
        """
        try:
            # Extract closing prices
            closing_prices = pd.DataFrame()
            for ticker, data in universe_data.items():
                closing_prices[ticker] = data['Close']
            
            # Calculate returns
            returns = closing_prices.pct_change()
            
            # Calculate correlation matrix
            correlation_matrix = returns.corr()
            
            logger.info("Completed correlation analysis")
            return correlation_matrix
        except Exception as e:
            logger.error(f"Error generating correlation analysis: {str(e)}")
            return pd.DataFrame()

    def generate_research_report(
        self,
        universe_data: Dict[str, pd.DataFrame]
    ) -> Dict:
        """
        Generate a comprehensive research report for the universe.
        
        Args:
            universe_data: Dictionary mapping tickers to their data
            
        Returns:
            Dictionary containing the complete research report
        """
        try:
            # Generate various analyses
            stock_analysis = self.analyze_universe(universe_data)
            sector_analysis = self.generate_sector_analysis(universe_data)
            correlation_matrix = self.generate_correlation_analysis(universe_data)
            
            # Compile the report
            report = {
                'timestamp': datetime.now().isoformat(),
                'universe_size': len(universe_data),
                'stock_analysis': stock_analysis,
                'sector_analysis': sector_analysis,
                'correlation_matrix': correlation_matrix.to_dict(),
                'summary': {
                    'top_performers': self._get_top_performers(stock_analysis),
                    'sector_distribution': self._get_sector_distribution(sector_analysis),
                    'risk_summary': self._get_risk_summary(stock_analysis)
                }
            }
            
            logger.info("Generated comprehensive research report")
            return report
        except Exception as e:
            logger.error(f"Error generating research report: {str(e)}")
            return {}

    def _get_top_performers(
        self,
        stock_analysis: Dict[str, Dict]
    ) -> List[Dict]:
        """Get top performing stocks based on Sharpe ratio."""
        performers = []
        for ticker, analysis in stock_analysis.items():
            performers.append({
                'ticker': ticker,
                'sharpe_ratio': analysis['returns']['sharpe_ratio'],
                'volatility': analysis['returns']['annualized_volatility']
            })
        return sorted(performers, key=lambda x: x['sharpe_ratio'], reverse=True)[:5]

    def _get_sector_distribution(
        self,
        sector_analysis: Dict[str, Dict]
    ) -> List[Dict]:
        """Get sector distribution statistics."""
        distribution = []
        for sector, data in sector_analysis.items():
            distribution.append({
                'sector': sector,
                'num_stocks': len(data['stocks']),
                'market_cap_share': data['total_market_cap']
            })
        return sorted(distribution, key=lambda x: x['market_cap_share'], reverse=True)

    def _get_risk_summary(
        self,
        stock_analysis: Dict[str, Dict]
    ) -> Dict:
        """Get summary of risk metrics."""
        max_drawdowns = []
        volatilities = []
        
        for analysis in stock_analysis.values():
            max_drawdowns.append(analysis['risk_metrics']['max_drawdown'])
            volatilities.append(analysis['returns']['annualized_volatility'])
        
        return {
            'avg_max_drawdown': np.mean(max_drawdowns),
            'max_drawdown_std': np.std(max_drawdowns),
            'avg_volatility': np.mean(volatilities),
            'volatility_std': np.std(volatilities)
        }

if __name__ == "__main__":
    # Example usage
    agent = ResearchAgent()
    
    # Create a sample universe
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
    universe_data = agent.data_agent.fetch_multiple_stocks(tickers)
    
    # Generate research report
    report = agent.generate_research_report(universe_data)
    
    # Print summary
    print("\nResearch Report Summary:")
    print(f"Universe Size: {report['universe_size']}")
    
    print("\nTop Performers:")
    for stock in report['summary']['top_performers']:
        print(f"{stock['ticker']}: Sharpe Ratio = {stock['sharpe_ratio']:.2f}")
    
    print("\nSector Distribution:")
    for sector in report['summary']['sector_distribution']:
        print(f"{sector['sector']}: {sector['num_stocks']} stocks")
    
    print("\nRisk Summary:")
    risk = report['summary']['risk_summary']
    print(f"Average Max Drawdown: {risk['avg_max_drawdown']:.2%}")
    print(f"Average Volatility: {risk['avg_volatility']:.2%}") 