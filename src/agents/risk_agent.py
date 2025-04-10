import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import os
from datetime import datetime, timedelta
from scipy import stats

from src.config.settings import CACHE_DIR, LOG_DIR
from src.agents.data_aggregation_agent import DataAggregationAgent

# Set up logging
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(LOG_DIR, 'risk_agent.log'),
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RiskAgent:
    def __init__(self):
        """Initialize the Risk Agent."""
        self.data_agent = DataAggregationAgent()
        self.cache_dir = CACHE_DIR
        os.makedirs(self.cache_dir, exist_ok=True)
        logger.info("Risk Agent initialized")

    def calculate_var(
        self,
        returns: pd.Series,
        confidence_level: float = 0.95,
        method: str = 'historical'
    ) -> float:
        """
        Calculate Value at Risk (VaR) using different methods.
        
        Args:
            returns: Series of returns
            confidence_level: Confidence level for VaR calculation
            method: Method to use ('historical', 'parametric', or 'monte_carlo')
            
        Returns:
            VaR value
        """
        try:
            if method == 'historical':
                # Historical VaR
                var = np.percentile(returns, (1 - confidence_level) * 100)
            elif method == 'parametric':
                # Parametric VaR (assuming normal distribution)
                mean = returns.mean()
                std = returns.std()
                var = stats.norm.ppf(1 - confidence_level, mean, std)
            else:
                raise ValueError(f"Unknown VaR method: {method}")
            
            logger.info(f"Calculated {method} VaR at {confidence_level} confidence level")
            return var
        except Exception as e:
            logger.error(f"Error calculating VaR: {str(e)}")
            return 0.0

    def calculate_expected_shortfall(
        self,
        returns: pd.Series,
        confidence_level: float = 0.95
    ) -> float:
        """
        Calculate Expected Shortfall (ES) or Conditional VaR.
        
        Args:
            returns: Series of returns
            confidence_level: Confidence level for ES calculation
            
        Returns:
            ES value
        """
        try:
            var = self.calculate_var(returns, confidence_level, 'historical')
            es = returns[returns <= var].mean()
            
            logger.info(f"Calculated Expected Shortfall at {confidence_level} confidence level")
            return es
        except Exception as e:
            logger.error(f"Error calculating Expected Shortfall: {str(e)}")
            return 0.0

    def calculate_portfolio_risk(
        self,
        returns: pd.DataFrame,
        weights: Optional[List[float]] = None
    ) -> Dict:
        """
        Calculate portfolio risk metrics.
        
        Args:
            returns: DataFrame of asset returns
            weights: Portfolio weights (if None, equal weights are used)
            
        Returns:
            Dictionary containing portfolio risk metrics
        """
        try:
            if weights is None:
                weights = [1.0 / len(returns.columns)] * len(returns.columns)
            
            # Calculate portfolio returns
            portfolio_returns = (returns * weights).sum(axis=1)
            
            # Calculate risk metrics
            volatility = portfolio_returns.std() * np.sqrt(252)  # Annualized
            var_95 = self.calculate_var(portfolio_returns, 0.95)
            es_95 = self.calculate_expected_shortfall(portfolio_returns, 0.95)
            
            # Calculate drawdown
            cumulative_returns = (1 + portfolio_returns).cumprod()
            peak = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - peak) / peak
            max_drawdown = drawdown.min()
            
            results = {
                'volatility': volatility,
                'var_95': var_95,
                'es_95': es_95,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252)
            }
            
            logger.info("Calculated portfolio risk metrics")
            return results
        except Exception as e:
            logger.error(f"Error calculating portfolio risk: {str(e)}")
            return {}

    def calculate_correlation_matrix(
        self,
        returns: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate correlation matrix for assets.
        
        Args:
            returns: DataFrame of asset returns
            
        Returns:
            Correlation matrix
        """
        try:
            correlation_matrix = returns.corr()
            logger.info("Calculated correlation matrix")
            return correlation_matrix
        except Exception as e:
            logger.error(f"Error calculating correlation matrix: {str(e)}")
            return pd.DataFrame()

    def calculate_beta(
        self,
        asset_returns: pd.Series,
        market_returns: pd.Series
    ) -> float:
        """
        Calculate beta of an asset relative to the market.
        
        Args:
            asset_returns: Series of asset returns
            market_returns: Series of market returns
            
        Returns:
            Beta value
        """
        try:
            # Calculate covariance and variance
            covariance = asset_returns.cov(market_returns)
            market_variance = market_returns.var()
            
            beta = covariance / market_variance
            logger.info("Calculated beta")
            return beta
        except Exception as e:
            logger.error(f"Error calculating beta: {str(e)}")
            return 0.0

    def stress_test_portfolio(
        self,
        returns: pd.DataFrame,
        weights: List[float],
        stress_scenarios: Dict[str, float]
    ) -> Dict:
        """
        Perform stress testing on a portfolio.
        
        Args:
            returns: DataFrame of asset returns
            weights: Portfolio weights
            stress_scenarios: Dictionary of stress scenarios and their impacts
            
        Returns:
            Dictionary containing stress test results
        """
        try:
            results = {}
            
            for scenario, impact in stress_scenarios.items():
                # Apply stress scenario
                stressed_returns = returns.copy()
                for asset in stressed_returns.columns:
                    stressed_returns[asset] = stressed_returns[asset] * (1 + impact)
                
                # Calculate stressed portfolio metrics
                portfolio_returns = (stressed_returns * weights).sum(axis=1)
                volatility = portfolio_returns.std() * np.sqrt(252)
                var_95 = self.calculate_var(portfolio_returns, 0.95)
                
                results[scenario] = {
                    'volatility': volatility,
                    'var_95': var_95,
                    'max_drawdown': ((1 + portfolio_returns).cumprod() - 1).min()
                }
            
            logger.info("Completed portfolio stress testing")
            return results
        except Exception as e:
            logger.error(f"Error in stress testing: {str(e)}")
            return {}

    def calculate_risk_contribution(
        self,
        returns: pd.DataFrame,
        weights: List[float]
    ) -> Dict[str, float]:
        """
        Calculate risk contribution of each asset in the portfolio.
        
        Args:
            returns: DataFrame of asset returns
            weights: Portfolio weights
            
        Returns:
            Dictionary mapping assets to their risk contributions
        """
        try:
            # Calculate portfolio volatility
            portfolio_returns = (returns * weights).sum(axis=1)
            portfolio_volatility = portfolio_returns.std() * np.sqrt(252)
            
            # Calculate marginal contributions
            marginal_contributions = {}
            for asset in returns.columns:
                # Calculate covariance with portfolio
                covariance = returns[asset].cov(portfolio_returns)
                # Calculate marginal contribution
                marginal_contributions[asset] = (weights[returns.columns.get_loc(asset)] * 
                                               covariance / portfolio_volatility)
            
            # Calculate percentage contributions
            total_contribution = sum(marginal_contributions.values())
            risk_contributions = {
                asset: contribution / total_contribution
                for asset, contribution in marginal_contributions.items()
            }
            
            logger.info("Calculated risk contributions")
            return risk_contributions
        except Exception as e:
            logger.error(f"Error calculating risk contributions: {str(e)}")
            return {}

if __name__ == "__main__":
    # Example usage
    agent = RiskAgent()
    
    # Fetch data for multiple stocks
    tickers = ["AAPL", "MSFT", "GOOGL"]
    stock_data = agent.data_agent.fetch_multiple_stocks(tickers)
    
    # Calculate returns
    returns = pd.DataFrame()
    for ticker, data in stock_data.items():
        returns[ticker] = data['Close'].pct_change()
    returns = returns.dropna()
    
    # Calculate risk metrics
    portfolio_risk = agent.calculate_portfolio_risk(returns)
    print("\nPortfolio Risk Metrics:")
    print(f"Volatility: {portfolio_risk['volatility']:.2%}")
    print(f"VaR (95%): {portfolio_risk['var_95']:.2%}")
    print(f"Max Drawdown: {portfolio_risk['max_drawdown']:.2%}")
    
    # Calculate correlation matrix
    correlation_matrix = agent.calculate_correlation_matrix(returns)
    print("\nCorrelation Matrix:")
    print(correlation_matrix)
    
    # Calculate risk contributions
    weights = [1/3, 1/3, 1/3]  # Equal weights
    risk_contributions = agent.calculate_risk_contribution(returns, weights)
    print("\nRisk Contributions:")
    for asset, contribution in risk_contributions.items():
        print(f"{asset}: {contribution:.2%}")
    
    # Perform stress testing
    stress_scenarios = {
        'market_crash': -0.20,  # 20% market decline
        'volatility_spike': 0.50,  # 50% increase in volatility
        'sector_rotation': -0.10  # 10% sector rotation impact
    }
    stress_results = agent.stress_test_portfolio(returns, weights, stress_scenarios)
    print("\nStress Test Results:")
    for scenario, results in stress_results.items():
        print(f"\n{scenario}:")
        print(f"Volatility: {results['volatility']:.2%}")
        print(f"VaR (95%): {results['var_95']:.2%}") 