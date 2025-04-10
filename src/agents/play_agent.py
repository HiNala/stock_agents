import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import os
from datetime import datetime, timedelta

from src.config.settings import CACHE_DIR, LOG_DIR, RISK_PER_TRADE
from src.agents.data_aggregation_agent import DataAggregationAgent
from src.agents.research_agent import ResearchAgent
from src.agents.strategy_agent import StrategyAgent
from src.agents.risk_agent import RiskAgent

# Set up logging
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(LOG_DIR, 'play_agent.log'),
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PlayAgent:
    def __init__(self):
        """Initialize the Play Agent."""
        self.data_agent = DataAggregationAgent()
        self.research_agent = ResearchAgent()
        self.strategy_agent = StrategyAgent()
        self.risk_agent = RiskAgent()
        self.cache_dir = CACHE_DIR
        os.makedirs(self.cache_dir, exist_ok=True)
        logger.info("Play Agent initialized")

    def generate_trade_recommendations(
        self,
        universe_data: Dict[str, pd.DataFrame],
        risk_tolerance: str = 'medium',
        time_horizon: str = 'medium',
        max_positions: int = 5
    ) -> Dict:
        """
        Generate trade recommendations based on research and analysis.
        
        Args:
            universe_data: Dictionary mapping tickers to their data
            risk_tolerance: Risk tolerance level ('low', 'medium', 'high')
            time_horizon: Investment time horizon ('short', 'medium', 'long')
            max_positions: Maximum number of positions to recommend
            
        Returns:
            Dictionary containing trade recommendations
        """
        try:
            # Get research analysis
            research_report = self.research_agent.generate_research_report(universe_data)
            
            # Get risk metrics
            returns = pd.DataFrame()
            for ticker, data in universe_data.items():
                returns[ticker] = data['Close'].pct_change()
            returns = returns.dropna()
            
            portfolio_risk = self.risk_agent.calculate_portfolio_risk(returns)
            correlation_matrix = self.risk_agent.calculate_correlation_matrix(returns)
            
            # Generate recommendations
            recommendations = []
            for ticker, data in universe_data.items():
                # Get stock info
                info = self.data_agent.get_stock_info(ticker)
                
                # Calculate technical indicators
                data['Returns'] = data['Close'].pct_change()
                data['MA20'] = data['Close'].rolling(window=20).mean()
                data['MA50'] = data['Close'].rolling(window=50).mean()
                data['RSI'] = self._calculate_rsi(data['Close'])
                
                # Generate signals
                momentum_signal = self._generate_momentum_signal(data)
                mean_reversion_signal = self._generate_mean_reversion_signal(data)
                
                # Calculate risk metrics
                volatility = data['Returns'].std() * np.sqrt(252)
                beta = self.risk_agent.calculate_beta(
                    data['Returns'],
                    returns.mean(axis=1)  # Use average market returns as proxy
                )
                
                # Determine recommendation
                recommendation = self._determine_recommendation(
                    momentum_signal,
                    mean_reversion_signal,
                    volatility,
                    beta,
                    risk_tolerance,
                    time_horizon
                )
                
                if recommendation['action'] != 'HOLD':
                    recommendations.append({
                        'ticker': ticker,
                        'action': recommendation['action'],
                        'reason': recommendation['reason'],
                        'price': data['Close'].iloc[-1],
                        'volatility': volatility,
                        'beta': beta,
                        'rsi': data['RSI'].iloc[-1],
                        'momentum_signal': momentum_signal,
                        'mean_reversion_signal': mean_reversion_signal
                    })
            
            # Sort and filter recommendations
            recommendations = sorted(
                recommendations,
                key=lambda x: (
                    1 if x['action'] == 'BUY' else 0,
                    -x['volatility'] if risk_tolerance == 'low' else x['volatility']
                ),
                reverse=True
            )[:max_positions]
            
            # Calculate position sizes
            for rec in recommendations:
                rec['position_size'] = self._calculate_position_size(
                    rec['volatility'],
                    risk_tolerance,
                    RISK_PER_TRADE
                )
            
            result = {
                'timestamp': datetime.now().isoformat(),
                'risk_tolerance': risk_tolerance,
                'time_horizon': time_horizon,
                'recommendations': recommendations,
                'portfolio_risk': portfolio_risk,
                'correlation_matrix': correlation_matrix.to_dict()
            }
            
            logger.info(f"Generated {len(recommendations)} trade recommendations")
            return result
        except Exception as e:
            logger.error(f"Error generating trade recommendations: {str(e)}")
            return {}

    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _generate_momentum_signal(self, data: pd.DataFrame) -> int:
        """Generate momentum trading signal."""
        if data['Close'].iloc[-1] > data['MA20'].iloc[-1] and data['MA20'].iloc[-1] > data['MA50'].iloc[-1]:
            return 1  # Strong buy
        elif data['Close'].iloc[-1] < data['MA20'].iloc[-1] and data['MA20'].iloc[-1] < data['MA50'].iloc[-1]:
            return -1  # Strong sell
        return 0  # Neutral

    def _generate_mean_reversion_signal(self, data: pd.DataFrame) -> int:
        """Generate mean reversion trading signal."""
        rsi = data['RSI'].iloc[-1]
        if rsi < 30:
            return 1  # Oversold, buy
        elif rsi > 70:
            return -1  # Overbought, sell
        return 0  # Neutral

    def _determine_recommendation(
        self,
        momentum_signal: int,
        mean_reversion_signal: int,
        volatility: float,
        beta: float,
        risk_tolerance: str,
        time_horizon: str
    ) -> Dict:
        """Determine final trading recommendation."""
        # Combine signals
        combined_signal = momentum_signal + mean_reversion_signal
        
        # Adjust for risk tolerance
        if risk_tolerance == 'low' and volatility > 0.3:
            combined_signal = 0
        elif risk_tolerance == 'high' and volatility < 0.1:
            combined_signal = 0
        
        # Adjust for time horizon
        if time_horizon == 'short' and beta < 0.8:
            combined_signal = 0
        elif time_horizon == 'long' and beta > 1.2:
            combined_signal = 0
        
        if combined_signal >= 1:
            return {'action': 'BUY', 'reason': 'Strong buy signal from multiple indicators'}
        elif combined_signal <= -1:
            return {'action': 'SELL', 'reason': 'Strong sell signal from multiple indicators'}
        return {'action': 'HOLD', 'reason': 'Neutral or conflicting signals'}

    def _calculate_position_size(
        self,
        volatility: float,
        risk_tolerance: str,
        risk_per_trade: float
    ) -> float:
        """Calculate position size based on risk parameters."""
        # Adjust risk per trade based on risk tolerance
        if risk_tolerance == 'low':
            adjusted_risk = risk_per_trade * 0.5
        elif risk_tolerance == 'high':
            adjusted_risk = risk_per_trade * 1.5
        else:
            adjusted_risk = risk_per_trade
        
        # Adjust for volatility
        position_size = adjusted_risk / volatility
        return min(position_size, 1.0)  # Cap at 100%

if __name__ == "__main__":
    # Example usage
    agent = PlayAgent()
    
    # Create a sample universe
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
    universe_data = agent.data_agent.fetch_multiple_stocks(tickers)
    
    # Generate recommendations
    recommendations = agent.generate_trade_recommendations(
        universe_data,
        risk_tolerance='medium',
        time_horizon='medium',
        max_positions=3
    )
    
    # Print recommendations
    print("\nTrade Recommendations:")
    for rec in recommendations['recommendations']:
        print(f"\n{rec['ticker']}:")
        print(f"Action: {rec['action']}")
        print(f"Reason: {rec['reason']}")
        print(f"Price: ${rec['price']:.2f}")
        print(f"Position Size: {rec['position_size']:.1%}")
        print(f"Volatility: {rec['volatility']:.1%}")
        print(f"Beta: {rec['beta']:.2f}")
        print(f"RSI: {rec['rsi']:.1f}")
    
    # Print portfolio risk metrics
    print("\nPortfolio Risk Metrics:")
    risk = recommendations['portfolio_risk']
    print(f"Volatility: {risk['volatility']:.1%}")
    print(f"VaR (95%): {risk['var_95']:.1%}")
    print(f"Max Drawdown: {risk['max_drawdown']:.1%}") 