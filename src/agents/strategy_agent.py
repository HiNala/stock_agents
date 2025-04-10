import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import os
from datetime import datetime, timedelta

from src.config.settings import CACHE_DIR, LOG_DIR, INITIAL_CAPITAL, COMMISSION_RATE
from src.agents.data_aggregation_agent import DataAggregationAgent

# Set up logging
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(LOG_DIR, 'strategy_agent.log'),
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class StrategyAgent:
    def __init__(self):
        """Initialize the Strategy Agent."""
        self.data_agent = DataAggregationAgent()
        self.cache_dir = CACHE_DIR
        os.makedirs(self.cache_dir, exist_ok=True)
        logger.info("Strategy Agent initialized")

    def backtest_momentum_strategy(
        self,
        data: pd.DataFrame,
        lookback_period: int = 20,
        holding_period: int = 5,
        initial_capital: float = INITIAL_CAPITAL,
        commission_rate: float = COMMISSION_RATE
    ) -> Dict:
        """
        Backtest a simple momentum strategy.
        
        Args:
            data: DataFrame containing price data
            lookback_period: Number of days to look back for momentum calculation
            holding_period: Number of days to hold positions
            initial_capital: Initial capital for backtesting
            commission_rate: Commission rate per trade
            
        Returns:
            Dictionary containing backtest results
        """
        try:
            # Calculate returns and momentum
            data['Returns'] = data['Close'].pct_change()
            data['Momentum'] = data['Close'].pct_change(periods=lookback_period)
            
            # Generate signals
            data['Signal'] = np.where(data['Momentum'] > 0, 1, -1)
            
            # Calculate positions
            data['Position'] = data['Signal'].shift(1)
            
            # Calculate strategy returns
            data['Strategy_Returns'] = data['Position'] * data['Returns']
            
            # Calculate commission costs
            data['Trades'] = data['Position'].diff().abs()
            data['Commission'] = data['Trades'] * commission_rate
            
            # Calculate equity curve
            data['Strategy_Returns_Net'] = data['Strategy_Returns'] - data['Commission']
            data['Equity'] = (1 + data['Strategy_Returns_Net']).cumprod() * initial_capital
            
            # Calculate performance metrics
            total_return = (data['Equity'].iloc[-1] / initial_capital) - 1
            annual_return = (1 + total_return) ** (252 / len(data)) - 1
            volatility = data['Strategy_Returns_Net'].std() * np.sqrt(252)
            sharpe_ratio = annual_return / volatility if volatility != 0 else 0
            
            # Calculate drawdown
            data['Peak'] = data['Equity'].cummax()
            data['Drawdown'] = (data['Equity'] - data['Peak']) / data['Peak']
            max_drawdown = data['Drawdown'].min()
            
            # Calculate win rate
            winning_trades = (data['Strategy_Returns_Net'] > 0).sum()
            total_trades = (data['Trades'] > 0).sum()
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            results = {
                'total_return': total_return,
                'annual_return': annual_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'total_trades': total_trades,
                'equity_curve': data['Equity'].to_dict(),
                'drawdown_curve': data['Drawdown'].to_dict()
            }
            
            logger.info("Completed momentum strategy backtest")
            return results
        except Exception as e:
            logger.error(f"Error in momentum strategy backtest: {str(e)}")
            return {}

    def backtest_mean_reversion_strategy(
        self,
        data: pd.DataFrame,
        lookback_period: int = 20,
        std_devs: float = 2.0,
        initial_capital: float = INITIAL_CAPITAL,
        commission_rate: float = COMMISSION_RATE
    ) -> Dict:
        """
        Backtest a mean reversion strategy using Bollinger Bands.
        
        Args:
            data: DataFrame containing price data
            lookback_period: Number of days to look back for mean calculation
            std_devs: Number of standard deviations for bands
            initial_capital: Initial capital for backtesting
            commission_rate: Commission rate per trade
            
        Returns:
            Dictionary containing backtest results
        """
        try:
            # Calculate Bollinger Bands
            data['MA'] = data['Close'].rolling(window=lookback_period).mean()
            data['STD'] = data['Close'].rolling(window=lookback_period).std()
            data['Upper_Band'] = data['MA'] + (data['STD'] * std_devs)
            data['Lower_Band'] = data['MA'] - (data['STD'] * std_devs)
            
            # Generate signals
            data['Signal'] = np.where(
                data['Close'] < data['Lower_Band'], 1,
                np.where(data['Close'] > data['Upper_Band'], -1, 0)
            )
            
            # Calculate positions
            data['Position'] = data['Signal'].shift(1)
            
            # Calculate strategy returns
            data['Returns'] = data['Close'].pct_change()
            data['Strategy_Returns'] = data['Position'] * data['Returns']
            
            # Calculate commission costs
            data['Trades'] = data['Position'].diff().abs()
            data['Commission'] = data['Trades'] * commission_rate
            
            # Calculate equity curve
            data['Strategy_Returns_Net'] = data['Strategy_Returns'] - data['Commission']
            data['Equity'] = (1 + data['Strategy_Returns_Net']).cumprod() * initial_capital
            
            # Calculate performance metrics
            total_return = (data['Equity'].iloc[-1] / initial_capital) - 1
            annual_return = (1 + total_return) ** (252 / len(data)) - 1
            volatility = data['Strategy_Returns_Net'].std() * np.sqrt(252)
            sharpe_ratio = annual_return / volatility if volatility != 0 else 0
            
            # Calculate drawdown
            data['Peak'] = data['Equity'].cummax()
            data['Drawdown'] = (data['Equity'] - data['Peak']) / data['Peak']
            max_drawdown = data['Drawdown'].min()
            
            # Calculate win rate
            winning_trades = (data['Strategy_Returns_Net'] > 0).sum()
            total_trades = (data['Trades'] > 0).sum()
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            results = {
                'total_return': total_return,
                'annual_return': annual_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'total_trades': total_trades,
                'equity_curve': data['Equity'].to_dict(),
                'drawdown_curve': data['Drawdown'].to_dict()
            }
            
            logger.info("Completed mean reversion strategy backtest")
            return results
        except Exception as e:
            logger.error(f"Error in mean reversion strategy backtest: {str(e)}")
            return {}

    def optimize_strategy_parameters(
        self,
        data: pd.DataFrame,
        strategy_type: str = 'momentum',
        param_ranges: Dict = None
    ) -> Dict:
        """
        Optimize strategy parameters using grid search.
        
        Args:
            data: DataFrame containing price data
            strategy_type: Type of strategy to optimize ('momentum' or 'mean_reversion')
            param_ranges: Dictionary of parameter ranges to test
            
        Returns:
            Dictionary containing optimization results
        """
        try:
            if param_ranges is None:
                if strategy_type == 'momentum':
                    param_ranges = {
                        'lookback_period': range(10, 51, 10),
                        'holding_period': range(1, 11, 2)
                    }
                else:  # mean_reversion
                    param_ranges = {
                        'lookback_period': range(10, 51, 10),
                        'std_devs': [1.5, 2.0, 2.5]
                    }
            
            best_sharpe = -float('inf')
            best_params = {}
            results = []
            
            if strategy_type == 'momentum':
                for lookback in param_ranges['lookback_period']:
                    for holding in param_ranges['holding_period']:
                        result = self.backtest_momentum_strategy(
                            data.copy(),
                            lookback_period=lookback,
                            holding_period=holding
                        )
                        if result and result['sharpe_ratio'] > best_sharpe:
                            best_sharpe = result['sharpe_ratio']
                            best_params = {
                                'lookback_period': lookback,
                                'holding_period': holding
                            }
                        results.append({
                            'params': {'lookback': lookback, 'holding': holding},
                            'sharpe_ratio': result.get('sharpe_ratio', 0)
                        })
            else:  # mean_reversion
                for lookback in param_ranges['lookback_period']:
                    for std_dev in param_ranges['std_devs']:
                        result = self.backtest_mean_reversion_strategy(
                            data.copy(),
                            lookback_period=lookback,
                            std_devs=std_dev
                        )
                        if result and result['sharpe_ratio'] > best_sharpe:
                            best_sharpe = result['sharpe_ratio']
                            best_params = {
                                'lookback_period': lookback,
                                'std_devs': std_dev
                            }
                        results.append({
                            'params': {'lookback': lookback, 'std_devs': std_dev},
                            'sharpe_ratio': result.get('sharpe_ratio', 0)
                        })
            
            optimization_results = {
                'best_params': best_params,
                'best_sharpe_ratio': best_sharpe,
                'all_results': results
            }
            
            logger.info(f"Completed {strategy_type} strategy optimization")
            return optimization_results
        except Exception as e:
            logger.error(f"Error in strategy optimization: {str(e)}")
            return {}

if __name__ == "__main__":
    # Example usage
    agent = StrategyAgent()
    
    # Fetch data for a stock
    data = agent.data_agent.fetch_stock_data("AAPL")
    
    # Backtest momentum strategy
    momentum_results = agent.backtest_momentum_strategy(data.copy())
    print("\nMomentum Strategy Results:")
    print(f"Total Return: {momentum_results['total_return']:.2%}")
    print(f"Sharpe Ratio: {momentum_results['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {momentum_results['max_drawdown']:.2%}")
    
    # Backtest mean reversion strategy
    mean_reversion_results = agent.backtest_mean_reversion_strategy(data.copy())
    print("\nMean Reversion Strategy Results:")
    print(f"Total Return: {mean_reversion_results['total_return']:.2%}")
    print(f"Sharpe Ratio: {mean_reversion_results['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {mean_reversion_results['max_drawdown']:.2%}")
    
    # Optimize momentum strategy
    momentum_optimization = agent.optimize_strategy_parameters(
        data.copy(),
        strategy_type='momentum'
    )
    print("\nMomentum Strategy Optimization:")
    print(f"Best Parameters: {momentum_optimization['best_params']}")
    print(f"Best Sharpe Ratio: {momentum_optimization['best_sharpe_ratio']:.2f}") 