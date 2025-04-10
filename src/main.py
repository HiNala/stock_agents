import logging
import os
import sys
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import pandas as pd
import argparse
import asyncio

from src.agents.data_aggregation_agent import DataAggregationAgent
from src.agents.universe_definition_agent import UniverseDefinitionAgent
from src.agents.research_agent import ResearchAgent
from src.agents.strategy_agent import StrategyAgent
from src.agents.risk_agent import RiskAgent
from src.agents.play_agent import PlayAgent
from src.config.settings import LOG_DIR, CACHE_DIR
from src.config.model_config import ModelProvider, model_config

# Set up logging
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(LOG_DIR, 'main.log'),
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class StockAgentsCLI:
    def __init__(self):
        """Initialize the CLI with all agents."""
        self.data_agent = DataAggregationAgent()
        self.universe_agent = UniverseDefinitionAgent()
        self.research_agent = ResearchAgent()
        self.strategy_agent = StrategyAgent()
        self.risk_agent = RiskAgent()
        self.play_agent = PlayAgent()
        self.current_data = None
        self.current_universe = None
        self.agents = {
            "research": ResearchAgent(),
            "universe": UniverseDefinitionAgent(),
            "strategy": StrategyAgent(),
            "risk": RiskAgent(),
            "play": PlayAgent()
        }
        logger.info("CLI initialized with all agents")

    def prompt_for_tickers(self) -> List[str]:
        """Prompt user for stock tickers."""
        while True:
            tickers = input("\nEnter stock tickers (comma-separated): ").strip().upper()
            if tickers:
                return [t.strip() for t in tickers.split(',')]
            print("Please enter at least one ticker.")

    def prompt_for_period(self) -> str:
        """Prompt user for data period."""
        while True:
            period = input("\nEnter data period (e.g., 1y, 6mo, 1mo): ").strip().lower()
            if period in ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']:
                return period
            print("Invalid period. Please try again.")

    def prompt_for_interval(self) -> str:
        """Prompt user for data interval."""
        while True:
            interval = input("\nEnter data interval (e.g., 1d, 1h, 5m): ").strip().lower()
            if interval in ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo']:
                return interval
            print("Invalid interval. Please try again.")

    def prompt_for_strategy(self) -> str:
        """Prompt user for universe strategy."""
        while True:
            strategy = input("\nEnter strategy (momentum/mean_reversion/value/growth): ").strip().lower()
            if strategy in ['momentum', 'mean_reversion', 'value', 'growth']:
                return strategy
            print("Invalid strategy. Please try again.")

    def prompt_for_risk_tolerance(self) -> str:
        """Prompt user for risk tolerance."""
        while True:
            risk = input("\nEnter risk tolerance (low/medium/high): ").strip().lower()
            if risk in ['low', 'medium', 'high']:
                return risk
            print("Invalid risk tolerance. Please try again.")

    def prompt_for_time_horizon(self) -> str:
        """Prompt user for time horizon."""
        while True:
            horizon = input("\nEnter time horizon (short/medium/long): ").strip().lower()
            if horizon in ['short', 'medium', 'long']:
                return horizon
            print("Invalid time horizon. Please try again.")

    def prompt_for_max_positions(self) -> int:
        """Prompt user for maximum positions."""
        while True:
            try:
                positions = int(input("\nEnter maximum number of positions: "))
                if positions > 0:
                    return positions
                print("Please enter a positive number.")
            except ValueError:
                print("Please enter a valid number.")

    def fetch_data(self) -> None:
        """Fetch data for multiple tickers."""
        try:
            tickers = self.prompt_for_tickers()
            period = self.prompt_for_period()
            interval = self.prompt_for_interval()
            
            self.current_data = self.data_agent.fetch_multiple_stocks(tickers, period, interval)
            print(f"\nFetched data for {len(tickers)} tickers")
        except Exception as e:
            logger.error(f"Error fetching data: {str(e)}")
            print(f"Error: {str(e)}")

    def define_universe(self) -> None:
        """Define a universe based on strategy."""
        try:
            if not self.current_data:
                print("No data available. Please fetch data first.")
                return
            
            strategy = self.prompt_for_strategy()
            self.current_universe = self.universe_agent.define_universe(self.current_data, strategy)
            print(f"\nDefined {strategy} universe with {len(self.current_universe)} stocks")
        except Exception as e:
            logger.error(f"Error defining universe: {str(e)}")
            print(f"Error: {str(e)}")

    def research_universe(self) -> None:
        """Generate research report for a universe."""
        try:
            if not self.current_universe:
                print("No universe defined. Please define a universe first.")
                return
            
            report = self.research_agent.generate_research_report(self.current_universe)
            print("\nGenerated research report")
            # TODO: Add formatted report printing
        except Exception as e:
            logger.error(f"Error generating research report: {str(e)}")
            print(f"Error: {str(e)}")

    def backtest_strategy(self) -> None:
        """Backtest a strategy on universe data."""
        try:
            if not self.current_universe:
                print("No universe defined. Please define a universe first.")
                return
            
            strategy = self.prompt_for_strategy()
            if strategy == "momentum":
                results = self.strategy_agent.backtest_momentum_strategy(self.current_universe)
            else:
                results = self.strategy_agent.backtest_mean_reversion_strategy(self.current_universe)
            print(f"\nBacktested {strategy} strategy")
            # TODO: Add formatted results printing
        except Exception as e:
            logger.error(f"Error backtesting strategy: {str(e)}")
            print(f"Error: {str(e)}")

    def analyze_risk(self) -> None:
        """Analyze risk metrics for a universe."""
        try:
            if not self.current_universe:
                print("No universe defined. Please define a universe first.")
                return
            
            returns = pd.DataFrame()
            for ticker, data in self.current_universe.items():
                returns[ticker] = data['Close'].pct_change()
            returns = returns.dropna()
            
            risk_metrics = self.risk_agent.calculate_portfolio_risk(returns)
            correlation = self.risk_agent.calculate_correlation_matrix(returns)
            
            print("\nAnalyzed risk metrics")
            # TODO: Add formatted risk metrics printing
        except Exception as e:
            logger.error(f"Error analyzing risk: {str(e)}")
            print(f"Error: {str(e)}")

    def generate_recommendations(self) -> None:
        """Generate trade recommendations."""
        try:
            if not self.current_universe:
                print("No universe defined. Please define a universe first.")
                return
            
            risk_tolerance = self.prompt_for_risk_tolerance()
            time_horizon = self.prompt_for_time_horizon()
            max_positions = self.prompt_for_max_positions()
            
            recommendations = self.play_agent.generate_trade_recommendations(
                self.current_universe,
                risk_tolerance,
                time_horizon,
                max_positions
            )
            self.print_recommendations(recommendations)
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            print(f"Error: {str(e)}")

    def print_recommendations(self, recommendations: Dict):
        """Print formatted recommendations."""
        if not recommendations or 'recommendations' not in recommendations:
            print("No recommendations available.")
            return

        print("\nTrade Recommendations:")
        print("=" * 50)
        for rec in recommendations['recommendations']:
            print(f"\n{rec['ticker']}:")
            print(f"Action: {rec['action']}")
            print(f"Reason: {rec['reason']}")
            print(f"Price: ${rec['price']:.2f}")
            print(f"Position Size: {rec['position_size']:.1%}")
            print(f"Volatility: {rec['volatility']:.1%}")
            print(f"Beta: {rec['beta']:.2f}")
            print(f"RSI: {rec['rsi']:.1f}")
            print("-" * 30)

        print("\nPortfolio Risk Metrics:")
        print("=" * 50)
        risk = recommendations['portfolio_risk']
        print(f"Volatility: {risk['volatility']:.1%}")
        print(f"VaR (95%): {risk['var_95']:.1%}")
        print(f"Max Drawdown: {risk['max_drawdown']:.1%}")

    def run_pipeline(self) -> None:
        """Run the full analysis pipeline."""
        print("\nRunning full analysis pipeline...")
        
        # 1. Fetch data
        print("\n1. Fetching data...")
        self.fetch_data()
        if not self.current_data:
            print("Error fetching data. Exiting.")
            return
        
        # 2. Define universe
        print("\n2. Defining universe...")
        self.define_universe()
        if not self.current_universe:
            print("Error defining universe. Exiting.")
            return
        
        # 3. Research
        print("\n3. Generating research report...")
        self.research_universe()
        
        # 4. Backtest
        print("\n4. Backtesting strategy...")
        self.backtest_strategy()
        
        # 5. Risk analysis
        print("\n5. Analyzing risk...")
        self.analyze_risk()
        
        # 6. Generate recommendations
        print("\n6. Generating recommendations...")
        self.generate_recommendations()

    async def configure_llm(self, agent_name: str) -> None:
        """Configure LLM settings for a specific agent through interactive prompts."""
        if agent_name not in self.agents:
            print(f"Unknown agent: {agent_name}")
            return

        print(f"\nConfiguring LLM for {agent_name} agent:")
        print("Available providers:")
        for provider in ModelProvider:
            print(f"- {provider.value}")

        # Get provider
        while True:
            provider = input("\nSelect provider: ").strip().lower()
            try:
                provider = ModelProvider(provider)
                break
            except ValueError:
                print("Invalid provider. Please try again.")

        # Get model
        provider_config = model_config.get_provider_config(provider)
        print("\nAvailable models:")
        for model in provider_config["available_models"]:
            print(f"- {model}")
        
        model = input("\nSelect model: ").strip()
        if model not in provider_config["available_models"]:
            print(f"Warning: {model} is not in the list of available models")

        # Get temperature
        while True:
            try:
                temperature = float(input("\nEnter temperature (0.0-1.0): ").strip())
                if 0.0 <= temperature <= 1.0:
                    break
                print("Temperature must be between 0.0 and 1.0")
            except ValueError:
                print("Invalid temperature. Please enter a number.")

        # Get max tokens
        while True:
            try:
                max_tokens = int(input("\nEnter max tokens: ").strip())
                if max_tokens > 0:
                    break
                print("Max tokens must be greater than 0")
            except ValueError:
                print("Invalid max tokens. Please enter a number.")

        # Update configuration
        new_config = {
            "provider": provider,
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        self.agents[agent_name].update_llm_config(new_config)
        print(f"\nSuccessfully updated {agent_name} agent configuration:")
        print(f"Provider: {provider.value}")
        print(f"Model: {model}")
        print(f"Temperature: {temperature}")
        print(f"Max Tokens: {max_tokens}")

    async def show_current_configs(self) -> None:
        """Display current LLM configurations for all agents."""
        print("\nCurrent LLM Configurations:")
        for agent_name, agent in self.agents.items():
            config = model_config.get_agent_config(agent_name)
            print(f"\n{agent_name} agent:")
            print(f"  Provider: {config['provider'].value}")
            print(f"  Model: {config['model']}")
            print(f"  Temperature: {config['temperature']}")
            print(f"  Max Tokens: {config['max_tokens']}")

    async def run(self) -> None:
        """Run the CLI interface."""
        while True:
            print("\nStock Agents CLI")
            print("1. Configure LLM for an agent")
            print("2. Show current configurations")
            print("3. Exit")

            choice = input("\nEnter your choice (1-3): ").strip()

            if choice == "1":
                print("\nAvailable agents:")
                for agent in self.agents.keys():
                    print(f"- {agent}")
                agent_name = input("\nSelect agent to configure: ").strip()
                await self.configure_llm(agent_name)
            elif choice == "2":
                await self.show_current_configs()
            elif choice == "3":
                print("Exiting...")
                break
            else:
                print("Invalid choice. Please try again.")

async def main():
    cli = StockAgentsCLI()
    await cli.run()

if __name__ == "__main__":
    asyncio.run(main()) 