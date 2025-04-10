# Stock Agents System

A comprehensive multi-agent system for stock research, universe creation, and trading strategy development. This system uses free data sources (primarily Yahoo Finance via `yfinance`) to provide actionable insights and recommendations.

## Features

- **Data Aggregation**: Fetch and normalize stock data from Yahoo Finance
- **Universe Definition**: Create specialized stock universes based on different strategies
- **Research Analysis**: Generate detailed research reports for stock universes
- **Strategy Backtesting**: Test momentum and mean reversion strategies
- **Risk Analysis**: Calculate various risk metrics and correlations
- **Trade Recommendations**: Generate actionable trade ideas with position sizing
- **Interactive CLI**: User-friendly prompt-based interface

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/stock_agents.git
cd stock_agents
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the main program:
```bash
python src/main.py
```

The program will present an interactive menu with the following options:

1. **Fetch Data**
   - Enter stock tickers (comma-separated)
   - Select data period (e.g., 1y, 6mo)
   - Choose data interval (e.g., 1d, 1h)

2. **Define Universe**
   - Choose strategy (momentum/mean_reversion/value/growth)
   - System will filter stocks based on strategy criteria

3. **Research Universe**
   - Generate detailed research report
   - Analyze trends and patterns
   - Evaluate fundamental metrics

4. **Backtest Strategy**
   - Select strategy type
   - View performance metrics
   - Analyze strategy effectiveness

5. **Analyze Risk**
   - Calculate portfolio risk metrics
   - View correlation matrix
   - Assess risk contributions

6. **Generate Recommendations**
   - Set risk tolerance (low/medium/high)
   - Choose time horizon (short/medium/long)
   - Specify maximum positions
   - Get detailed trade recommendations

7. **Run Full Pipeline**
   - Execute complete analysis from data fetching to recommendations
   - View comprehensive results

## Project Structure

```
stock_agents/
├── src/
│   ├── agents/
│   │   ├── data_aggregation_agent.py
│   │   ├── universe_definition_agent.py
│   │   ├── research_agent.py
│   │   ├── strategy_agent.py
│   │   ├── risk_agent.py
│   │   └── play_agent.py
│   ├── config/
│   │   └── settings.py
│   └── main.py
├── data/
│   └── cache/
├── logs/
├── requirements.txt
└── README.md
```

## Agents Overview

1. **Data Aggregation Agent**
   - Fetches stock data from Yahoo Finance
   - Normalizes and caches data
   - Handles data updates

2. **Universe Definition Agent**
   - Creates strategy-based stock universes
   - Implements filtering criteria
   - Manages universe composition

3. **Research Agent**
   - Generates research reports
   - Analyzes trends and patterns
   - Evaluates fundamental metrics

4. **Strategy Agent**
   - Implements trading strategies
   - Performs backtesting
   - Optimizes strategy parameters

5. **Risk Agent**
   - Calculates risk metrics
   - Analyzes correlations
   - Performs stress testing

6. **Play Agent**
   - Generates trade recommendations
   - Calculates position sizes
   - Considers risk parameters

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [yfinance](https://pypi.org/project/yfinance/) for providing free market data
- [pandas](https://pandas.pydata.org/) for data manipulation
- [numpy](https://numpy.org/) for numerical computations 