# Stock Agents: Multi-Agent System for Stock Research & Universe Creation

A comprehensive multi-agent system for stock research, universe creation, and trading strategy development. This system leverages free data sources and modern AI capabilities to provide actionable insights for traders and portfolio managers.

## Features

- **Data Aggregation**: Collect and normalize market data from free sources
- **Universe Definition**: Create and filter stock universes based on various strategies
- **Deep Research**: Perform in-depth analysis of each universe
- **Strategy Backtesting**: Simulate and backtest trading strategies
- **Risk Analysis**: Calculate risk metrics and correlations
- **Trade Recommendations**: Generate actionable trade ideas
- **Real-time Monitoring**: Track live market data and adjust strategies
- **Interactive Dashboard**: Visualize research reports and recommendations
- **Flexible Configuration**: Easily configure data frequency, date ranges, and LLM models

## Prerequisites

- Python 3.8+
- pip (Python package manager)
- Git

## Installation

1. Clone the repository:
```bash
git clone https://github.com/HiNala/stock_agents.git
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

4. Set up environment variables:
```bash
cp .env.example .env
```
Edit the `.env` file with your API keys and configurations.

## Configuration

The system uses several environment variables for configuration. Copy `.env.example` to `.env` and fill in your values:

```env
# API Keys
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
HUGGINGFACE_API_KEY=your_huggingface_api_key_here

# Data Settings
DATA_PERIOD=1y
DATA_INTERVAL=1d
CACHE_DIR=./cache
LOG_DIR=./logs

# Trading Parameters
INITIAL_CAPITAL=100000
COMMISSION_RATE=0.001

# LLM Model Settings
DEFAULT_MODEL_PROVIDER=openai
DEFAULT_MODEL=gpt-3.5-turbo
DEFAULT_TEMPERATURE=0.7
DEFAULT_MAX_TOKENS=2000

# Agent-Specific Settings
RESEARCH_AGENT_MODEL=gpt-4
RESEARCH_AGENT_TEMPERATURE=0.7
RESEARCH_AGENT_MAX_TOKENS=2000

UNIVERSE_AGENT_MODEL=gpt-3.5-turbo
UNIVERSE_AGENT_TEMPERATURE=0.5
UNIVERSE_AGENT_MAX_TOKENS=1000

STRATEGY_AGENT_MODEL=gpt-3.5-turbo
STRATEGY_AGENT_TEMPERATURE=0.3
STRATEGY_AGENT_MAX_TOKENS=1000

RISK_AGENT_MODEL=gpt-3.5-turbo
RISK_AGENT_TEMPERATURE=0.3
RISK_AGENT_MAX_TOKENS=1000

PLAY_AGENT_MODEL=gpt-4
PLAY_AGENT_TEMPERATURE=0.7
PLAY_AGENT_MAX_TOKENS=2000

# Email Notifications (Optional)
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
EMAIL_USER=your_email@gmail.com
EMAIL_PASSWORD=your_app_specific_password
NOTIFICATION_EMAIL=your_notification_email@gmail.com

# Dashboard Settings
DASHBOARD_HOST=localhost
DASHBOARD_PORT=5000
DEBUG_MODE=True

# Backtesting Settings
BACKTEST_START_DATE=2020-01-01
BACKTEST_END_DATE=2023-12-31

# Universe Filter Settings
MIN_MARKET_CAP=1000000000  # $1B
MIN_AVG_VOLUME=1000000    # 1M shares
MAX_PE_RATIO=30
MIN_PRICE=5.0
```

## Usage

1. Start the CLI interface:
```bash
python src/main.py
```

2. Follow the interactive prompts to:
   - Configure data settings (period, interval, backtest range)
   - Configure LLM models for different agents
   - Fetch market data
   - Define stock universes
   - Generate research reports
   - Backtest strategies
   - Analyze risk
   - Get trade recommendations

### Data Configuration Options

The system supports various data frequency and date range configurations:

#### Data Periods
- 1d (1 day)
- 5d (5 days)
- 1mo (1 month)
- 3mo (3 months)
- 6mo (6 months)
- 1y (1 year)
- 2y (2 years)
- 5y (5 years)
- 10y (10 years)
- ytd (year to date)
- max (maximum available)

#### Data Intervals
- 1m (1 minute)
- 2m (2 minutes)
- 5m (5 minutes)
- 15m (15 minutes)
- 30m (30 minutes)
- 60m (60 minutes)
- 90m (90 minutes)
- 1h (1 hour)
- 1d (1 day)
- 5d (5 days)
- 1wk (1 week)
- 1mo (1 month)
- 3mo (3 months)

#### Backtest Date Range
- Start and end dates can be specified in YYYY-MM-DD format
- The system will use this range for strategy backtesting

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
│   │   ├── settings.py
│   │   └── model_config.py
│   ├── llm/
│   │   └── base_llm_client.py
│   └── main.py
├── tests/
├── cache/
├── logs/
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [yfinance](https://pypi.org/project/yfinance/) for market data
- OpenAI, Anthropic, and HuggingFace for LLM capabilities
- The open-source community for various Python libraries 