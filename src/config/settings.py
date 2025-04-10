import os
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent.parent
CACHE_DIR = Path(os.getenv('CACHE_DIR', './cache'))
LOG_DIR = Path(os.getenv('LOG_DIR', './logs'))

# Create directories if they don't exist
CACHE_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# API Keys
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
HUGGINGFACE_API_KEY = os.getenv('HUGGINGFACE_API_KEY')

# Data Settings
DATA_PERIOD = os.getenv('DATA_PERIOD', '1y')
DATA_INTERVAL = os.getenv('DATA_INTERVAL', '1d')

# Trading Parameters
INITIAL_CAPITAL = float(os.getenv('INITIAL_CAPITAL', '100000'))
COMMISSION_RATE = float(os.getenv('COMMISSION_RATE', '0.001'))

# Default LLM Settings
DEFAULT_MODEL_PROVIDER = os.getenv('DEFAULT_MODEL_PROVIDER', 'openai')
DEFAULT_MODEL = os.getenv('DEFAULT_MODEL', 'gpt-3.5-turbo')
DEFAULT_TEMPERATURE = float(os.getenv('DEFAULT_TEMPERATURE', '0.7'))
DEFAULT_MAX_TOKENS = int(os.getenv('DEFAULT_MAX_TOKENS', '2000'))

# Agent-specific LLM Settings
AGENT_SETTINGS: Dict[str, Dict[str, Any]] = {
    'research_agent': {
        'model': os.getenv('RESEARCH_AGENT_MODEL', 'gpt-4'),
        'temperature': float(os.getenv('RESEARCH_AGENT_TEMPERATURE', '0.7')),
        'max_tokens': int(os.getenv('RESEARCH_AGENT_MAX_TOKENS', '2000'))
    },
    'universe_agent': {
        'model': os.getenv('UNIVERSE_AGENT_MODEL', 'gpt-3.5-turbo'),
        'temperature': float(os.getenv('UNIVERSE_AGENT_TEMPERATURE', '0.5')),
        'max_tokens': int(os.getenv('UNIVERSE_AGENT_MAX_TOKENS', '1000'))
    },
    'strategy_agent': {
        'model': os.getenv('STRATEGY_AGENT_MODEL', 'gpt-3.5-turbo'),
        'temperature': float(os.getenv('STRATEGY_AGENT_TEMPERATURE', '0.3')),
        'max_tokens': int(os.getenv('STRATEGY_AGENT_MAX_TOKENS', '1000'))
    },
    'risk_agent': {
        'model': os.getenv('RISK_AGENT_MODEL', 'gpt-3.5-turbo'),
        'temperature': float(os.getenv('RISK_AGENT_TEMPERATURE', '0.3')),
        'max_tokens': int(os.getenv('RISK_AGENT_MAX_TOKENS', '1000'))
    },
    'play_agent': {
        'model': os.getenv('PLAY_AGENT_MODEL', 'gpt-4'),
        'temperature': float(os.getenv('PLAY_AGENT_TEMPERATURE', '0.7')),
        'max_tokens': int(os.getenv('PLAY_AGENT_MAX_TOKENS', '2000'))
    }
}

# Email Settings (Optional)
EMAIL_SETTINGS = {
    'smtp_server': os.getenv('SMTP_SERVER'),
    'smtp_port': int(os.getenv('SMTP_PORT', '587')),
    'email_user': os.getenv('EMAIL_USER'),
    'email_password': os.getenv('EMAIL_PASSWORD'),
    'notification_email': os.getenv('NOTIFICATION_EMAIL')
}

def get_agent_settings(agent_name: str) -> Dict[str, Any]:
    """Get the LLM settings for a specific agent."""
    return AGENT_SETTINGS.get(agent_name, {
        'model': DEFAULT_MODEL,
        'temperature': DEFAULT_TEMPERATURE,
        'max_tokens': DEFAULT_MAX_TOKENS
    })

# API Settings
YAHOO_FINANCE_API_KEY = os.getenv('YAHOO_FINANCE_API_KEY')

# Database Settings
DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_PORT = os.getenv('DB_PORT', '5432')
DB_NAME = os.getenv('DB_NAME', 'stock_agents')
DB_USER = os.getenv('DB_USER', 'postgres')
DB_PASSWORD = os.getenv('DB_PASSWORD')

# Application Settings
DEBUG = os.getenv('DEBUG', 'True').lower() == 'true'
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
ENVIRONMENT = os.getenv('ENVIRONMENT', 'development')

# Trading Parameters
DEFAULT_TRADING_HOURS = os.getenv('DEFAULT_TRADING_HOURS', '9:30-16:00')
DEFAULT_TIMEZONE = os.getenv('DEFAULT_TIMEZONE', 'America/New_York')
MAX_POSITION_SIZE = float(os.getenv('MAX_POSITION_SIZE', '100000'))
RISK_PER_TRADE = float(os.getenv('RISK_PER_TRADE', '0.02'))

# Data Storage
DATA_DIR = os.getenv('DATA_DIR', os.path.join(BASE_DIR, 'data'))

# Dashboard Settings
DASHBOARD_PORT = int(os.getenv('DASHBOARD_PORT', '5000'))
DASHBOARD_HOST = os.getenv('DASHBOARD_HOST', '0.0.0.0')
DASHBOARD_DEBUG = os.getenv('DASHBOARD_DEBUG', 'True').lower() == 'true'

# Backtesting Settings
BACKTEST_START_DATE = os.getenv('BACKTEST_START_DATE', '2020-01-01')
BACKTEST_END_DATE = os.getenv('BACKTEST_END_DATE', '2023-12-31')

# Email Settings
EMAIL_NOTIFICATIONS = os.getenv('EMAIL_NOTIFICATIONS', 'True').lower() == 'true'
EMAIL_HOST = os.getenv('EMAIL_HOST', 'smtp.gmail.com')
EMAIL_PORT = int(os.getenv('EMAIL_PORT', '587'))

# Database URL
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}" 