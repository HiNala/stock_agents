import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

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
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.getenv('DATA_DIR', os.path.join(BASE_DIR, 'data'))
CACHE_DIR = os.getenv('CACHE_DIR', os.path.join(DATA_DIR, 'cache'))
LOG_DIR = os.getenv('LOG_DIR', os.path.join(BASE_DIR, 'logs'))

# Create necessary directories
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Dashboard Settings
DASHBOARD_PORT = int(os.getenv('DASHBOARD_PORT', '5000'))
DASHBOARD_HOST = os.getenv('DASHBOARD_HOST', '0.0.0.0')
DASHBOARD_DEBUG = os.getenv('DASHBOARD_DEBUG', 'True').lower() == 'true'

# Backtesting Settings
BACKTEST_START_DATE = os.getenv('BACKTEST_START_DATE', '2020-01-01')
BACKTEST_END_DATE = os.getenv('BACKTEST_END_DATE', '2023-12-31')
INITIAL_CAPITAL = float(os.getenv('INITIAL_CAPITAL', '100000'))
COMMISSION_RATE = float(os.getenv('COMMISSION_RATE', '0.001'))

# Email Settings
EMAIL_NOTIFICATIONS = os.getenv('EMAIL_NOTIFICATIONS', 'True').lower() == 'true'
EMAIL_HOST = os.getenv('EMAIL_HOST', 'smtp.gmail.com')
EMAIL_PORT = int(os.getenv('EMAIL_PORT', '587'))
EMAIL_USER = os.getenv('EMAIL_USER')
EMAIL_PASSWORD = os.getenv('EMAIL_PASSWORD')

# Database URL
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}" 