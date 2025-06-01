"""Run Pipeline Script"""

import logging
from data_processor.data_processor import DataProcessor
import polars as pl

logging.basicConfig(
    level=logging.DEBUG,  # Set the minimum logging level
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"), # Log to a file
        logging.StreamHandler()         # Log to the console
    ]
)

logger = logging.getLogger(__name__) # Get a logger for the main module

def run_pipeline():
    equity_prices = 'sharadar_data_20250531/sharadar_equity_prices_SHARADAR_SEP.csv'
    fundamentals = 'sharadar_data_20250531/core_us_fundamentals_SHARADAR_SF1.csv'
    daily_metrics = 'sharadar_data_20250531/daily_metrics_SHARADAR_DAILY.csv'
    monthly_prices = 'sharadar_data_20250531/fund_prices_SHARADAR_SFP.csv'

    data_processor = DataProcessor(
        equity_prices_filepath=equity_prices,
        fund_prices_filepath=monthly_prices,
        daily_metrics_filepath=daily_metrics,
        fundamentals_filepath=fundamentals,
        is_test=False
    )
    data_processor.process_pipeline(ticker_column='ticker', date_column='date')

if __name__ == "__main__":
    run_pipeline()