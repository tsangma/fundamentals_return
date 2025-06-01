"""Data Processor for calculating rolling returns per ticker."""
import logging
import polars as pl
import yaml
from typing import List, Optional

# Set up a module-level logger.
logger = logging.getLogger(__name__)

class DataProcessor:
    """
    Data Processor class for handling data operations.
    """

    def __init__(
            self,
            equity_prices_filepath: str,
            fund_prices_filepath: str,
            daily_metrics_filepath: str,
            fundamentals_filepath: str,
            is_test: bool = False) -> None:
        """
        Initialize the DataProcessor with a Polars DataFrame.
        """
        try:
            self.data = pl.DataFrame()  # Placeholder for final processed data
            self.dataframes = {
                'equity_prices': pl.read_csv(equity_prices_filepath),
                'fund_prices': pl.read_csv(fund_prices_filepath),
                'daily_metrics': pl.read_csv(daily_metrics_filepath),
                'fundamentals': pl.read_csv(fundamentals_filepath),
            }
            if is_test:
                self.dataframes['equity_prices'] = self.dataframes['equity_prices'].filter(
                    pl.col('ticker').is_in(['AAPL', 'MSFT', 'GOOGL'])
                )
                self.dataframes['fundamentals'] = self.dataframes['fundamentals'].filter(
                    pl.col('ticker').is_in(['AAPL', 'MSFT', 'GOOGL'])
                )
                self.dataframes['daily_metrics'] = self.dataframes['daily_metrics'].filter(
                    pl.col('ticker').is_in(['AAPL', 'MSFT', 'GOOGL'])
                )
            logger.info("DataProcessor initialized with data from provided file paths.")
        except IOError as e:
            logger.error("Error reading data files: %s", e)
            raise
        self.monthly_datframe = pl.DataFrame()  # Placeholder for monthly data
        logger.debug("DataProcessor initialized")
    
    def initial_column_selection(self) -> None:
        """
        Initial column selection.
        """
        try:
            with open('config.yaml', encoding='UTF-8', mode='r') as file:
                config = yaml.safe_load(file)
        except FileNotFoundError as e:
            logger.error("Configuration file not found: %s", e)
            raise
        for df in self.dataframes:
            columns = config['dataframes'][df]['columns']
            self.dataframes[df] = self.dataframes[df].select(columns)
        
    
    def filter_for_selected_date_range(self) -> None:
        """
        Initial cleaning.
        """
        start_date = pl.datetime(year=2001, month=1, day=1)
        end_date = pl.datetime(year=2025, month=5, day=1)

        self.data = self.data.filter(
            (pl.col('date') >= start_date) & (pl.col('date') <= end_date)
        )
    
        
    def initiate_final_dataframe(self) -> None:
        """
        Initializes the final data DataFrame by merging all dataframes.
        """
        logger.info("Merging all dataframes into final datframe")
        equity_prices = self.dataframes['equity_prices']
        fundamentals = self.dataframes['fundamentals']
        daily_metrics = self.dataframes['daily_metrics']
        fund_prices = (
            self.dataframes['fund_prices']
            .filter(pl.col('ticker') == 'VTI')
            )
        logger.debug("Initial equity prices data shape: %s", self.dataframes['equity_prices'].shape)
        self.data = equity_prices.extend(fund_prices)
        logger.debug("Data shape after fund prices added: %s", self.data.shape)
        self.data = self.data.join(daily_metrics, on=['ticker', 'date'], how='left')
        logger.debug("Data after joining daily metrics shape: %s", self.data.shape)
        self.data = self.data.join(fundamentals, left_on=['ticker', 'date'], right_on=['ticker', 'datekey'], how='left')
        logger.debug("Data after joining fundamentals shape: %s", self.data.shape)

    def _create_rolling_return_expression(
            self, column_name: str, window_period: int, ticker_column: str) -> pl.Expr:
        """
        Create a Polars expression for a single rolling return calculation,
        partitioned by the ticker column.
        """
        return_col_name = f"{column_name}_rolling_return_{window_period}p"
        logger.debug(
            "Creating rolling return expression for column '%s' over %s periods (per ticker in '%s'), aliased as '%s'.",
            column_name, window_period, ticker_column, return_col_name
        )
        return (
            (pl.col(column_name).pct_change(n=window_period) + 1)
            .over(ticker_column) # Apply this calculation per ticker
            .alias(return_col_name)
        )

    def _add_rolling_returns_for_columns(
            self,
            df: pl.DataFrame,
            columns_to_process: List[str],
            window_periods: List[int],
            ticker_column: str,
            date_column_for_sorting: Optional[str] = None) -> pl.DataFrame:
        """
        Add rolling returns for specified columns in the DataFrame,
        calculated per ticker.
        """
        if not columns_to_process:
            logger.warning("No columns specified for rolling return calculation. Skipping.")
            return
        if not window_periods:
            logger.warning("No window periods specified for rolling return calculation. Skipping.")
            return
        if ticker_column not in df.columns:
            logger.error("Ticker column '%s' not found in DataFrame.", ticker_column)
            raise ValueError(f"Ticker column '{ticker_column}' not found in DataFrame.")

        logger.info(
            "Adding rolling returns for columns: %s with window periods: %s, grouped by ticker column '%s'.",
            columns_to_process, window_periods, ticker_column
        )

        # Ensure data is sorted correctly for rolling calculations if a date column is provided
        if date_column_for_sorting:
            if date_column_for_sorting not in df.columns:
                logger.error("Date column for sorting '%s' not found in DataFrame.", date_column_for_sorting)
                raise ValueError(f"Date column for sorting '{date_column_for_sorting}' not found.")
            logger.debug("Sorting data by '%s' (within '%s') before rolling calculations.", date_column_for_sorting, ticker_column)
            df = df.sort(ticker_column, date_column_for_sorting)

        expressions_to_add = []
        for col_name in columns_to_process:
            if col_name not in df.columns:
                logger.warning("Column '%s' not found in DataFrame. Skipping for this column.", col_name)
                continue
            for period in window_periods:
                if period <= 0:
                    logger.warning(
                        "Skipping invalid window period %s for column '%s'. Must be positive.",
                        period, col_name
                    )
                    continue
                expr = self._create_rolling_return_expression(col_name, period, ticker_column)
                expressions_to_add.append(expr)
        
        if expressions_to_add:
            df = df.with_columns(expressions_to_add)
            logger.info("Successfully added %s rolling return columns (per ticker).", len(expressions_to_add))
        else:
            logger.info("No valid rolling return expressions were generated to add.")
        return df
    
    def _create_monthly_data(self, df: pl.DataFrame, date_column: str = 'date') -> pl.DataFrame:
        """
        Converts daily data to monthly snapshots
        """
        logger.info("Creating monthly data from daily data")
        if date_column not in df.columns:
            logger.error("Date column '%s' not found in DataFrame.", date_column)
            raise ValueError(f"Date column '{date_column}' not found in DataFrame.")
        # Group by month and aggregate
        df = (
            df
            .with_columns(pl.col(date_column).dt.truncate("1mo").alias("date_month"))
            .filter(pl.col(date_column) == pl.col("date_month"))  # Keep only the first day of each month
        )
        return df.drop("date_month")
    
    def process_fundamentals(self) -> pl.DataFrame:
        """Process fundamentals data."""
        logger.info("Processing fundamentals data.")
        logger.debug("Initial fundamentals data shape: %s", self.dataframes['fundamentals'].shape)
        df = self.dataframes['fundamentals']
        df = (
            df
            .filter(pl.col('dimension') == 'ARQ')
        )
        df_prepared = df.clone()
        datekey_dtype = df_prepared["datekey"].dtype

        if datekey_dtype == pl.String:
            try:
                df_prepared = df_prepared.with_columns(
                    pl.col("datekey").str.to_date(format="%Y%m%d", strict=True)
                )
            except pl.exceptions.PolarsError as e:
                raise ValueError(
                    f"Failed to parse string 'datekey' column with format YYYYMMDD. "
                    f"Ensure all date strings are in this format. Original error: {e}"
                )
        elif datekey_dtype == pl.Int64:
            try:
                df_prepared = df_prepared.with_columns(
                    pl.col("datekey").cast(pl.String).str.to_date(format="%Y%m%d", strict=True)
                )
            except pl.exceptions.PolarsError as e:
                raise ValueError(
                    f"Failed to parse integer 'datekey' column as YYYYMMDD. "
                    f"Ensure all integer dates represent YYYYMMDD. Original error: {e}"
                )
        elif not isinstance(datekey_dtype, pl.Date):
            raise ValueError(
                f"'datekey' column has an unsupported type: {datekey_dtype}. "
                "Expected pl.String (YYYYMMDD), pl.Int64 (YYYYMMDD), or pl.Date."
            ) from e

        if df_prepared.is_empty():
            return df_prepared

        # --- 2. Define the universal end date ---
        universal_end_date_str = "2025-05-01"
        universal_end_date_lit = pl.lit(universal_end_date_str).str.to_date("%Y-%m-%d")
        scaffold_df = df_prepared.group_by("ticker").agg(
            pl.date_range(
                start=pl.min("datekey"),    # Start is the min date *for the current group*
                end=universal_end_date_lit, # End is the universal end date literal
                interval="1d",
                eager=False                 # Generates a List[Date] for each group
            ).alias("datekey")
        ).explode("datekey") # Turns the List[Date] into multiple rows per ticker
        logger.debug("Scaffold DataFrame shape after date range generation: %s", scaffold_df.shape)

        # --- 4. Join the scaffold DataFrame with the prepared original data ---
        expanded_df = scaffold_df.join(
            df_prepared,
            on=["ticker", "datekey"],
            how="left"
        )

        logger.debug("Expanded fundamentals data shape after join: %s", expanded_df.shape)
        # --- 5. Forward-fill missing values in information columns ---
        info_columns = [
            col_name for col_name in df_prepared.columns if col_name not in ["ticker", "datekey"]
        ]

        if info_columns:
            # Sort by ticker and datekey is essential for correct forward filling within each group.
            expanded_df = expanded_df.sort(["ticker", "datekey"])
            fill_expressions = [pl.col(c).forward_fill().over("ticker") for c in info_columns]
            expanded_df = expanded_df.with_columns(fill_expressions)
        logger.debug("Expanded fundamentals data shape after forward fill: %s", expanded_df.shape)
        
        # Final sort for consistent output ordering.
        expanded_df = expanded_df.sort(["ticker", "datekey"])
        self.dataframes['fundamentals'] = expanded_df

    def find_returns_in_excess_of_market(self, df: pl.DataFrame, market_ticker: str = 'VTI') -> pl.DataFrame:
        """
        Find returns in excess of the market.
        """
        logger.info("Calculating returns in excess of the market for ticker: %s", market_ticker)
        if market_ticker not in self.data['ticker'].unique().to_list():
            logger.error("Market ticker '%s' not found in data.", market_ticker)
            raise ValueError(f"Market ticker '{market_ticker}' not found in data.")

        return_columns = [col for col in df.columns if "return" in col]
        if not return_columns:
            logger.error("No return columns found in data. Ensure rolling returns have been calculated.")
            raise ValueError("No return columns found in data. Ensure rolling returns have been calculated.")

        # Join with the main data to calculate excess returns
        market_returns = (
            df
            .filter(pl.col('ticker') == market_ticker)
            .select(['date'] + return_columns)
            .rename({col: f'market_{col}' for col in return_columns})
        )
        
        logger.debug("Market returns %s", market_returns.head(5))
        
        excess_returns = (
            df
            .join(market_returns, on='date', how='left')
            .with_columns(
                *[
                    (pl.col(col) - pl.col(f'market_{col}')).alias(f'excess_{col}')
                    for col in return_columns
                ]
            )
        )

        logger.info("Excess returns calculated. Data shape: %s", excess_returns.shape)
        return excess_returns
 
    def process_pipeline(self, ticker_column: str = "ticker", date_column: str = "date") -> None:
        """
        Process pipeline.
        """
        logger.info("Starting data processing pipeline.")
        self.initial_column_selection()

        self.dataframes['equity_prices'] = (
            self.dataframes['equity_prices']
            .with_columns(pl.col('date').str.to_date('%Y-%m-%d').alias('date'))
        )
        self.dataframes['fund_prices'] = (
            self.dataframes['fund_prices']
            .with_columns(pl.col('date').str.to_date('%Y-%m-%d').alias('date'))
        )
        self.dataframes['daily_metrics'] = (
            self.dataframes['daily_metrics']
            .with_columns(pl.col('date').str.to_date('%Y-%m-%d').alias('date'))
        )
        self.dataframes['fundamentals'] = (
            self.dataframes['fundamentals']
            .with_columns(pl.col('datekey').str.to_date('%Y-%m-%d').alias('datekey'))
        )

        self.process_fundamentals()
        self.initiate_final_dataframe()
        self.filter_for_selected_date_range()
        logger.info("Initial cleaning and merging completed. Data shape: %s", self.data.shape)
        self.monthly_datframe = self._create_monthly_data(self.data, date_column)

        target_columns = ['close'] # Columns to calculate returns on
        rolling_window_periods = [7, 30, 90, 180, 365]

        self.data = self._add_rolling_returns_for_columns(
            df=self.data,
            columns_to_process=target_columns,
            window_periods=rolling_window_periods,
            ticker_column=ticker_column,
            date_column_for_sorting=date_column # Crucial for correct rolling calculations
        )

        self.monthly_datframe = self._add_rolling_returns_for_columns(
            df=self.monthly_datframe,
            columns_to_process=target_columns,
            window_periods=[1],
            ticker_column=ticker_column,
            date_column_for_sorting=date_column # Crucial for correct rolling calculations
        )

        self.data = self.find_returns_in_excess_of_market(self.data, market_ticker='VTI')
        self.monthly_datframe = self.find_returns_in_excess_of_market(self.monthly_datframe, market_ticker='VTI')

        logger.info("Data processing pipeline completed. Final data shape: %s", self.data.shape)
        # Save the processed data to a file
        self.data.write_parquet(
            'processed_data/final_processed_data.parquet',
            compression='snappy'
        )
        logger.info("Processed data saved to 'processed_data/final_processed_data.parquet'.")
        # Save the monthly data to a file
        self.monthly_datframe.write_parquet(
            'processed_data/monthly_data.parquet',
            compression='snappy'
        )
        logger.info("Monthly data saved to 'processed_data/monthly_data.parquet'.")