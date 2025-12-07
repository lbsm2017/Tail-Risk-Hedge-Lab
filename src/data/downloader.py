"""
Data Downloader Module

Author: L.Bassetti
Downloads and caches price data from Yahoo Finance for tail-risk hedging analysis.
Handles data alignment, missing values, and computes returns.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

try:
    import pandas_datareader as pdr
    FRED_AVAILABLE = True
except ImportError:
    FRED_AVAILABLE = False
    warnings.warn("pandas_datareader not available. FRED data download disabled.")


class DataDownloader:
    """
    Downloads and manages financial data from Yahoo Finance.
    
    Attributes:
        TICKERS (dict): Predefined ticker groups for different asset classes
        prices (pd.DataFrame): Downloaded price data
        returns (pd.DataFrame): Computed returns
    """
    
    TICKERS = {
        'base': ['ACWI'],
        'bonds': ['TLT', 'IEF', 'SHY'],
        'gold': ['GLD'],
        'silver': ['SLV'],
        'crypto': ['BTC-USD', 'ETH-USD'],
        'cta': ['DBMF'],
        'vix': ['^VIX']
    }
    
    def __init__(self, config: dict):
        """
        Initialize DataDownloader.
        
        Args:
            config: Configuration dict with keys: start_date, end_date, cache_path
        """
        self.config = config
        self.start_date = config.get('start_date', '2008-04-01')
        self.end_date = config.get('end_date') or datetime.now().strftime('%Y-%m-%d')
        self.cache_path = config.get('cache_path', 'data/prices.parquet')
        self.prices = None
        self.returns = None
        # Track actual asset inception dates (before any filling/merging)
        self.asset_inception_dates = {}
        
    def download_all(self, progress: bool = True) -> pd.DataFrame:
        """
        Download all tickers from Yahoo Finance.
        
        Args:
            progress: Show progress bar
            
        Returns:
            DataFrame with adjusted close prices for all tickers
        """
        # Flatten all tickers into a single list
        all_tickers = []
        for category, tickers in self.TICKERS.items():
            all_tickers.extend(tickers)
        
        print(f"Downloading data for {len(all_tickers)} tickers from {self.start_date} to {self.end_date}...")
        
        # Download data
        data = yf.download(
            all_tickers,
            start=self.start_date,
            end=self.end_date,
            progress=progress
        )
        
        # Handle different yfinance return formats
        if isinstance(data.columns, pd.MultiIndex):
            # New yfinance format: MultiIndex columns (Price, Ticker)
            if 'Adj Close' in data.columns.get_level_values(0):
                prices = data['Adj Close'].copy()
            elif 'Close' in data.columns.get_level_values(0):
                prices = data['Close'].copy()
            else:
                raise ValueError(f"Could not find price data. Columns: {data.columns}")
        else:
            # Single ticker or old format
            if 'Adj Close' in data.columns:
                prices = data[['Adj Close']].copy()
                prices.columns = all_tickers[:1]
            elif 'Close' in data.columns:
                prices = data[['Close']].copy()
                prices.columns = all_tickers[:1]
            else:
                prices = data.copy()
        
        # Ensure prices is a DataFrame
        if isinstance(prices, pd.Series):
            prices = prices.to_frame()
        
        # Handle VIX (uses Close, not Adj Close)
        if '^VIX' in all_tickers and '^VIX' in prices.columns:
            # VIX data already included, but might need fixing
            pass
        elif '^VIX' in all_tickers:
            vix_data = yf.download('^VIX', start=self.start_date, end=self.end_date, progress=False)
            if not vix_data.empty:
                if isinstance(vix_data.columns, pd.MultiIndex):
                    prices['^VIX'] = vix_data['Close']['^VIX']
                else:
                    prices['^VIX'] = vix_data['Close']
        
        self.prices = prices
        print(f"Downloaded {len(prices)} days of data for {len(prices.columns)} assets")
        
        return self.prices
    
    def align_and_clean(self) -> pd.DataFrame:
        """
        Align dates and handle missing data.
        
        Strategy:
        - Find common date range where base asset (ACWI) has data
        - Forward fill up to 5 days for minor gaps (weekends, holidays)
        - Drop columns that have >50% missing after alignment
        - Keep assets with partial history (crypto, DBMF started later)
        
        Returns:
            Cleaned DataFrame
        """
        if self.prices is None:
            raise ValueError("No data to clean. Run download_all() first.")
        
        df = self.prices.copy()
        
        # First, find where ACWI has valid data (our base asset)
        if 'ACWI' in df.columns:
            acwi_valid = df['ACWI'].first_valid_index()
            if acwi_valid is not None:
                df = df.loc[acwi_valid:]
                print(f"\nAligned to ACWI start date: {acwi_valid.date()}")
        
        # Forward fill up to 5 days for minor gaps (weekends, holidays)
        df = df.ffill(limit=5)
        
        # Report missing data after forward fill
        missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
        print("\nMissing data percentage (after forward fill):")
        for ticker, pct in missing_pct.items():
            if pct > 0:
                print(f"  {ticker}: {pct:.2f}%")
        
        # Only drop columns with >50% missing (not 20% - be more lenient)
        # This allows assets like crypto/DBMF that started later
        threshold = 0.50
        drop_cols = missing_pct[missing_pct > threshold * 100].index.tolist()
        if drop_cols:
            print(f"\nDropping columns with >{threshold*100}% missing: {drop_cols}")
            df = df.drop(columns=drop_cols)
        
        # Ensure ACWI (base) has no missing values - drop those rows
        if 'ACWI' in df.columns:
            df = df[df['ACWI'].notna()]
        
        # For remaining assets, we'll keep NaN values (they'll be handled in analysis)
        # But let's report final stats
        self.prices = df
        print(f"\nFinal dataset: {len(df)} days x {len(df.columns)} assets")
        print(f"Date range: {df.index[0].date()} to {df.index[-1].date()}")
        print(f"Assets: {', '.join(df.columns)}")
        
        return self.prices
    
    def compute_returns(self, freq: str = 'D') -> pd.DataFrame:
        """
        Compute returns from prices.
        
        Args:
            freq: Frequency for returns
                  'D' - Daily (default)
                  'W' - Weekly
                  'M' - Monthly
                  
        Returns:
            DataFrame of returns
        """
        if self.prices is None:
            raise ValueError("No price data. Run download_all() and align_and_clean() first.")
        
        prices = self.prices.copy()
        
        # Resample if needed
        if freq == 'W':
            prices = prices.resample('W-FRI').last()
        elif freq == 'M':
            prices = prices.resample('M').last()
        
        # Compute log returns
        returns = np.log(prices / prices.shift(1))
        
        # Only drop the first row (NaN from shift), keep NaN for assets with partial history
        # Each analysis function will align data as needed
        returns = returns.iloc[1:]
        
        self.returns = returns
        
        # Track actual inception dates for standard assets (before any filling)
        for col in self.returns.columns:
            if col not in self.asset_inception_dates:  # Don't overwrite custom assets
                first_valid = self.returns[col].first_valid_index()
                if first_valid is not None:
                    self.asset_inception_dates[col] = first_valid
        
        print(f"\nComputed {freq} returns: {len(returns)} periods")
        
        return self.returns
    
    def cache_to_disk(self, path: Optional[str] = None) -> None:
        """
        Save prices to disk for faster loading.
        
        Args:
            path: Path to save parquet file. Uses config cache_path if None.
        """
        if self.prices is None:
            raise ValueError("No data to cache. Run download_all() first.")
        
        if path is None:
            path = self.cache_path
        
        # Create directory if needed
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save to parquet
        self.prices.to_parquet(path)
        print(f"\nCached data to {path}")
        
        # Also save metadata
        metadata = {
            'start_date': self.start_date,
            'end_date': self.end_date,
            'n_assets': len(self.prices.columns),
            'n_periods': len(self.prices),
            'date_range': f"{self.prices.index[0].date()} to {self.prices.index[-1].date()}"
        }
        
        metadata_df = pd.Series(metadata)
        metadata_path = path.replace('.parquet', '_metadata.csv')
        metadata_df.to_csv(metadata_path)
        print(f"Saved metadata to {metadata_path}")
    
    def load_from_cache(self, path: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load prices from cached parquet file.
        
        Args:
            path: Path to parquet file. Uses config cache_path if None.
            
        Returns:
            Tuple of (prices DataFrame, returns DataFrame)
        """
        if path is None:
            path = self.cache_path
            
        if not Path(path).exists():
            raise FileNotFoundError(f"Cache file not found: {path}")
        
        self.prices = pd.read_parquet(path)
        print(f"Loaded cached data from {path}")
        print(f"Date range: {self.prices.index[0].date()} to {self.prices.index[-1].date()}")
        print(f"Assets: {len(self.prices.columns)}")
        
        # Compute returns
        self.returns = self.compute_returns(freq='D')
        
        return self.prices, self.returns
    
    def download_risk_free_rate(
        self,
        ticker: str = 'DGS3MO',
        cache_path: str = 'data/risk_free_rate.parquet',
        use_cache: bool = True
    ) -> pd.Series:
        """
        Download US Treasury risk-free rate from FRED.
        
        Args:
            ticker: FRED ticker symbol (default: DGS3MO = 3-month Treasury)
            cache_path: Path to cache the risk-free rate data
            use_cache: Whether to use cached data if available
            
        Returns:
            Series of daily risk-free rates (annualized decimal, e.g., 0.04 = 4%)
        """
        cache_file = Path(cache_path)
        
        # Try to load from cache first
        if use_cache and cache_file.exists():
            try:
                rf_rate = pd.read_parquet(cache_path)
                if isinstance(rf_rate, pd.DataFrame):
                    rf_rate = rf_rate.iloc[:, 0]  # Get first column as Series
                print(f"Loaded risk-free rate from cache: {cache_path}")
                print(f"  Range: {rf_rate.index[0].date()} to {rf_rate.index[-1].date()}")
                print(f"  Mean rate: {rf_rate.mean():.2%}")
                return rf_rate
            except Exception as e:
                print(f"Cache load failed, downloading: {e}")
        
        # Download from FRED
        if not FRED_AVAILABLE:
            raise ImportError(
                "pandas_datareader is required for FRED data. "
                "Install with: pip install pandas-datareader"
            )
        
        print(f"Downloading risk-free rate ({ticker}) from FRED...")
        print(f"  Date range: {self.start_date} to {self.end_date}")
        
        try:
            # Download from FRED (uses config start_date)
            rf_data = pdr.DataReader(ticker, 'fred', self.start_date, self.end_date)
            
            # Convert to Series and clean
            if isinstance(rf_data, pd.DataFrame):
                rf_rate = rf_data.iloc[:, 0]  # Get first column
            else:
                rf_rate = rf_data
            
            # Remove name if it exists
            rf_rate.name = 'risk_free_rate'
            
            # Convert from percentage to decimal (FRED returns percentages like 4.5)
            rf_rate = rf_rate / 100.0
            
            # Check if we have data from requested start_date
            actual_start = rf_rate.first_valid_index()
            requested_start = pd.to_datetime(self.start_date)
            
            if actual_start and actual_start > requested_start:
                # FRED data doesn't go back far enough - backfill with earliest available rate
                days_missing = (actual_start - requested_start).days
                print(f"  Warning: FRED data starts at {actual_start.date()}, requested {requested_start.date()}")
                print(f"  Backfilling {days_missing} days with earliest rate: {rf_rate.loc[actual_start]:.2%}")
                
                # Create date range from requested start to first valid FRED date
                backfill_dates = pd.date_range(start=requested_start, end=actual_start, freq='D', inclusive='left')
                backfill_values = pd.Series(rf_rate.loc[actual_start], index=backfill_dates)
                
                # Prepend backfilled data
                rf_rate = pd.concat([backfill_values, rf_rate])
            
            # Handle missing data with linear interpolation
            missing_before = rf_rate.isnull().sum()
            if missing_before > 0:
                rf_rate = rf_rate.interpolate(method='linear', limit_direction='both')
                missing_after = rf_rate.isnull().sum()
                print(f"  Interpolated {missing_before - missing_after} missing values")
            
            # Still have NaN at edges? Forward/backward fill
            if rf_rate.isnull().any():
                rf_rate = rf_rate.fillna(method='ffill').fillna(method='bfill')
            
            print(f"  Downloaded {len(rf_rate)} observations")
            print(f"  Mean rate: {rf_rate.mean():.2%}")
            print(f"  Range: {rf_rate.min():.2%} to {rf_rate.max():.2%}")
            
            # Cache to disk
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            rf_rate.to_frame().to_parquet(cache_path)
            print(f"  Cached to {cache_path}")
            
            return rf_rate
            
        except Exception as e:
            raise RuntimeError(f"Failed to download risk-free rate from FRED: {e}")
    
    def get_asset_info(self) -> pd.DataFrame:
        """
        Get summary information about downloaded assets.
        
        Returns:
            DataFrame with asset statistics
        """
        if self.prices is None:
            raise ValueError("No data loaded.")
        
        info = []
        for ticker in self.prices.columns:
            data = self.prices[ticker].dropna()
            
            # Determine category
            category = 'Other'
            for cat, tickers in self.TICKERS.items():
                if ticker in tickers:
                    category = cat
                    break
            
            info.append({
                'Ticker': ticker,
                'Category': category,
                'Start Date': data.index[0].date(),
                'End Date': data.index[-1].date(),
                'N Obs': len(data),
                'Missing %': (self.prices[ticker].isnull().sum() / len(self.prices) * 100).round(2),
                'First Price': data.iloc[0].round(2),
                'Last Price': data.iloc[-1].round(2)
            })
        
        return pd.DataFrame(info)
    
    def load_custom_data(self, import_folder: str = 'data/import') -> Dict[str, pd.Series]:
        """
        Load custom return data from Excel files in the import folder.
        
        Each Excel file should have columns: Date, Ret % (or similar)
        The filename (without extension) becomes the asset name.
        
        Supports both daily and monthly data - will be resampled to daily.
        
        Args:
            import_folder: Path to folder containing Excel files
            
        Returns:
            Dictionary of {asset_name: returns_series}
        """
        import_path = Path(import_folder)
        custom_data = {}
        
        if not import_path.exists():
            print(f"Import folder not found: {import_folder}")
            return custom_data
        
        # Find all Excel files
        excel_files = list(import_path.glob('*.xlsx')) + list(import_path.glob('*.xls'))
        
        if not excel_files:
            print(f"No Excel files found in {import_folder}")
            return custom_data
        
        print(f"\nLoading {len(excel_files)} custom data files from {import_folder}...")
        
        for filepath in excel_files:
            asset_name = filepath.stem  # Filename without extension
            
            try:
                # Read Excel file
                df = pd.read_excel(filepath)
                
                # Find the date column
                date_col = None
                for col in df.columns:
                    if 'date' in col.lower():
                        date_col = col
                        break
                if date_col is None:
                    date_col = df.columns[0]  # Assume first column is date
                
                # Find the return column
                ret_col = None
                for col in df.columns:
                    if 'ret' in col.lower() or '%' in col.lower():
                        ret_col = col
                        break
                if ret_col is None:
                    ret_col = df.columns[1]  # Assume second column is returns
                
                # Parse data
                df[date_col] = pd.to_datetime(df[date_col])
                df = df.set_index(date_col)
                
                # Convert returns to decimal if they're in percentage format
                returns = df[ret_col].copy()
                
                # Check if returns are in percentage format (values > 1 or < -1 commonly)
                if returns.abs().max() > 1:
                    # Likely percentage format like 1.9 for 1.9%
                    returns = returns / 100
                
                # Handle string percentages like "1.9%"
                if returns.dtype == object:
                    returns = returns.str.replace('%', '').astype(float) / 100
                
                returns = returns.dropna()
                returns.name = asset_name
                
                # Store the returns
                custom_data[asset_name] = returns
                
                # Detect frequency
                if len(returns) > 1:
                    avg_diff = (returns.index[1:] - returns.index[:-1]).mean()
                    if avg_diff.days > 20:
                        freq = 'Monthly'
                    elif avg_diff.days > 5:
                        freq = 'Weekly'
                    else:
                        freq = 'Daily'
                else:
                    freq = 'Unknown'
                
                print(f"  Loaded: {asset_name} ({freq}, {len(returns)} periods, "
                      f"{returns.index[0].date()} to {returns.index[-1].date()})")
                
            except Exception as e:
                print(f"  Error loading {filepath.name}: {e}")
        
        return custom_data
    
    def merge_custom_data(self, custom_returns: Dict[str, pd.Series]) -> None:
        """
        Merge custom return data into the main returns DataFrame.
        
        Custom data can be at different frequencies (monthly/daily).
        Monthly data will be kept as-is and aligned to month-end dates.
        
        Args:
            custom_returns: Dictionary of {asset_name: returns_series}
        """
        if self.returns is None:
            raise ValueError("No returns data. Run compute_returns() first.")
        
        if not custom_returns:
            return
        
        print(f"\nMerging {len(custom_returns)} custom assets into returns data...")
        
        for asset_name, returns in custom_returns.items():
            # Store the ACTUAL inception date before any filling
            if len(returns) > 0:
                self.asset_inception_dates[asset_name] = returns.index[0]
            # Detect if monthly or daily
            if len(returns) > 1:
                avg_diff = (returns.index[1:] - returns.index[:-1]).mean()
                is_monthly = avg_diff.days > 20
            else:
                is_monthly = False
            
            if is_monthly:
                # Resample main returns to monthly for comparison
                # But for backtesting, we'll expand monthly returns to daily
                # by assigning the monthly return to the last day of each month
                
                # Create a daily series with the monthly returns on month-end dates
                daily_returns = pd.Series(index=self.returns.index, dtype=float)
                daily_returns[:] = 0.0  # Default to 0
                
                for date, ret in returns.items():
                    # Find the closest date in our index
                    mask = self.returns.index.to_period('M') == date.to_period('M')
                    if mask.any():
                        # Put the monthly return on the last day of that month in our data
                        month_dates = self.returns.index[mask]
                        last_day = month_dates[-1]
                        daily_returns[last_day] = ret
                
                self.returns[asset_name] = daily_returns
                print(f"  Added {asset_name} (monthly -> daily, {(daily_returns != 0).sum()} months)")
            else:
                # Daily data - just merge directly
                self.returns[asset_name] = returns.reindex(self.returns.index)
                n_valid = self.returns[asset_name].notna().sum()
                print(f"  Added {asset_name} (daily, {n_valid} periods)")
        
        # Also create synthetic prices for reporting (not needed for returns-based analysis)
        if self.prices is not None:
            for asset_name in custom_returns.keys():
                if asset_name in self.returns.columns:
                    # Create synthetic price series starting at 100
                    ret_series = self.returns[asset_name].fillna(0)
                    price_series = 100 * np.exp(ret_series.cumsum())
                    self.prices[asset_name] = price_series
    
    def get_overlap_periods(self) -> Dict[str, Tuple[str, str]]:
        """
        Identify common date ranges where multiple assets have data.
        
        Returns:
            Dictionary with period names and date ranges
        """
        if self.prices is None:
            raise ValueError("No data loaded.")
        
        periods = {}
        
        # Full period (all assets with some data)
        periods['full'] = (
            self.prices.index[0].strftime('%Y-%m-%d'),
            self.prices.index[-1].strftime('%Y-%m-%d')
        )
        
        # ACWI period (2008+)
        if 'ACWI' in self.prices.columns:
            acwi_data = self.prices['ACWI'].dropna()
            periods['acwi'] = (
                acwi_data.index[0].strftime('%Y-%m-%d'),
                acwi_data.index[-1].strftime('%Y-%m-%d')
            )
        
        # Bitcoin period (2014+)
        if 'BTC-USD' in self.prices.columns:
            btc_data = self.prices['BTC-USD'].dropna()
            if len(btc_data) > 0:
                periods['crypto'] = (
                    btc_data.index[0].strftime('%Y-%m-%d'),
                    btc_data.index[-1].strftime('%Y-%m-%d')
                )
        
        # CTA period (2019+)
        if 'DBMF' in self.prices.columns:
            cta_data = self.prices['DBMF'].dropna()
            if len(cta_data) > 0:
                periods['cta'] = (
                    cta_data.index[0].strftime('%Y-%m-%d'),
                    cta_data.index[-1].strftime('%Y-%m-%d')
                )
        
        return periods


# Convenience function for quick data loading
def quick_download(start_date: str = '2008-04-01', 
                   end_date: Optional[str] = None,
                   cache: bool = True,
                   cache_path: str = 'data/prices.parquet') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Quick data download and processing.
    
    Args:
        start_date: Start date
        end_date: End date (None for today)
        cache: Whether to cache to disk
        cache_path: Path for cache file
        
    Returns:
        Tuple of (prices, returns)
    """
    downloader = DataDownloader(start_date, end_date)
    
    # Try loading from cache first
    if cache and Path(cache_path).exists():
        try:
            print("Attempting to load from cache...")
            downloader.load_from_cache(cache_path)
        except Exception as e:
            print(f"Cache load failed: {e}")
            print("Downloading fresh data...")
            downloader.download_all()
            downloader.align_and_clean()
            if cache:
                downloader.cache_to_disk(cache_path)
    else:
        downloader.download_all()
        downloader.align_and_clean()
        if cache:
            downloader.cache_to_disk(cache_path)
    
    returns = downloader.compute_returns('D')
    
    return downloader.prices, returns
