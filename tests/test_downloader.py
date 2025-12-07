"""
Test script for Data Downloader module

Author: L.Bassetti
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.data.downloader import DataDownloader, quick_download


def test_downloader():
    """Test basic downloader functionality."""
    
    print("=" * 70)
    print("Testing Data Downloader")
    print("=" * 70)
    
    # Initialize downloader
    downloader = DataDownloader(start_date='2020-01-01', end_date='2024-12-31')
    print(f"\n✓ DataDownloader initialized")
    print(f"  Start: {downloader.start_date}")
    print(f"  End: {downloader.end_date}")
    
    # Download data
    print("\n1. Downloading data...")
    prices = downloader.download_all(progress=False)
    print(f"✓ Downloaded {prices.shape[0]} days × {prices.shape[1]} assets")
    
    # Clean data
    print("\n2. Cleaning data...")
    clean_prices = downloader.align_and_clean()
    print(f"✓ Cleaned data shape: {clean_prices.shape}")
    
    # Compute returns
    print("\n3. Computing returns...")
    returns = downloader.compute_returns('D')
    print(f"✓ Returns shape: {returns.shape}")
    
    # Get asset info
    print("\n4. Asset information:")
    info = downloader.get_asset_info()
    print(info.to_string(index=False))
    
    # Cache data
    print("\n5. Caching data...")
    downloader.cache_to_disk('data/test_prices.parquet')
    print("✓ Data cached")
    
    print("\n" + "=" * 70)
    print("Data Downloader Test: PASSED")
    print("=" * 70)
    
    return downloader


if __name__ == "__main__":
    test_downloader()
