"""Debug script to understand CVaR calculation differences."""
import pandas as pd
import numpy as np
from src.metrics.tail_risk import cvar, max_drawdown, cagr

# Load data directly from parquet
prices = pd.read_parquet('data/prices.parquet')
returns = prices.pct_change().dropna()

# Load custom data
import glob
import os
from src.data.downloader import DataDownloader

# Get ACWI
acwi = returns['ACWI'].dropna()

# Try to load MAN AHL from Excel
excel_files = glob.glob('data/import/*.xlsx')
man_ahl = None
for f in excel_files:
    if 'MAN' in f.upper() or 'AHL' in f.upper():
        df = pd.read_excel(f)
        if 'Date' in df.columns and 'Price' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.set_index('Date').sort_index()
            man_ahl = df['Price'].pct_change().dropna()
            man_ahl.name = 'MAN AHL Evolution'
            break

if man_ahl is None:
    print("Could not find MAN AHL data, using mock monthly data")
    # Create mock monthly data
    dates = pd.date_range('2012-12-31', '2025-06-30', freq='ME')
    man_ahl = pd.Series(np.random.randn(len(dates)) * 0.02, index=dates, name='MAN AHL Evolution')

print("=" * 60)
print("DATA OVERVIEW")
print("=" * 60)
print(f"ACWI: {len(acwi)} observations ({acwi.index[0]} to {acwi.index[-1]})")
print(f"MAN AHL: {len(man_ahl)} observations ({man_ahl.index[0]} to {man_ahl.index[-1]})")

# Align data
aligned = pd.concat([acwi, man_ahl], axis=1).dropna()
acwi_aligned = aligned['ACWI']
print(f"\nAligned: {len(acwi_aligned)} observations ({acwi_aligned.index[0]} to {acwi_aligned.index[-1]})")

# Check frequency
avg_days = (acwi_aligned.index[-1] - acwi_aligned.index[0]).days / (len(acwi_aligned) - 1)
print(f"Avg days between observations: {avg_days:.1f} (monthly if > 20)")

print("\n" + "=" * 60)
print("CVaR COMPARISON")
print("=" * 60)

# CVaR with daily data (full date range)
date_start = acwi_aligned.index[0]
date_end = acwi_aligned.index[-1]
acwi_daily = acwi.loc[date_start:date_end]
print(f"\nACWI Daily (full): {len(acwi_daily)} observations")

cvar_daily_monthly_freq = cvar(acwi_daily, alpha=0.95, frequency='monthly')
cvar_daily_daily_freq = cvar(acwi_daily, alpha=0.95, frequency='daily')
print(f"  CVaR (monthly resampled): {cvar_daily_monthly_freq*100:.2f}%")
print(f"  CVaR (daily freq): {cvar_daily_daily_freq*100:.2f}%")

# CVaR with aligned monthly data
cvar_aligned_monthly_freq = cvar(acwi_aligned, alpha=0.95, frequency='monthly')
cvar_aligned_daily_freq = cvar(acwi_aligned, alpha=0.95, frequency='daily')
print(f"\nACWI Aligned (monthly): {len(acwi_aligned)} observations")
print(f"  CVaR (monthly resampled): {cvar_aligned_monthly_freq*100:.2f}%")
print(f"  CVaR (daily freq): {cvar_aligned_daily_freq*100:.2f}%")

print("\n" + "=" * 60)
print("CAGR COMPARISON")
print("=" * 60)

cagr_daily = cagr(acwi_daily.values, periods_per_year=252) * 100
cagr_aligned_daily = cagr(acwi_aligned.values, periods_per_year=252) * 100
cagr_aligned_monthly = cagr(acwi_aligned.values, periods_per_year=12) * 100

print(f"ACWI Daily CAGR (252 periods/year): {cagr_daily:.2f}%")
print(f"ACWI Aligned CAGR (252 periods/year - WRONG): {cagr_aligned_daily:.2f}%")
print(f"ACWI Aligned CAGR (12 periods/year): {cagr_aligned_monthly:.2f}%")

print("\n" + "=" * 60)
print("MAX DRAWDOWN COMPARISON")
print("=" * 60)

mdd_daily, _, _ = max_drawdown((1 + acwi_daily).cumprod())
mdd_aligned, _, _ = max_drawdown((1 + acwi_aligned).cumprod())

print(f"ACWI Daily MDD: {mdd_daily*100:.2f}%")
print(f"ACWI Aligned MDD: {mdd_aligned*100:.2f}%")

print("\n" + "=" * 60)
print("CONCLUSION")
print("=" * 60)
print("""
The issue is clear:
- CVaR on daily data with monthly resampling ≈ 9.44%
- CVaR on aligned monthly data ≈ 1.70% (much lower!)

The optimizer uses aligned monthly data, achieving 25% reduction
from 1.70% to ~1.28%. But the report uses daily data (9.44% baseline)
showing 82% reduction (9.44% to 1.70%).

FIX: Both optimizer and report must use the SAME baseline.
""")
