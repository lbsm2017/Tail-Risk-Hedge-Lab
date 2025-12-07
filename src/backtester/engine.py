"""
Backtesting Engine Module

Author: L.Bassetti
Main orchestration for backtesting hedge strategies.
Optimized with parallel processing for performance.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import yaml
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing
from functools import partial
import time

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

from ..data.downloader import DataDownloader
from ..regime.detector import RegimeDetector
from ..metrics.tail_risk import compute_all_metrics
from ..metrics.correlations import correlation_breakdown
from ..optimization.weight_finder import optimize_for_multiple_targets
from ..optimization.multi_asset import (
    optimize_multi_asset_cvar,
    greedy_sequential_allocation,
    portfolio_analytics
)
from ..hypothesis.tests import comprehensive_hypothesis_tests
from .rebalancing import (
    simulate_rebalanced_portfolio,
    get_rebalancing_summary,
    get_asset_inception_dates,
    get_portfolio_data_window
)


# Module-level function for parallel processing (can't pickle instance methods)
def _analyze_hedge_worker(args: Tuple) -> Tuple[str, Dict]:
    """
    Worker function to analyze a single hedge asset.
    Designed to be called in parallel.
    Uses all available overlapping data between base and hedge.
    """
    (ticker, base_returns, hedge_returns, regime_labels, 
     targets, max_weight, weight_step, alpha, 
     n_bootstrap, hypothesis_alpha, target_reduction, rf_rate, cvar_frequency, tie_break_tolerance) = args
    
    # Align data - use all available overlapping periods
    aligned = pd.DataFrame({
        'base': base_returns,
        'hedge': hedge_returns,
        'regime': regime_labels
    }).dropna()
    
    if len(aligned) < 252:  # Need at least 1 year of data
        return ticker, {
            'ticker': ticker,
            'correlations': {},
            'optimization': [],
            'hypothesis_tests': {},
            'optimal_weight': 0.0,
            'hedged_metrics': {},
            'data_info': {'periods': len(aligned), 'error': 'Insufficient data'}
        }
    
    aligned_base = aligned['base']
    aligned_hedge = aligned['hedge']
    aligned_regime = aligned['regime']
    
    # Correlation analysis
    corr_results = correlation_breakdown(
        equity_ret=aligned_base,
        hedge_ret=aligned_hedge,
        regime_labels=aligned_regime
    )
    
    # Optimization for multiple targets
    opt_results = optimize_for_multiple_targets(
        base_returns=aligned_base,
        hedge_returns=aligned_hedge,
        targets=targets,
        metrics=['cvar', 'mdd'],
        max_weight=max_weight,
        weight_step=weight_step,
        alpha=alpha,
        cvar_frequency=cvar_frequency,
        tie_break_tolerance=tie_break_tolerance
    )
    
    # Hypothesis tests for primary target
    primary_target = opt_results[
        (opt_results['metric'] == 'cvar') & 
        (opt_results['target_reduction'] == target_reduction)
    ]
    
    if len(primary_target) > 0:
        optimal_weight = primary_target.iloc[0]['optimal_weight']
        
        hypothesis_results = comprehensive_hypothesis_tests(
            base_returns=aligned_base,
            hedge_returns=aligned_hedge,
            hedge_weight=optimal_weight,
            regime_labels=aligned_regime,
            n_bootstrap=n_bootstrap,
            alpha=hypothesis_alpha,
            cvar_frequency=cvar_frequency
        )
    else:
        optimal_weight = 0.0
        hypothesis_results = {}
    
    # Compute hedged metrics
    if optimal_weight > 0:
        hedged_returns = (1 - optimal_weight) * aligned_base + optimal_weight * aligned_hedge
        # Align risk-free rate to hedged returns
        aligned_rf = rf_rate.reindex(hedged_returns.index, method='ffill').mean()
        hedged_metrics = compute_all_metrics(hedged_returns, rf_rate=aligned_rf, 
                                             cvar_frequency=cvar_frequency)
    else:
        hedged_metrics = {}
    
    # Add data info for reporting
    data_info = {
        'periods': len(aligned),
        'start_date': str(aligned.index[0].date()),
        'end_date': str(aligned.index[-1].date())
    }
    
    return ticker, {
        'ticker': ticker,
        'correlations': corr_results,
        'optimization': opt_results.to_dict('records'),
        'hypothesis_tests': hypothesis_results,
        'optimal_weight': optimal_weight,
        'hedged_metrics': hedged_metrics,
        'data_info': data_info
    }


class Backtester:
    """
    Main backtesting engine for tail-risk hedging strategies.
    Optimized with parallel processing.
    """
    
    def __init__(self, config_path: Union[str, dict] = 'config.yaml', n_workers: int = None):
        """
        Initialize backtester with configuration.
        
        Args:
            config_path: Path to YAML configuration file or dict with config
            n_workers: Number of parallel workers (default: CPU count)
        """
        if isinstance(config_path, dict):
            self.config = config_path
        else:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        
        self.downloader = DataDownloader(self.config['data'])
        self.regime_detector = RegimeDetector(self.config['regime'])
        
        self.prices = None
        self.returns = None
        self.regime_labels = None
        self.vix = None
        self.risk_free_rate = None  # Will store Series of daily risk-free rates
        self.mean_rf_rate = 0.0     # Mean annualized risk-free rate for reporting
        
        # Number of parallel workers (-1 or None = all CPUs)
        if n_workers is None or n_workers == -1:
            self.n_workers = multiprocessing.cpu_count()
        else:
            self.n_workers = n_workers
        
        # Parse hedges config into dicts
        self.hedge_weights = {}
        self.hedge_names = {}
        for hedge in self.config['assets']['hedges']:
            ticker = hedge['ticker']
            self.hedge_weights[ticker] = hedge.get('max_weight', 0.50)
            self.hedge_names[ticker] = hedge.get('name', ticker)
        
        # Individual asset analysis constraint: complement of base_min_weight_individual
        # All individual hedge tests can go up to this limit (e.g., 50%)
        self.individual_max_hedge_weight = 1.0 - self.config['assets'].get('base_min_weight_individual', 0.50)
        
    def load_data(self, use_cache: bool = True) -> None:
        """
        Load and prepare data, including custom Excel files from data/import.
        
        Args:
            use_cache: Whether to use cached data
        """
        print("Loading data...")
        
        if use_cache:
            try:
                self.prices, self.returns = self.downloader.load_from_cache()
                print(f"Loaded from cache: {len(self.prices)} days")
            except FileNotFoundError:
                print("Cache not found, downloading...")
                use_cache = False
        
        if not use_cache:
            self.downloader.download_all(full_config=self.config)
            self.prices = self.downloader.align_and_clean()
            self.returns = self.downloader.compute_returns(freq='D')
            self.downloader.cache_to_disk()
            print(f"Downloaded: {len(self.prices)} days")
        
        # Load custom data from Excel files
        import_folder = self.config.get('data', {}).get('import_folder', 'data/import')
        custom_data = self.downloader.load_custom_data(import_folder)
        if custom_data:
            self.downloader.merge_custom_data(custom_data)
            self.prices = self.downloader.prices
            self.returns = self.downloader.returns
            
            # Add custom assets to hedge config if not already there
            # Use multi-asset max weight for custom assets in portfolio construction
            default_max_weight = self.config.get('data', {}).get('custom_assets_max_weight_multi', 0.15)
            for asset_name in custom_data.keys():
                if asset_name not in self.hedge_weights:
                    self.hedge_weights[asset_name] = default_max_weight
                    self.hedge_names[asset_name] = asset_name  # Use filename as display name
                    print(f"  Added custom hedge: {asset_name} (multi-asset max: {default_max_weight:.0%})")
        
        # Extract VIX if available
        if '^VIX' in self.prices.columns:
            self.vix = self.prices['^VIX']
        
        # Load risk-free rate
        self._load_risk_free_rate()
        
        print(f"Date range: {self.prices.index[0]} to {self.prices.index[-1]}")
        print(f"Assets: {', '.join(self.prices.columns)}")
    
    def _load_risk_free_rate(self) -> None:
        """
        Load risk-free rate from FRED or use static value.
        Aligns to returns date range and converts to daily.
        """
        rf_config = self.config.get('metrics', {}).get('risk_free_rate', {})
        source = rf_config.get('source', 'static')
        
        if source == 'FRED':
            try:
                ticker = rf_config.get('ticker', 'DGS3MO')
                cache_path = rf_config.get('cache_path', 'data/risk_free_rate.parquet')
                
                # Download risk-free rate (annualized)
                rf_annual = self.downloader.download_risk_free_rate(
                    ticker=ticker,
                    cache_path=cache_path,
                    use_cache=True
                )
                
                # Align to returns date range
                rf_annual = rf_annual.reindex(self.returns.index, method='ffill')
                
                # Store both annualized (for metadata) and mean
                self.risk_free_rate = rf_annual
                self.mean_rf_rate = rf_annual.mean()
                
                print(f"\nRisk-free rate loaded from FRED ({ticker})")
                print(f"  Mean rate: {self.mean_rf_rate:.2%}")
                print(f"  Range: {rf_annual.min():.2%} to {rf_annual.max():.2%}")
                
            except Exception as e:
                print(f"\nWarning: Failed to load FRED data: {e}")
                print("Falling back to static risk-free rate")
                static_rate = rf_config.get('static_value', 0.04)
                self.risk_free_rate = pd.Series(static_rate, index=self.returns.index)
                self.mean_rf_rate = static_rate
        else:
            # Use static value
            static_rate = rf_config.get('static_value', 0.04)
            self.risk_free_rate = pd.Series(static_rate, index=self.returns.index)
            self.mean_rf_rate = static_rate
            print(f"\nUsing static risk-free rate: {self.mean_rf_rate:.2%}")
    
    def identify_regimes(self) -> pd.Series:
        """
        Detect crisis vs normal regimes.
        
        Returns:
            Binary regime series (0=Normal, 1=Crisis)
        """
        print("\nIdentifying regimes...")
        
        base_asset = self.config['assets']['base']
        base_prices = self.prices[base_asset]
        base_returns = self.returns[base_asset]
        
        self.regime_labels = self.regime_detector.detect(
            equity_prices=base_prices,
            equity_returns=base_returns,
            vix=self.vix
        )
        
        # Get regime statistics
        stats = self.regime_detector.get_regime_statistics(self.regime_labels)
        
        print(f"Regime method: {self.config['regime']['method']}")
        print(f"Total periods: {stats['total_periods']}")
        print(f"Crisis periods: {stats['crisis_periods']} ({stats['crisis_pct']:.1f}%)")
        print(f"Crisis episodes: {stats['crisis_episodes']}")
        print(f"Avg crisis length: {stats['avg_crisis_length']:.1f} days")
        
        return self.regime_labels
    
    def analyze_single_hedge(
        self,
        hedge_ticker: str,
        target_reduction: float = 0.25
    ) -> Dict:
        """
        Analyze single hedge asset effectiveness.
        
        Args:
            hedge_ticker: Ticker symbol for hedge asset
            target_reduction: Target risk reduction (e.g., 0.25 for 25%)
            
        Returns:
            Dictionary with analysis results
        """
        base_asset = self.config['assets']['base']
        base_returns = self.returns[base_asset]
        hedge_returns = self.returns[hedge_ticker]
        
        # Correlation analysis
        corr_results = correlation_breakdown(
            equity_ret=base_returns,
            hedge_ret=hedge_returns,
            regime_labels=self.regime_labels
        )
        
        # Optimization for multiple targets
        opt_results = optimize_for_multiple_targets(
            base_returns=base_returns,
            hedge_returns=hedge_returns,
            targets=self.config['optimization']['targets'],
            metrics=['cvar', 'mdd'],
            max_weight=self.individual_max_hedge_weight,
            weight_step=self.config['optimization']['weight_step'],
            alpha=self.config['metrics']['cvar_confidence'],
            tie_break_tolerance=self.config['optimization'].get('tie_break_tolerance', 0.001)
        )
        
        # Hypothesis tests for primary target
        primary_target = opt_results[
            (opt_results['metric'] == 'cvar') & 
            (opt_results['target_reduction'] == target_reduction)
        ]
        
        if len(primary_target) > 0:
            optimal_weight = primary_target.iloc[0]['optimal_weight']
            
            hypothesis_results = comprehensive_hypothesis_tests(
                base_returns=base_returns,
                hedge_returns=hedge_returns,
                hedge_weight=optimal_weight,
                regime_labels=self.regime_labels,
                n_bootstrap=self.config['hypothesis']['n_bootstrap'],
                alpha=self.config['hypothesis']['alpha']
            )
        else:
            optimal_weight = 0.0
            hypothesis_results = {}
        
        # Compute performance metrics
        if optimal_weight > 0:
            hedged_returns = (1 - optimal_weight) * base_returns + optimal_weight * hedge_returns
            # Use mean risk-free rate for single hedge analysis
            hedged_metrics = compute_all_metrics(hedged_returns, rf_rate=self.mean_rf_rate)
        else:
            hedged_metrics = {}
        
        return {
            'ticker': hedge_ticker,
            'correlations': corr_results,
            'optimization': opt_results.to_dict('records'),
            'hypothesis_tests': hypothesis_results,
            'optimal_weight': optimal_weight,
            'hedged_metrics': hedged_metrics
        }
    
    def analyze_all_hedges(self) -> Dict[str, Dict]:
        """
        Analyze all hedge assets in parallel with progress bar.
        
        Returns:
            Dictionary with results for each hedge asset
        """
        start_time = time.time()
        
        base_asset = self.config['assets']['base']
        base_returns = self.returns[base_asset]
        
        # Get available hedge tickers
        hedge_tickers = [
            ticker 
            for ticker in self.hedge_weights.keys()
            if ticker in self.returns.columns
        ]
        
        # Prepare arguments for parallel workers
        worker_args = []
        for ticker in hedge_tickers:
            args = (
                ticker,
                base_returns,
                self.returns[ticker],
                self.regime_labels,
                self.config['optimization']['targets'],
                self.individual_max_hedge_weight,  # Uniform constraint for individual analysis
                self.config['optimization']['weight_step'],
                self.config['metrics']['cvar_confidence'],
                self.config['hypothesis']['n_bootstrap'],
                self.config['hypothesis']['alpha'],
                0.25,  # target_reduction for hypothesis tests
                self.risk_free_rate,  # risk-free rate series
                self.config['metrics'].get('cvar_frequency', 'monthly'),  # CVaR frequency
                self.config['optimization'].get('tie_break_tolerance', 0.001)  # Tie-breaking tolerance
            )
            worker_args.append(args)
        
        # Run in parallel using ThreadPoolExecutor with progress bar
        results = {}
        n_assets = len(worker_args)
        
        if TQDM_AVAILABLE:
            # Create progress bar that shows preparation phase
            with tqdm(total=n_assets + 1, desc="Individual Hedges", 
                     bar_format='{l_bar}{bar:30}{r_bar}{bar:-10b}',
                     colour='green') as pbar:
                
                pbar.set_postfix_str(f"Preparing {n_assets} assets...")
                pbar.update(1)  # Show progress for preparation
                
                with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
                    futures = {
                        executor.submit(_analyze_hedge_worker, args): args[0] 
                        for args in worker_args
                    }
                    
                    for future in as_completed(futures):
                        ticker = futures[future]
                        try:
                            _, result = future.result()
                            results[ticker] = result
                            pbar.set_postfix_str(f"✓ {ticker}")
                            pbar.update(1)
                        except Exception as e:
                            pbar.set_postfix_str(f"✗ {ticker}")
                            pbar.update(1)
                            print(f"\n  Error analyzing {ticker}: {e}")
        else:
            print(f"\nAnalyzing hedge assets in parallel ({self.n_workers} workers)...")
            with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
                futures = {
                    executor.submit(_analyze_hedge_worker, args): args[0] 
                    for args in worker_args
                }
                
                for future in as_completed(futures):
                    ticker = futures[future]
                    try:
                        _, result = future.result()
                        results[ticker] = result
                        print(f"  Completed: {ticker}")
                    except Exception as e:
                        print(f"  Error analyzing {ticker}: {e}")
        
        elapsed = time.time() - start_time
        print(f"Individual hedge analysis completed in {elapsed:.2f}s")
        
        return results
    
    def build_optimal_portfolio(
        self,
        target_cvar_reduction: float = 0.25,
        target_mdd_reduction: float = 0.25,
        method: str = 'greedy'
    ) -> Dict:
        """
        Build optimal multi-asset hedge portfolio.
        
        Args:
            target_cvar_reduction: Target CVaR reduction
            target_mdd_reduction: Target MDD reduction
            method: 'greedy', 'cvar', or 'sharpe'
            
        Returns:
            Dictionary with portfolio composition and performance
        """
        print(f"\nBuilding optimal portfolio (method={method})...")
        
        base_asset = self.config['assets']['base']
        base_returns = self.returns[base_asset]
        
        # Get available hedge assets
        hedge_tickers = [
            ticker 
            for ticker in self.hedge_weights.keys()
            if ticker in self.returns.columns
        ]
        hedge_returns = self.returns[hedge_tickers]
        
        # Get max weights - use multi-asset limits for custom assets
        max_weights = {}
        custom_multi_max = self.config.get('data', {}).get('custom_assets_max_weight_multi', 0.30)
        for ticker in hedge_tickers:
            # Check if this is a custom asset (not in original config hedges)
            is_custom = ticker not in [h['ticker'] for h in self.config['assets']['hedges']]
            if is_custom:
                max_weights[ticker] = custom_multi_max
            else:
                max_weights[ticker] = self.hedge_weights[ticker]
        
        # Optimize
        if method == 'greedy':
            weights = greedy_sequential_allocation(
                base_returns=base_returns,
                hedge_returns=hedge_returns,
                target_reduction=target_cvar_reduction,
                metric='cvar',
                max_total_weight=self.config['optimization']['max_total_hedge_weight'],
                max_weights=max_weights,
                weight_step=self.config['optimization']['weight_step'],
                alpha=self.config['metrics']['cvar_confidence'],
                cvar_frequency=self.config['metrics'].get('cvar_frequency', 'monthly'),
                tie_break_tolerance=self.config['optimization'].get('tie_break_tolerance', 0.001)
            )
        elif method == 'cvar':
            weights = optimize_multi_asset_cvar(
                base_returns=base_returns,
                hedge_returns=hedge_returns,
                target_cvar_reduction=target_cvar_reduction,
                max_total_weight=self.config['optimization']['max_total_hedge_weight'],
                max_weights=max_weights,
                alpha=self.config['metrics']['cvar_confidence'],
                cvar_frequency=self.config['metrics'].get('cvar_frequency', 'monthly')
            )
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Calculate portfolio analytics
        analytics = portfolio_analytics(
            base_returns=base_returns,
            hedge_returns=hedge_returns,
            weights=weights,
            alpha=self.config['metrics']['cvar_confidence'],
            rf_rate=self.mean_rf_rate,
            cvar_frequency=self.config['metrics'].get('cvar_frequency', 'monthly')
        )
        
        # Calculate regime-conditional performance
        normal_periods = self.regime_labels == 0
        crisis_periods = self.regime_labels == 1
        
        # Construct portfolio with rebalancing
        total_weight = sum(weights.values())
        
        # Build full weights dict including base asset
        full_weights = {base_asset: 1 - total_weight}
        full_weights.update(weights)
        
        # Get rebalancing frequency from config
        rebalance_freq = self.config.get('rebalancing', {}).get('frequency', 'quarterly')
        
        # ============================================================================
        # PORTFOLIO DATE RANGE LOGIC (Critical for accuracy)
        # ============================================================================
        # 1. Select only assets with non-zero weights (> 0.1%)
        # 2. Determine the latest inception date among selected assets
        # 3. Filter returns data to start from that date (longest valid window)
        # 4. This ensures we only use dates where ALL portfolio assets have real data
        # ============================================================================
        
        # Step 1: Identify assets actually in this portfolio (non-zero weights only)
        portfolio_assets = [
            asset for asset, weight in full_weights.items() 
            if weight > 0.001 and asset in self.returns.columns
        ]
        
        # Step 2: Determine latest inception date from asset inception dates
        # This comes from the downloader which tracks actual start dates
        # (handles zero-filled monthly data like MAN AHL correctly)
        latest_inception = None
        if hasattr(self, 'downloader') and hasattr(self.downloader, 'asset_inception_dates'):
            inception_dates = self.downloader.asset_inception_dates
            for asset in portfolio_assets:
                if asset in inception_dates:
                    asset_start = inception_dates[asset]
                    if latest_inception is None or asset_start > latest_inception:
                        latest_inception = asset_start
        
        # Step 3: Filter returns to the valid date range
        portfolio_returns_data = self.returns[portfolio_assets].copy()
        if latest_inception is not None:
            portfolio_returns_data = portfolio_returns_data.loc[latest_inception:]
        
        # Validate: Ensure we have data after filtering
        if len(portfolio_returns_data) < 252:  # Minimum 1 year of data
            raise ValueError(f"Insufficient data for portfolio simulation: only {len(portfolio_returns_data)} days available")
        
        # Step 4: Simulate rebalanced portfolio
        # Filter weights to only include assets in the filtered returns data
        portfolio_weights = {asset: weight for asset, weight in full_weights.items() 
                             if asset in portfolio_returns_data.columns}
        
        rebalanced_sim = simulate_rebalanced_portfolio(
            returns=portfolio_returns_data,
            weights=portfolio_weights,
            rebalance_frequency=rebalance_freq
        )
        
        # Use rebalanced portfolio returns for performance metrics
        portfolio_returns = rebalanced_sim['portfolio_return']
        portfolio_values = rebalanced_sim['portfolio_value']
        
        # Align base_returns to the same date range as portfolio_returns (after dropna)
        # This ensures date ranges match in metadata and regime analysis
        base_returns = base_returns.loc[portfolio_returns.index]
        
        # Add rebalancing summary to analytics
        rebal_summary = get_rebalancing_summary(rebalanced_sim)
        analytics['rebalancing'] = {
            'frequency': rebalance_freq,
            'n_rebalances': rebal_summary['n_rebalances'],
            'avg_drift': rebal_summary['avg_drift_at_rebalance'],
            'max_drift': rebal_summary['max_drift_at_rebalance']
        }
        
        # Store portfolio values for drawdown chart
        analytics['portfolio_values'] = portfolio_values
        analytics['portfolio_returns'] = portfolio_returns
        
        # ============================================================================
        # PORTFOLIO METADATA (for reporting)
        # ============================================================================
        # The portfolio_start_date comes from the actual simulated returns index[0]
        # This should match the latest_inception we calculated earlier
        # ============================================================================
        try:
            # Get actual date range from the simulated portfolio
            portfolio_start = portfolio_returns.index[0]
            portfolio_end = portfolio_returns.index[-1]
            
            # Validation: Ensure the analysis period starts at the latest inception
            # (within 1 day tolerance for edge cases with non-trading days)
            if latest_inception is not None:
                days_diff = abs((portfolio_start - latest_inception).days)
                if days_diff > 5:  # Allow small tolerance for weekends/holidays
                    print(f"  Warning: Portfolio start ({portfolio_start.date()}) differs from "
                          f"latest inception ({latest_inception.date()}) by {days_diff} days")
            
            # Get asset inception dates ONLY for assets actually in this portfolio
            # Use downloader's pre-computed inception dates (accurate for zero-filled custom data)
            asset_inceptions = {}
            for asset, weight in full_weights.items():
                if weight > 0.001 and asset in self.returns.columns:
                    # Prefer pre-computed inception dates from downloader
                    if hasattr(self, 'downloader') and asset in self.downloader.asset_inception_dates:
                        asset_inceptions[asset] = self.downloader.asset_inception_dates[asset]
                    else:
                        # Fallback to first_valid_index (for non-filled data)
                        first_valid = self.returns[asset].first_valid_index()
                        if first_valid is not None:
                            asset_inceptions[asset] = first_valid
            
            analytics['portfolio_metadata'] = {
                'portfolio_start_date': portfolio_start,
                'portfolio_end_date': portfolio_end,
                'asset_inception_dates': asset_inceptions,
                'earliest_all_data_date': self.returns.index[0],
                'latest_all_data_date': self.returns.index[-1]
            }
        except Exception as e:
            analytics['portfolio_metadata'] = {}
        
        # Performance by regime
        if crisis_periods.sum() > 0:
            crisis_base_ret = base_returns[crisis_periods].mean() * 252
            crisis_portfolio_ret = portfolio_returns[crisis_periods].mean() * 252
        else:
            crisis_base_ret = np.nan
            crisis_portfolio_ret = np.nan
        
        if normal_periods.sum() > 0:
            normal_base_ret = base_returns[normal_periods].mean() * 252
            normal_portfolio_ret = portfolio_returns[normal_periods].mean() * 252
        else:
            normal_base_ret = np.nan
            normal_portfolio_ret = np.nan
        
        analytics['crisis_base_return'] = crisis_base_ret
        analytics['crisis_portfolio_return'] = crisis_portfolio_ret
        analytics['normal_base_return'] = normal_base_ret
        analytics['normal_portfolio_return'] = normal_portfolio_ret
        
        print(f"Total hedge weight: {total_weight:.1%}")
        print(f"CVaR reduction: {analytics['cvar_reduction_pct']:.1f}%")
        print(f"MDD reduction: {analytics['mdd_reduction_pct']:.1f}%")
        
        for ticker, w in weights.items():
            if w > 0:
                print(f"  {ticker}: {w:.1%}")
        
        return analytics
    
    def run_full_backtest(self) -> Dict:
        """
        Run complete backtest pipeline with parallel processing.
        
        Returns:
            Dictionary with all results
        """
        total_start = time.time()
        
        print("\n" + "=" * 60)
        print("TAIL-RISK HEDGE BACKTESTING ENGINE")
        print("=" * 60)
        
        # Overall pipeline steps
        pipeline_steps = [
            "Loading data",
            "Detecting regimes", 
            "Analyzing individual hedges",
            "Building optimal portfolios"
        ]
        
        if TQDM_AVAILABLE:
            # Create overall pipeline progress bar
            pipeline_pbar = tqdm(total=len(pipeline_steps), desc="Pipeline Progress",
                                bar_format='{l_bar}{bar:40}{r_bar}',
                                colour='yellow', position=0, leave=True)
            
            # Step 1: Load data
            pipeline_pbar.set_postfix_str("Loading data...")
            self.load_data(use_cache=True)
            pipeline_pbar.update(1)
            
            # Step 2: Identify regimes
            pipeline_pbar.set_postfix_str("Detecting regimes...")
            self.identify_regimes()
            pipeline_pbar.update(1)
            
            # Step 3: Analyze individual hedges (has its own progress bar)
            pipeline_pbar.set_postfix_str("Analyzing hedges...")
            individual_results = self.analyze_all_hedges()
            pipeline_pbar.update(1)
            
            # Step 4: Build optimal portfolios (has its own progress bar)
            pipeline_pbar.set_postfix_str("Building portfolios...")
            portfolio_targets = [0.10, 0.25, 0.50]
            portfolios = self._build_portfolios_parallel(portfolio_targets)
            pipeline_pbar.update(1)
            
            pipeline_pbar.set_postfix_str("Complete!")
            pipeline_pbar.close()
        else:
            # Step 1: Load data
            self.load_data(use_cache=True)
            
            # Step 2: Identify regimes
            self.identify_regimes()
            
            # Step 3: Analyze individual hedges (parallelized)
            individual_results = self.analyze_all_hedges()
            
            # Step 4: Build optimal portfolios (parallelized)
            print("\n" + "=" * 60)
            print("BUILDING OPTIMAL PORTFOLIOS")
            print("=" * 60)
            
            portfolio_targets = [0.10, 0.25, 0.50]
            portfolios = self._build_portfolios_parallel(portfolio_targets)
        
        total_elapsed = time.time() - total_start
        print(f"\n{'=' * 60}")
        print(f"Total backtest time: {total_elapsed:.2f}s")
        print(f"{'=' * 60}")
        
        # Calculate baseline (100% ACWI) cumulative values for drawdown comparison
        base_asset = self.config['assets']['base']
        baseline_returns = self.returns[base_asset].dropna()
        baseline_values = (1 + baseline_returns).cumprod()
        
        results = {
            'config': self.config,
            'hedge_names': self.hedge_names,
            'data_info': {
                'start_date': str(self.prices.index[0]),
                'end_date': str(self.prices.index[-1]),
                'n_days': len(self.prices),
                'assets': list(self.prices.columns),
                'risk_free_rate': self.mean_rf_rate
            },
            'regime_stats': self.regime_detector.get_regime_statistics(self.regime_labels),
            'individual_hedges': individual_results,
            'portfolios': portfolios,
            'returns': self.returns,
            'regime_labels': self.regime_labels,
            'baseline_values': baseline_values
        }
        
        return results
    
    def _build_portfolios_parallel(self, targets: List[float]) -> Dict:
        """
        Build optimal portfolios for multiple targets in parallel with progress bar.
        
        Args:
            targets: List of CVaR reduction targets
            
        Returns:
            Dictionary of portfolios keyed by target name
        """
        start_time = time.time()
        
        def build_single(target):
            return self.build_optimal_portfolio(
                target_cvar_reduction=target,
                method='greedy'
            )
        
        portfolios = {}
        n_targets = len(targets)
        
        if TQDM_AVAILABLE:
            # Progress bar with preparation phase
            with tqdm(total=n_targets + 1, desc="Multi-Asset Portfolios",
                     bar_format='{l_bar}{bar:30}{r_bar}{bar:-10b}',
                     colour='blue') as pbar:
                
                pbar.set_postfix_str("Preparing optimization...")
                pbar.update(1)
                
                with ThreadPoolExecutor(max_workers=min(n_targets, self.n_workers)) as executor:
                    futures = {
                        executor.submit(build_single, t): t 
                        for t in targets
                    }
                    
                    for future in as_completed(futures):
                        target = futures[future]
                        try:
                            result = future.result()
                            key = f"{int(target * 100)}pct_reduction"
                            portfolios[key] = result
                            pbar.set_postfix_str(f"✓ {int(target*100)}% target")
                            pbar.update(1)
                        except Exception as e:
                            pbar.set_postfix_str(f"✗ {int(target*100)}% target")
                            pbar.update(1)
                            print(f"\n  Error building {target*100:.0f}% portfolio: {e}")
        else:
            print(f"Building {n_targets} portfolios in parallel...")
            with ThreadPoolExecutor(max_workers=min(n_targets, self.n_workers)) as executor:
                futures = {
                    executor.submit(build_single, t): t 
                    for t in targets
                }
                
                for future in as_completed(futures):
                    target = futures[future]
                    try:
                        result = future.result()
                        key = f"{int(target * 100)}pct_reduction"
                        portfolios[key] = result
                    except Exception as e:
                        print(f"  Error building {target*100:.0f}% portfolio: {e}")
        
        elapsed = time.time() - start_time
        print(f"Portfolio optimization completed in {elapsed:.2f}s")
        
        return portfolios


def quick_backtest(config_path: str = 'config.yaml') -> Dict:
    """
    Convenience function to run complete backtest.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Dictionary with all results
    """
    backtester = Backtester(config_path)
    return backtester.run_full_backtest()
