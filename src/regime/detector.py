"""
Regime Detection Module

Author: L.Bassetti
Implements multiple methods to identify crisis vs normal market regimes.
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, List, Dict
from scipy import stats


class RegimeDetector:
    """
    Multi-method regime detector for crisis identification.
    
    Supports:
    - Drawdown-based regime
    - VIX threshold regime
    - Volatility percentile regime
    - Markov-switching (simplified)
    - Ensemble voting
    """
    
    def __init__(self, config: dict):
        """
        Initialize detector with configuration.
        
        Args:
            config: Dictionary with regime detection parameters
        """
        self.config = config
        self.method = config.get('method', 'ensemble')
        self.drawdown_threshold = config.get('drawdown_threshold', -0.10)
        self.vix_crisis_threshold = config.get('vix_crisis_threshold', 30)
        self.vix_recovery_threshold = config.get('vix_recovery_threshold', 20)
        self.volatility_window = config.get('volatility_window', 21)
        self.volatility_percentile = config.get('volatility_percentile', 0.75)
        
    def detect(
        self,
        equity_prices: pd.Series,
        equity_returns: pd.Series,
        vix: Optional[pd.Series] = None
    ) -> pd.Series:
        """
        Detect regime using configured method.
        
        Args:
            equity_prices: Price series for base asset
            equity_returns: Return series for base asset
            vix: Optional VIX series
            
        Returns:
            Binary series (0=Normal, 1=Crisis)
        """
        if self.method == 'drawdown':
            return self.drawdown_regime(equity_prices)
        elif self.method == 'vix':
            if vix is None:
                raise ValueError("VIX data required for vix method")
            return self.vix_regime(vix)
        elif self.method == 'volatility':
            return self.volatility_percentile_regime(equity_returns)
        elif self.method == 'markov':
            return self.markov_regime(equity_returns)
        elif self.method == 'ensemble':
            return self.ensemble_regime(equity_prices, equity_returns, vix)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def drawdown_regime(self, prices: pd.Series) -> pd.Series:
        """
        Identify crisis as periods when drawdown exceeds threshold.
        
        Args:
            prices: Price series
            
        Returns:
            Binary regime (0=Normal, 1=Crisis)
        """
        # Calculate drawdown
        cum_max = prices.expanding().max()
        drawdown = (prices - cum_max) / cum_max
        
        # Crisis when drawdown below threshold
        regime = (drawdown < self.drawdown_threshold).astype(int)
        
        return regime
    
    def vix_regime(self, vix: pd.Series) -> pd.Series:
        """
        Identify crisis using VIX threshold with hysteresis.
        
        Uses two thresholds to avoid rapid switching:
        - Enter crisis when VIX > crisis_threshold
        - Exit crisis when VIX < recovery_threshold
        
        Optimized using numpy arrays for faster iteration.
        
        Args:
            vix: VIX series
            
        Returns:
            Binary regime (0=Normal, 1=Crisis)
        """
        import numpy as np
        
        # Convert to numpy for faster iteration
        vix_values = vix.values
        n = len(vix_values)
        regime_arr = np.zeros(n, dtype=np.int32)
        
        crisis_threshold = self.vix_crisis_threshold
        recovery_threshold = self.vix_recovery_threshold
        current_state = 0  # Start in normal
        
        for i in range(n):
            vix_value = vix_values[i]
            
            if np.isnan(vix_value):
                regime_arr[i] = current_state
                continue
            
            if current_state == 0:  # Currently normal
                if vix_value > crisis_threshold:
                    current_state = 1  # Enter crisis
            else:  # Currently in crisis
                if vix_value < recovery_threshold:
                    current_state = 0  # Exit crisis
            
            regime_arr[i] = current_state
        
        return pd.Series(regime_arr, index=vix.index)
    
    def volatility_percentile_regime(
        self,
        returns: pd.Series,
        window: Optional[int] = None,
        percentile: Optional[float] = None
    ) -> pd.Series:
        """
        Identify crisis when realized volatility exceeds historical percentile.
        
        Args:
            returns: Return series
            window: Rolling window for volatility (default from config)
            percentile: Percentile threshold (default from config)
            
        Returns:
            Binary regime (0=Normal, 1=Crisis)
        """
        if window is None:
            window = self.volatility_window
        if percentile is None:
            percentile = self.volatility_percentile
        
        # Ensure percentile is in [0, 1] range
        if percentile > 1:
            percentile = percentile / 100.0
        
        # Calculate rolling volatility
        vol = returns.rolling(window).std()
        
        # Calculate expanding percentile threshold
        threshold = vol.expanding().quantile(percentile)
        
        # Crisis when volatility exceeds threshold
        regime = (vol > threshold).astype(int)
        
        # Fill initial NaN with 0 (normal)
        regime = regime.fillna(0).astype(int)
        
        return regime
    
    def markov_regime(
        self,
        returns: pd.Series,
        k_regimes: int = 2
    ) -> pd.Series:
        """
        Simplified Markov-switching regime detection.
        
        Uses rolling mean and volatility to classify regimes:
        - Low vol + positive mean = Normal
        - High vol + negative mean = Crisis
        
        (This is a simplified version. Full Hamilton model requires statsmodels.)
        
        Args:
            returns: Return series
            k_regimes: Number of regimes (default 2)
            
        Returns:
            Binary regime (0=Normal, 1=Crisis)
        """
        # Calculate rolling statistics
        window = 63  # ~3 months
        roll_mean = returns.rolling(window).mean()
        roll_std = returns.rolling(window).std()
        
        # Standardize
        z_mean = (roll_mean - roll_mean.mean()) / roll_mean.std()
        z_std = (roll_std - roll_std.mean()) / roll_std.std()
        
        # Crisis = negative returns + high volatility
        # Score: higher = more crisis-like
        crisis_score = z_std - z_mean  # High vol + low returns = high score
        
        # Use median as threshold
        threshold = crisis_score.median()
        regime = (crisis_score > threshold).astype(int)
        
        # Fill NaN with 0
        regime = regime.fillna(0).astype(int)
        
        return regime
    
    def ensemble_regime(
        self,
        prices: pd.Series,
        returns: pd.Series,
        vix: Optional[pd.Series] = None
    ) -> pd.Series:
        """
        Ensemble method combining multiple signals with majority voting.
        
        Uses parallel execution to run all regime detection methods concurrently,
        significantly reducing total computation time.
        
        Args:
            prices: Price series
            returns: Return series
            vix: Optional VIX series
            
        Returns:
            Binary regime (0=Normal, 1=Crisis)
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        # Define worker functions for parallel execution
        def run_drawdown():
            return ('drawdown', self.drawdown_regime(prices))
        
        def run_volatility():
            return ('volatility', self.volatility_percentile_regime(returns))
        
        def run_markov():
            return ('markov', self.markov_regime(returns))
        
        def run_vix():
            if vix is not None:
                vix_signal = self.vix_regime(vix)
                # Align with returns index
                vix_signal = vix_signal.reindex(returns.index, fill_value=0)
                return ('vix', vix_signal)
            return None
        
        # Run all regime methods in parallel
        signals = []
        workers = [run_drawdown, run_volatility, run_markov, run_vix]
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(worker) for worker in workers]
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    signals.append(result[1])
        
        # Combine into DataFrame
        signal_df = pd.concat(signals, axis=1)
        signal_df.columns = [f'signal_{i}' for i in range(len(signals))]
        
        # Majority vote (>50% of signals say crisis)
        vote_sum = signal_df.sum(axis=1)
        threshold = len(signals) / 2
        regime = (vote_sum > threshold).astype(int)
        
        return regime
    
    def get_regime_statistics(self, regime: pd.Series) -> dict:
        """
        Calculate statistics about regime periods.
        
        Args:
            regime: Binary regime series
            
        Returns:
            Dictionary with regime statistics
        """
        crisis_periods = regime == 1
        normal_periods = regime == 0
        
        # Count transitions
        transitions = (regime.diff() != 0).sum()
        
        # Count crisis episodes
        crisis_starts = (regime.diff() == 1).sum()
        
        # Average duration
        if crisis_starts > 0:
            crisis_lengths = []
            in_crisis = False
            current_length = 0
            
            for value in regime:
                if value == 1:
                    if not in_crisis:
                        in_crisis = True
                        current_length = 1
                    else:
                        current_length += 1
                else:
                    if in_crisis:
                        crisis_lengths.append(current_length)
                        in_crisis = False
                        current_length = 0
            
            # If still in crisis at end
            if in_crisis:
                crisis_lengths.append(current_length)
            
            avg_crisis_length = np.mean(crisis_lengths) if crisis_lengths else 0
        else:
            avg_crisis_length = 0
        
        stats = {
            'total_periods': len(regime),
            'crisis_periods': crisis_periods.sum(),
            'normal_periods': normal_periods.sum(),
            'crisis_pct': crisis_periods.sum() / len(regime) * 100,
            'transitions': transitions,
            'crisis_episodes': crisis_starts,
            'avg_crisis_length': avg_crisis_length
        }
        
        return stats


def identify_crisis_dates(
    regime: pd.Series,
    min_duration: int = 5
) -> list:
    """
    Extract start and end dates of crisis episodes.
    
    Args:
        regime: Binary regime series
        min_duration: Minimum crisis duration to include (days)
        
    Returns:
        List of tuples (start_date, end_date, duration)
    """
    episodes = []
    in_crisis = False
    start_date = None
    
    for date, value in regime.items():
        if value == 1 and not in_crisis:
            # Crisis starts
            in_crisis = True
            start_date = date
        elif value == 0 and in_crisis:
            # Crisis ends
            end_date = date
            duration = len(regime[start_date:end_date])
            
            if duration >= min_duration:
                episodes.append((start_date, end_date, duration))
            
            in_crisis = False
            start_date = None
    
    # Handle case where crisis continues to end of data
    if in_crisis and start_date is not None:
        end_date = regime.index[-1]
        duration = len(regime[start_date:end_date])
        if duration >= min_duration:
            episodes.append((start_date, end_date, duration))
    
    return episodes


def align_regime_with_returns(
    regime: pd.Series,
    returns: pd.Series
) -> pd.Series:
    """
    Align regime labels with return series index.
    
    Args:
        regime: Regime series (possibly from prices)
        returns: Return series
        
    Returns:
        Regime series aligned to returns index
    """
    # Reindex regime to match returns
    aligned = regime.reindex(returns.index, method='ffill')
    
    # Fill any remaining NaN with 0 (normal)
    aligned = aligned.fillna(0).astype(int)
    
    return aligned


# Named historical crisis periods (approximate peak-to-trough dates)
NAMED_CRISES = {
    'Global Financial Crisis': ('2008-09-15', '2009-03-09'),
    'European Debt Crisis': ('2011-07-01', '2011-10-04'),
    'China/Oil Shock': ('2015-08-18', '2016-02-11'),
    'COVID-19 Crash': ('2020-02-19', '2020-03-23'),
    '2022 Bear Market': ('2022-01-03', '2022-10-12'),
}


def get_named_crisis_periods(
    regime_labels: pd.Series,
    base_returns: Optional[pd.Series] = None,
    min_duration: int = 10
) -> List[Dict]:
    """
    Identify and name major crisis episodes from regime labels.
    
    Matches detected crisis periods against known historical crises.
    Unnamed periods are labeled as "Crisis N".
    
    Args:
        regime_labels: Binary regime series (0=Normal, 1=Crisis)
        base_returns: Optional returns series for severity calculation
        min_duration: Minimum days for a crisis to be included
        
    Returns:
        List of crisis dictionaries with keys:
        - name: Crisis name (e.g., 'Global Financial Crisis' or 'Crisis 1')
        - start: Start date (Timestamp)
        - end: End date (Timestamp)
        - duration_days: Number of trading days
        - severity: Cumulative return during crisis (if base_returns provided)
    """
    # Get crisis episodes from regime
    episodes = identify_crisis_dates(regime_labels, min_duration=min_duration)
    
    if not episodes:
        return []
    
    result = []
    unnamed_counter = 1
    
    for start, end, duration in episodes:
        # Try to match with named crises
        crisis_name = None
        
        for name, (known_start, known_end) in NAMED_CRISES.items():
            known_start_ts = pd.Timestamp(known_start)
            known_end_ts = pd.Timestamp(known_end)
            
            # Check if this episode overlaps with a known crisis
            # Allow some flexibility (within 30 days of known dates)
            if (start <= known_end_ts + pd.Timedelta(days=30) and
                end >= known_start_ts - pd.Timedelta(days=30)):
                crisis_name = name
                break
        
        if crisis_name is None:
            crisis_name = f"Crisis {unnamed_counter}"
            unnamed_counter += 1
        
        # Calculate severity if returns provided
        severity = None
        if base_returns is not None:
            crisis_returns = base_returns.loc[start:end]
            if len(crisis_returns) > 0:
                severity = (1 + crisis_returns).prod() - 1  # Cumulative return
        
        result.append({
            'name': crisis_name,
            'start': start,
            'end': end,
            'duration_days': duration,
            'severity': severity
        })
    
    # Sort by start date
    result.sort(key=lambda x: x['start'])
    
    return result
