"""
Reporting Module - Investment Banking Research Style

Author: L.Bassetti
Generates professional HTML reports for tail-risk hedge analysis.
Includes embedded charts for rolling correlations.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import base64
from io import BytesIO

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

# Import matplotlib for chart generation
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def calculate_drawdown_series(prices: pd.Series) -> pd.Series:
    """
    Convert price/value series to drawdown series.
    
    Args:
        prices: Cumulative price/value series (starting around 1.0)
        
    Returns:
        Drawdown series (-1 to 0 range, where -1 = -100%)
    """
    if prices is None or len(prices) == 0:
        return pd.Series(dtype='float64')
    
    # Handle both Series and array input
    if isinstance(prices, np.ndarray):
        prices = pd.Series(prices)
    
    rolling_max = prices.expanding().max()
    drawdown = (prices - rolling_max) / rolling_max
    return drawdown


def format_pct(value: float, decimals: int = 2, show_sign: bool = False) -> str:
    """Format as percentage."""
    if pd.isna(value):
        return "—"
    sign = "+" if show_sign and value > 0 else ""
    return f"{sign}{value:.{decimals}f}%"


def format_num(value: float, decimals: int = 2) -> str:
    """Format number."""
    if pd.isna(value):
        return "—"
    return f"{value:.{decimals}f}"


def format_bps(value: float) -> str:
    """Format as basis points."""
    if pd.isna(value):
        return "—"
    return f"{value * 10000:.0f} bps"


def get_asset_name(ticker: str, hedge_names: Dict[str, str]) -> str:
    """Get display name for asset."""
    name_map = {
        'ACWI': 'MSCI All Country World Index',
        '^VIX': 'VIX Volatility Index',
    }
    if ticker in hedge_names:
        return hedge_names[ticker]
    return name_map.get(ticker, ticker)


def create_rolling_correlation_chart(
    base_returns: pd.Series,
    hedge_returns: pd.Series,
    regime_labels: pd.Series,
    asset_name: str,
    window: int = 63
) -> str:
    """
    Create a rolling correlation chart with crisis periods highlighted.
    
    Args:
        base_returns: Base asset returns (ACWI)
        hedge_returns: Hedge asset returns
        regime_labels: Binary series (1=crisis, 0=normal)
        asset_name: Display name for the hedge asset
        window: Rolling window in days (default 63 = ~3 months)
        
    Returns:
        Base64 encoded PNG image string for embedding in HTML
    """
    # Align data
    aligned = pd.DataFrame({
        'base': base_returns,
        'hedge': hedge_returns,
        'regime': regime_labels
    }).dropna()
    
    if len(aligned) < window:
        return ""
    
    # Calculate rolling correlation
    rolling_corr = aligned['base'].rolling(window=window).corr(aligned['hedge'])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 3.5), dpi=100)
    
    # Set style
    ax.set_facecolor('#fafafa')
    fig.patch.set_facecolor('white')
    
    # Highlight crisis periods with pale red background
    crisis_periods = aligned['regime'] == 1
    if crisis_periods.any():
        # Find contiguous crisis periods
        crisis_starts = []
        crisis_ends = []
        in_crisis = False
        
        for i, (date, is_crisis) in enumerate(crisis_periods.items()):
            if is_crisis and not in_crisis:
                crisis_starts.append(date)
                in_crisis = True
            elif not is_crisis and in_crisis:
                crisis_ends.append(date)
                in_crisis = False
        
        # Handle case where crisis extends to end
        if in_crisis:
            crisis_ends.append(crisis_periods.index[-1])
        
        # Draw crisis backgrounds
        for start, end in zip(crisis_starts, crisis_ends):
            ax.axvspan(start, end, alpha=0.3, color='#ffcccc', zorder=0)
    
    # Plot rolling correlation
    ax.plot(rolling_corr.index, rolling_corr.values, color='#1a1a2e', linewidth=1.2, zorder=2)
    
    # Add zero line
    ax.axhline(y=0, color='#888888', linestyle='--', linewidth=0.8, alpha=0.7)
    
    # Add average correlation line
    avg_corr = rolling_corr.mean()
    ax.axhline(y=avg_corr, color='#0f3460', linestyle=':', linewidth=1, alpha=0.8)
    ax.text(rolling_corr.index[-1], avg_corr, f' Avg: {avg_corr:.2f}', 
            va='center', fontsize=8, color='#0f3460')
    
    # Formatting
    ax.set_ylabel('Correlation', fontsize=10)
    ax.set_title(f'{asset_name} — {window}-Day Rolling Correlation vs. ACWI', 
                 fontsize=11, fontweight='bold', color='#1a1a2e', pad=10)
    
    # Y-axis limits
    ax.set_ylim(-1, 1)
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    
    # X-axis formatting
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
    # Grid
    ax.grid(True, axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
    ax.grid(True, axis='x', alpha=0.2, linestyle='-', linewidth=0.5)
    
    # Spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#cccccc')
    ax.spines['bottom'].set_color('#cccccc')
    
    # Legend for crisis periods
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#ffcccc', alpha=0.5, label='Crisis Periods')]
    ax.legend(handles=legend_elements, loc='lower left', fontsize=8, framealpha=0.9)
    
    plt.tight_layout()
    
    # Convert to base64
    buffer = BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight', facecolor='white', edgecolor='none')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close(fig)
    
    return image_base64


def create_crisis_performance_chart(
    crisis_periods: List[Dict],
    returns: pd.DataFrame,
    base_asset: str = 'ACWI'
) -> str:
    """
    Create grouped bar chart showing asset performance during each crisis period.
    
    Args:
        crisis_periods: List of crisis dicts with 'name', 'start', 'end' keys
        returns: DataFrame with asset returns (columns = tickers)
        base_asset: Name of the base asset column
        
    Returns:
        Base64 encoded PNG image string for embedding in HTML
    """
    if not crisis_periods:
        return ""
    
    # Calculate cumulative returns during each crisis for each asset
    crisis_names = []
    crisis_returns = {}
    
    # Get list of assets to plot
    assets = [base_asset] + [col for col in returns.columns if col != base_asset and col != '^VIX']
    
    # Limit to key assets for readability
    key_assets = [base_asset, 'TLT', 'GLD', 'BTC-USD', 'DBMF']
    assets = [a for a in key_assets if a in returns.columns]
    
    for crisis in crisis_periods:
        name = crisis['name']
        start = crisis['start']
        end = crisis['end']
        
        # Add date range to name for clarity
        start_str = start.strftime('%b %Y') if hasattr(start, 'strftime') else str(start)[:7]
        crisis_names.append(f"{name}\n({start_str})")
        
        # Calculate cumulative return for each asset during this crisis
        for asset in assets:
            if asset not in returns.columns:
                continue
            
            asset_returns = returns[asset].loc[start:end].dropna()
            if len(asset_returns) > 0:
                cum_return = (1 + asset_returns).prod() - 1
            else:
                cum_return = 0.0
            
            if asset not in crisis_returns:
                crisis_returns[asset] = []
            crisis_returns[asset].append(cum_return * 100)  # Convert to percentage
    
    if not crisis_returns:
        return ""
    
    # Create figure
    n_crises = len(crisis_names)
    n_assets = len(crisis_returns)
    
    fig, ax = plt.subplots(figsize=(max(10, n_crises * 1.5), 6), dpi=100)
    ax.set_facecolor('#fafafa')
    fig.patch.set_facecolor('white')
    
    # Bar positions
    x = np.arange(n_crises)
    bar_width = 0.8 / n_assets
    
    # Color scheme
    color_map = {
        'ACWI': '#c0392b',      # Red for baseline
        'TLT': '#2980b9',       # Blue for treasuries
        'IEF': '#3498db',
        'GLD': '#f39c12',       # Gold color
        'SLV': '#95a5a6',       # Silver color
        'BTC-USD': '#f7931a',   # Bitcoin orange
        'ETH-USD': '#627eea',   # Ethereum purple
        'DBMF': '#1abc9c',      # Teal for managed futures
    }
    
    # Plot bars for each asset
    for i, (asset, rets) in enumerate(crisis_returns.items()):
        offset = (i - n_assets / 2 + 0.5) * bar_width
        color = color_map.get(asset, '#888888')
        
        bars = ax.bar(x + offset, rets, bar_width * 0.9, 
                     label=asset, color=color, edgecolor='white', linewidth=0.5)
        
        # Add value labels on bars
        for bar, ret in zip(bars, rets):
            height = bar.get_height()
            va = 'bottom' if height >= 0 else 'top'
            offset_y = 0.5 if height >= 0 else -0.5
            ax.annotate(f'{ret:.0f}%',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, offset_y),
                       textcoords="offset points",
                       ha='center', va=va, fontsize=7, color='#333333')
    
    # Zero line
    ax.axhline(y=0, color='#333333', linestyle='-', linewidth=1)
    
    # Formatting
    ax.set_ylabel('Cumulative Return (%)', fontsize=10)
    ax.set_title('Asset Performance During Crisis Periods', 
                 fontsize=12, fontweight='bold', color='#1a1a2e', pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(crisis_names, fontsize=9)
    
    # Grid
    ax.grid(True, axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#cccccc')
    ax.spines['bottom'].set_color('#cccccc')
    
    # Legend
    ax.legend(loc='upper right', fontsize=8, framealpha=0.9, ncol=min(n_assets, 4))
    
    plt.tight_layout()
    
    # Convert to base64
    buffer = BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight', facecolor='white', edgecolor='none')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close(fig)
    
    return image_base64


def get_styles() -> str:
    """Professional IB-style CSS."""
    return """
    <style>
        :root {
            --primary: #1a1a2e;
            --secondary: #16213e;
            --accent: #0f3460;
            --text: #2c3e50;
            --text-light: #5d6d7e;
            --border: #d5d8dc;
            --positive: #1e8449;
            --negative: #c0392b;
            --bg: #ffffff;
            --bg-alt: #f8f9fa;
        }
        
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: 'Times New Roman', Georgia, serif;
            background: var(--bg);
            color: var(--text);
            line-height: 1.5;
            font-size: 11pt;
        }
        
        .container {
            max-width: 900px;
            margin: 0 auto;
            padding: 40px 50px;
        }
        
        /* Header */
        header {
            border-bottom: 3px double var(--border);
            padding-bottom: 20px;
            margin-bottom: 30px;
        }
        
        .report-title {
            font-size: 22pt;
            font-weight: normal;
            color: var(--primary);
            letter-spacing: 1px;
            margin-bottom: 5px;
        }
        
        .report-subtitle {
            font-size: 12pt;
            color: var(--text-light);
            font-style: italic;
        }
        
        .report-date {
            font-size: 10pt;
            color: var(--text-light);
            margin-top: 10px;
        }
        
        /* Sections */
        section {
            margin-bottom: 35px;
            page-break-inside: avoid;
        }
        
        h2 {
            font-size: 14pt;
            font-weight: bold;
            color: var(--primary);
            border-bottom: 1px solid var(--border);
            padding-bottom: 8px;
            margin-bottom: 15px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        h3 {
            font-size: 12pt;
            font-weight: bold;
            color: var(--secondary);
            margin: 20px 0 10px 0;
        }
        
        h4 {
            font-size: 11pt;
            font-weight: bold;
            color: var(--text);
            margin: 15px 0 8px 0;
        }
        
        /* Tables */
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 12px 0 20px 0;
            font-size: 10pt;
        }
        
        th {
            background: var(--bg-alt);
            border-top: 2px solid var(--primary);
            border-bottom: 1px solid var(--border);
            padding: 10px 12px;
            text-align: left;
            font-weight: bold;
            color: var(--primary);
        }
        
        td {
            padding: 8px 12px;
            border-bottom: 1px solid #eee;
            vertical-align: top;
        }
        
        tr:last-child td {
            border-bottom: 2px solid var(--primary);
        }
        
        .text-right { text-align: right; }
        .text-center { text-align: center; }
        
        .positive { color: var(--positive); }
        .negative { color: var(--negative); }
        
        /* Summary boxes */
        .summary-box {
            background: var(--bg-alt);
            border: 1px solid var(--border);
            padding: 15px 20px;
            margin: 15px 0;
        }
        
        .summary-row {
            display: flex;
            justify-content: space-between;
            padding: 5px 0;
            border-bottom: 1px dotted var(--border);
        }
        
        .summary-row:last-child {
            border-bottom: none;
        }
        
        .summary-label {
            color: var(--text-light);
        }
        
        .summary-value {
            font-weight: bold;
        }
        
        /* Key findings */
        .key-finding {
            background: linear-gradient(to right, var(--bg-alt), var(--bg));
            border-left: 3px solid var(--accent);
            padding: 12px 15px;
            margin: 15px 0;
        }
        
        /* Footnotes */
        .footnote {
            font-size: 9pt;
            color: var(--text-light);
            font-style: italic;
            margin-top: 8px;
        }
        
        /* Info box for date ranges */
        .info-box {
            background-color: #f0f8ff;
            border-left: 4px solid #2c5aa0;
            padding: 12px 16px;
            margin: 15px 0;
            border-radius: 4px;
            font-size: 10pt;
        }
        
        /* Asset cards */
        .asset-section {
            border: 1px solid var(--border);
            margin: 20px 0;
            padding: 20px;
            background: var(--bg);
        }
        
        .asset-header {
            font-size: 13pt;
            font-weight: bold;
            color: var(--primary);
            border-bottom: 1px solid var(--border);
            padding-bottom: 10px;
            margin-bottom: 15px;
        }
        
        /* Comparison tables */
        .comparison-table th:not(:first-child),
        .comparison-table td:not(:first-child) {
            text-align: right;
        }
        
        .highlight-row {
            background: #fef9e7;
        }
        
        /* Footer */
        footer {
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid var(--border);
            font-size: 9pt;
            color: var(--text-light);
        }
        
        .disclaimer {
            font-style: italic;
            margin-top: 10px;
        }
        
        @media print {
            body { font-size: 10pt; }
            .container { padding: 20px; }
            section { page-break-inside: avoid; }
        }
    </style>
    """


def generate_html_report(
    results: Dict,
    output_path: str = 'output/analysis_report.html',
    returns: Optional[pd.DataFrame] = None,
    regime_labels: Optional[pd.Series] = None
) -> None:
    """
    Generate professional HTML report with embedded charts.
    
    Args:
        results: Dictionary with backtest results
        output_path: Path to save report
        returns: DataFrame of asset returns (for generating charts)
        regime_labels: Binary series of regime labels (for chart highlighting)
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    data_info = results.get('data_info', {})
    regime_stats = results.get('regime_stats', {})
    config = results.get('config', {})
    hedge_names = results.get('hedge_names', {})
    individual_hedges = results.get('individual_hedges', {})
    portfolios = results.get('portfolios', {})
    
    base_ticker = config.get('assets', {}).get('base', 'ACWI')
    
    # Pre-generate charts with progress bar
    charts = {}
    if returns is not None and regime_labels is not None:
        tickers_for_charts = [t for t in individual_hedges.keys() 
                             if t in returns.columns and base_ticker in returns.columns]
        
        if tickers_for_charts:
            if TQDM_AVAILABLE:
                print(f"\nGenerating {len(tickers_for_charts)} correlation charts...")
                iterator = tqdm(tickers_for_charts, desc="Charts", 
                               bar_format='{l_bar}{bar:30}{r_bar}{bar:-10b}',
                               colour='cyan')
            else:
                print(f"\nGenerating {len(tickers_for_charts)} charts...")
                iterator = tickers_for_charts
            
            for ticker in iterator:
                asset_name = get_asset_name(ticker, hedge_names)
                chart_base64 = create_rolling_correlation_chart(
                    base_returns=returns[base_ticker],
                    hedge_returns=returns[ticker],
                    regime_labels=regime_labels,
                    asset_name=asset_name,
                    window=63
                )
                charts[ticker] = chart_base64
                if TQDM_AVAILABLE:
                    iterator.set_postfix_str(f"✓ {ticker}")
    
    print("Building HTML report...")
    
    regime_method = config.get('regime', {}).get('method', 'ensemble')
    cvar_conf = config.get('metrics', {}).get('cvar_confidence', 0.95)
    base_ticker = config.get('assets', {}).get('base', 'ACWI')
    base_name = get_asset_name(base_ticker, hedge_names)
    
    report_date = datetime.now().strftime('%B %d, %Y')
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Tail-Risk Hedge Analysis</title>
    {get_styles()}
</head>
<body>
    <div class="container">
        <header>
            <div class="report-title">Tail-Risk Hedge Analysis</div>
            <div class="report-subtitle">Optimal Portfolio Protection Strategies</div>
            <div class="report-date">{report_date}</div>
        </header>
        
        <!-- Executive Summary -->
        <section>
            <h2>Executive Summary</h2>
            <div class="summary-box">
                <div class="summary-row">
                    <span class="summary-label">Analysis Period</span>
                    <span class="summary-value">{data_info.get('start_date', 'N/A')[:10]} to {data_info.get('end_date', 'N/A')[:10]}</span>
                </div>
                <div class="summary-row">
                    <span class="summary-label">Total Trading Days</span>
                    <span class="summary-value">{data_info.get('n_days', 0):,}</span>
                </div>
                <div class="summary-row">
                    <span class="summary-label">Base Portfolio</span>
                    <span class="summary-value">{base_name} ({base_ticker})</span>
                </div>
                <div class="summary-row">
                    <span class="summary-label">Hedge Assets Analyzed</span>
                    <span class="summary-value">{len(individual_hedges)}</span>
                </div>
                <div class="summary-row">
                    <span class="summary-label">Crisis Detection Method</span>
                    <span class="summary-value">{regime_method.title()}</span>
                </div>
            </div>
        </section>
        
        <!-- Market Regime Analysis -->
        <section>
            <h2>Market Regime Analysis</h2>
            <p>Market conditions were classified using a {regime_method} detection methodology to identify periods of elevated tail risk.</p>
            
            <table>
                <tr>
                    <th>Metric</th>
                    <th class="text-right">Value</th>
                </tr>
                <tr>
                    <td>Days in Crisis Regime</td>
                    <td class="text-right">{regime_stats.get('crisis_periods', 0):,}</td>
                </tr>
                <tr>
                    <td>Percentage of Sample in Crisis</td>
                    <td class="text-right">{format_pct(regime_stats.get('crisis_pct', 0))}</td>
                </tr>
                <tr>
                    <td>Number of Distinct Crisis Episodes</td>
                    <td class="text-right">{regime_stats.get('crisis_episodes', 0)}</td>
                </tr>
                <tr>
                    <td>Average Crisis Duration</td>
                    <td class="text-right">{regime_stats.get('avg_crisis_length', 0):.1f} days</td>
                </tr>
            </table>
            <p class="footnote">Crisis periods are identified using {regime_method} methodology with parameters specified in configuration.</p>
        </section>
        
        <!-- Individual Hedge Analysis -->
        <section>
            <h2>Individual Hedge Asset Analysis</h2>
            <p>Each hedge asset was tested individually to determine the allocation weight required to achieve 10%, 25%, and 50% risk reduction targets. The constraint limits hedge allocation to a maximum of 50% (maintaining at least 50% in {base_ticker}).</p>
            
            <table class="comparison-table">
                <tr>
                    <th>Asset</th>
                    <th>Crisis Corr.</th>
                    <th>Max Weight</th>
                    <th>Max CVaR Reduction</th>
                    <th>Max MDD Reduction</th>
                </tr>
"""
    
    # Summary table of all hedges
    for ticker, hedge_data in individual_hedges.items():
        asset_name = get_asset_name(ticker, hedge_names)
        corr = hedge_data.get('correlations', {})
        crisis_corr = corr.get('correlation_crisis', corr.get('pearson_full', 0))
        
        # Get max weight from config
        max_weight = 0.50
        for h in config.get('assets', {}).get('hedges', []):
            if h.get('ticker') == ticker:
                max_weight = h.get('max_weight', 0.50)
                break
        
        # Find max achieved reductions
        max_cvar_reduction = 0
        max_mdd_reduction = 0
        if hedge_data.get('optimization'):
            for opt in hedge_data['optimization']:
                metric = opt.get('metric', '')
                achieved = opt.get('achieved_reduction', 0)
                if metric == 'cvar' and achieved > max_cvar_reduction:
                    max_cvar_reduction = achieved
                elif metric == 'mdd' and achieved > max_mdd_reduction:
                    max_mdd_reduction = achieved
        
        html += f"""
                <tr>
                    <td>{asset_name}</td>
                    <td class="text-right {'negative' if crisis_corr < 0 else ''}">{format_num(crisis_corr)}</td>
                    <td class="text-right">{format_pct(max_weight * 100)}</td>
                    <td class="text-right {'positive' if max_cvar_reduction > 0 else ''}">{format_pct(max_cvar_reduction)}</td>
                    <td class="text-right {'positive' if max_mdd_reduction > 0 else ''}">{format_pct(max_mdd_reduction)}</td>
                </tr>
"""
    
    html += """
            </table>
            <p class="footnote">Max reductions show the best achievable risk reduction within the weight constraint for each asset.</p>
"""
    
    # Detailed breakdown per asset
    for ticker, hedge_data in individual_hedges.items():
        asset_name = get_asset_name(ticker, hedge_names)
        
        # Get max weight constraint
        max_weight = 0.50
        for h in config.get('assets', {}).get('hedges', []):
            if h.get('ticker') == ticker:
                max_weight = h.get('max_weight', 0.50)
                break
        
        # Get data period info
        data_info = hedge_data.get('data_info', {})
        start_date = data_info.get('start_date', 'N/A')
        end_date = data_info.get('end_date', 'N/A')
        n_periods = data_info.get('periods', 0)
        
        html += f"""
            <div class="asset-section">
                <div class="asset-header">{asset_name} ({ticker})</div>
                <p style="margin-bottom: 8px; color: var(--text-light);">Maximum allocation: {format_pct(max_weight * 100)} | Minimum {base_ticker}: {format_pct((1 - max_weight) * 100)}</p>
                <p style="margin-bottom: 12px; color: var(--text-light); font-size: 9pt;">Analysis period: {start_date} to {end_date} ({n_periods:,} trading days)</p>
                
                <table class="comparison-table">
                    <tr>
                        <th>Target</th>
                        <th>Metric</th>
                        <th>Unhedged Risk</th>
                        <th>Hedged Risk</th>
                        <th>Weight Used</th>
                        <th>Status</th>
                    </tr>
"""
        
        if hedge_data.get('optimization'):
            for opt in hedge_data['optimization']:
                target = opt.get('target_reduction', 0)
                metric = opt.get('metric', 'cvar').upper()
                weight = opt.get('optimal_weight', 0)
                achieved = opt.get('achieved_reduction', 0)
                feasible = opt.get('feasible', False)
                at_constraint = opt.get('at_constraint', False)
                
                baseline_risk = opt.get('baseline_risk', 0)
                hedged_risk = opt.get('hedged_risk', 0)
                
                # Determine status with clear explanation
                if feasible:
                    status = f"Yes ({format_pct(achieved * 100)} reduction)"
                    status_class = "positive"
                elif achieved > 0:
                    status = f"Max: {format_pct(achieved * 100)}"
                    status_class = ""
                else:
                    status = "No reduction"
                    status_class = "negative"
                
                html += f"""
                    <tr>
                        <td>{format_pct(target * 100)} reduction</td>
                        <td>{metric}</td>
                        <td class="text-right">{format_pct(baseline_risk * 100)}</td>
                        <td class="text-right">{format_pct(hedged_risk * 100)}</td>
                        <td class="text-right">{format_pct(weight * 100)}</td>
                        <td class="text-center {status_class}">{status}</td>
                    </tr>
"""
        
        html += """
                </table>
"""
        
        # Add pre-generated rolling correlation chart
        if ticker in charts and charts[ticker]:
            html += f"""
                <div style="margin-top: 15px;">
                    <img src="data:image/png;base64,{charts[ticker]}" style="width: 100%; max-width: 800px;" alt="Rolling Correlation Chart">
                </div>
"""
        
        html += """
            </div>
"""
    
    html += """
        </section>
        
        <!-- Optimal Portfolio Strategies -->
        <section>
            <h2>Optimal Multi-Asset Portfolio Strategies</h2>
            <p>The following portfolios represent optimized allocations designed to achieve specific risk reduction targets while minimizing hedge costs.</p>
"""
    
    for target_key, portfolio in portfolios.items():
        target_label = target_key.replace('_', ' ').replace('pct', '%').title()
        
        total_hedge = portfolio.get('total_hedge_weight', 0)
        base_weight = 1 - total_hedge
        
        baseline_cvar = portfolio.get('baseline_cvar', 0)
        portfolio_cvar = portfolio.get('portfolio_cvar', 0)
        cvar_reduction = portfolio.get('cvar_reduction_pct', 0)
        
        baseline_mdd = portfolio.get('baseline_mdd', 0)
        portfolio_mdd = portfolio.get('portfolio_mdd', 0)
        mdd_reduction = portfolio.get('mdd_reduction_pct', 0)
        
        baseline_sharpe = portfolio.get('baseline_sharpe', 0)
        portfolio_sharpe = portfolio.get('portfolio_sharpe', 0)
        
        baseline_cagr = portfolio.get('baseline_cagr', 0)
        portfolio_cagr = portfolio.get('portfolio_cagr', 0)
        
        # Get portfolio metadata for date range display
        portfolio_metadata = portfolio.get('portfolio_metadata', {})
        portfolio_start = portfolio_metadata.get('portfolio_start_date')
        portfolio_end = portfolio_metadata.get('portfolio_end_date')
        asset_inceptions = portfolio_metadata.get('asset_inception_dates', {})
        
        # Format date range string
        date_range_info = ""
        if portfolio_start and portfolio_end:
            date_range_info = f"""
            <p class="info-box">
                <strong>Analysis Period:</strong> {portfolio_start.strftime('%Y-%m-%d')} to {portfolio_end.strftime('%Y-%m-%d')} 
                ({(portfolio_end - portfolio_start).days} days)
            </p>
"""
            # Add asset inception details if available
            if asset_inceptions:
                inception_list = []
                for ticker, date in sorted(asset_inceptions.items(), key=lambda x: x[1]):
                    asset_display = get_asset_name(ticker, hedge_names)
                    inception_list.append(f"{asset_display} ({ticker}): {date.strftime('%Y-%m-%d')}")
                
                if inception_list:
                    date_range_info += f"""
            <p class="footnote" style="margin-top: 10px; font-size: 0.9em; color: #666;">
                <strong>Asset Availability:</strong><br>
                {' | '.join(inception_list)}
            </p>
"""
        
        html += f"""
            <h3>{target_label}</h3>
            {date_range_info}
            
            <h4>Asset Allocation</h4>
            <table>
                <tr>
                    <th>Asset</th>
                    <th class="text-right">Weight</th>
                </tr>
                <tr>
                    <td>{base_name} ({base_ticker})</td>
                    <td class="text-right">{format_pct(base_weight * 100)}</td>
                </tr>
"""
        
        for asset, weight in portfolio.get('weights', {}).items():
            if weight > 0.001:
                asset_display = get_asset_name(asset, hedge_names)
                html += f"""
                <tr>
                    <td>{asset_display} ({asset})</td>
                    <td class="text-right">{format_pct(weight * 100)}</td>
                </tr>
"""
        
        html += f"""
                <tr class="highlight-row">
                    <td><strong>Total Hedge Allocation</strong></td>
                    <td class="text-right"><strong>{format_pct(total_hedge * 100)}</strong></td>
                </tr>
            </table>
            
            <h4>Risk-Return Comparison: Before vs. After Hedging</h4>
            <table class="comparison-table">
                <tr>
                    <th>Metric</th>
                    <th>Unhedged Portfolio</th>
                    <th>Hedged Portfolio</th>
                    <th>Change</th>
                </tr>
                <tr>
                    <td>CVaR ({cvar_conf*100:.0f}%)</td>
                    <td class="text-right">{format_pct(abs(baseline_cvar * 100))}</td>
                    <td class="text-right">{format_pct(abs(portfolio_cvar * 100))}</td>
                    <td class="text-right {'positive' if cvar_reduction > 0 else 'negative'}">{format_pct(cvar_reduction, show_sign=True)}</td>
                </tr>
                <tr>
                    <td>Maximum Drawdown</td>
                    <td class="text-right">{format_pct(abs(baseline_mdd * 100))}</td>
                    <td class="text-right">{format_pct(abs(portfolio_mdd * 100))}</td>
                    <td class="text-right {'positive' if mdd_reduction > 0 else 'negative'}">{format_pct(mdd_reduction, show_sign=True)}</td>
                </tr>
                <tr>
                    <td>Sharpe Ratio</td>
                    <td class="text-right">{format_num(baseline_sharpe)}</td>
                    <td class="text-right">{format_num(portfolio_sharpe)}</td>
                    <td class="text-right {'positive' if portfolio_sharpe > baseline_sharpe else 'negative'}">{format_num(portfolio_sharpe - baseline_sharpe, 3)}</td>
                </tr>
                <tr>
                    <td>CAGR</td>
                    <td class="text-right">{format_pct(baseline_cagr)}</td>
                    <td class="text-right">{format_pct(portfolio_cagr)}</td>
                    <td class="text-right {'positive' if portfolio_cagr >= baseline_cagr else 'negative'}">{format_pct(portfolio_cagr - baseline_cagr, show_sign=True)}</td>
                </tr>
            </table>
"""
    
    # Key Findings
    html += """
        </section>
        
        <section>
            <h2>Key Findings and Recommendations</h2>
"""
    
    # Find best crisis hedge
    best_crisis_ticker = None
    best_crisis_corr = None
    for ticker, data in individual_hedges.items():
        corr = data.get('correlations', {}).get('correlation_crisis', np.nan)
        if not np.isnan(corr) and (best_crisis_corr is None or corr < best_crisis_corr):
            best_crisis_corr = corr
            best_crisis_ticker = ticker
    
    if best_crisis_ticker:
        best_name = get_asset_name(best_crisis_ticker, hedge_names)
        html += f"""
            <div class="key-finding">
                <strong>Strongest Crisis Hedge:</strong> {best_name} ({best_crisis_ticker}) exhibited the most favorable crisis correlation of {format_num(best_crisis_corr)}, indicating strong diversification benefits during market stress.
            </div>
"""
    
    # Recommended portfolio
    portfolio_25 = portfolios.get('25pct_reduction', {})
    if portfolio_25:
        html += f"""
            <div class="key-finding">
                <strong>Recommended Allocation (25% Risk Reduction Target):</strong> 
                A total hedge allocation of {format_pct(portfolio_25.get('total_hedge_weight', 0) * 100)} achieved 
                CVaR reduction of {format_pct(portfolio_25.get('cvar_reduction_pct', 0))} and 
                maximum drawdown reduction of {format_pct(portfolio_25.get('mdd_reduction_pct', 0))}.
            </div>
"""
    
    html += """
        </section>
"""
    
    html += """
        <footer>
            <div>Tail-Risk Hedge Analysis Framework</div>
            <div class="disclaimer">
                This analysis is provided for informational purposes only and does not constitute investment advice. 
                Past performance is not indicative of future results. All investments involve risk, including the 
                possible loss of principal.
            </div>
        </footer>
    </div>
</body>
</html>
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"\nReport saved to: {output_path}")


# Compatibility alias
def generate_markdown_report(results: Dict, output_path: str = 'output/analysis_report.md') -> None:
    """Deprecated: Generates HTML report instead."""
    html_path = output_path.replace('.md', '.html')
    generate_html_report(results, html_path)
