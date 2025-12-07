"""
Tail-Risk Hedging Backtester - Main Entry Point

Author: L.Bassetti
Main entry point for running the complete tail-risk hedging analysis pipeline.

Usage:
    python main.py
"""

from datetime import datetime
from src.backtester.engine import quick_backtest
from src.reporting.report import generate_html_report


def main():
    """Run the complete tail-risk hedging analysis pipeline."""
    
    print("=" * 70)
    print("Tail-Risk Hedging Backtesting Framework")
    print("=" * 70)
    
    # Run complete backtest
    results = quick_backtest(config_path='config.yaml')
    
    # Extract returns and regime data for charts
    returns = results.pop('returns', None)
    regime_labels = results.pop('regime_labels', None)
    
    # Generate timestamped filename
    timestamp = datetime.now().strftime('%Y.%m.%d.%H.%M.%S')
    output_file = f'output/tail_risk_analysis_{timestamp}.html'
    
    # Generate HTML report with embedded charts
    generate_html_report(
        results, 
        output_path=output_file,
        returns=returns,
        regime_labels=regime_labels
    )
    
    print("\n" + "=" * 70)
    print(f"Complete! Results saved to {output_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()
