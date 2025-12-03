"""
Performance metrics and reporting for pairs trading backtests.

This module provides functions for calculating and formatting
backtest performance metrics.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

import pandas as pd
import yaml


def calculate_performance_metrics(trades: List[Dict]) -> Dict[str, Any]:
    """
    Calculate comprehensive performance metrics from trade list.
    
    Parameters
    ----------
    trades : list
        List of trade dictionaries
        
    Returns
    -------
    dict
        Performance metrics
    """
    if not trades:
        return {
            'total_pnl': 0.0,
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0.0,
            'avg_pnl': 0.0,
            'avg_winner': 0.0,
            'avg_loser': 0.0,
            'profit_factor': 0.0,
            'max_drawdown': 0.0,
        }
    
    df = pd.DataFrame(trades)
    
    total_pnl = df['pnl'].sum()
    n_trades = len(df)
    winners = df[df['pnl'] > 0]
    losers = df[df['pnl'] <= 0]
    
    gross_profit = winners['pnl'].sum() if len(winners) > 0 else 0
    gross_loss = abs(losers['pnl'].sum()) if len(losers) > 0 else 0
    
    # Calculate cumulative PnL for drawdown
    cum_pnl = df['pnl'].cumsum()
    running_max = cum_pnl.cummax()
    drawdown = running_max - cum_pnl
    max_drawdown = drawdown.max()
    
    return {
        'total_pnl': float(total_pnl),
        'total_trades': n_trades,
        'winning_trades': len(winners),
        'losing_trades': len(losers),
        'win_rate': len(winners) / n_trades * 100 if n_trades > 0 else 0,
        'avg_pnl': float(df['pnl'].mean()),
        'avg_winner': float(winners['pnl'].mean()) if len(winners) > 0 else 0,
        'avg_loser': float(losers['pnl'].mean()) if len(losers) > 0 else 0,
        'profit_factor': gross_profit / gross_loss if gross_loss > 0 else float('inf'),
        'max_drawdown': float(max_drawdown),
        'avg_holding_days': float(df['holding_days'].mean()),
    }


def pnl_by_exit_reason(trades: List[Dict]) -> Dict[str, Dict[str, float]]:
    """
    Calculate PnL breakdown by exit reason.
    
    Parameters
    ----------
    trades : list
        List of trade dictionaries
        
    Returns
    -------
    dict
        PnL stats by exit reason
    """
    if not trades:
        return {}
    
    df = pd.DataFrame(trades)
    
    result = {}
    for reason in df['exit_reason'].unique():
        subset = df[df['exit_reason'] == reason]
        result[reason] = {
            'total_pnl': float(subset['pnl'].sum()),
            'trades': len(subset),
            'avg_pnl': float(subset['pnl'].mean()),
            'win_rate': len(subset[subset['pnl'] > 0]) / len(subset) * 100,
        }
    
    return result


def pnl_by_sector(trades: List[Dict]) -> Dict[str, Dict[str, float]]:
    """
    Calculate PnL breakdown by sector.
    
    Parameters
    ----------
    trades : list
        List of trade dictionaries
        
    Returns
    -------
    dict
        PnL stats by sector
    """
    if not trades:
        return {}
    
    df = pd.DataFrame(trades)
    
    result = {}
    for sector in df['sector'].unique():
        subset = df[df['sector'] == sector]
        result[sector] = {
            'total_pnl': float(subset['pnl'].sum()),
            'trades': len(subset),
            'avg_pnl': float(subset['pnl'].mean()),
            'win_rate': len(subset[subset['pnl'] > 0]) / len(subset) * 100,
        }
    
    return result


def print_backtest_report(
    trades: List[Dict],
    summary_df: pd.DataFrame,
    config_name: str = "default",
) -> None:
    """
    Print formatted backtest report to console.
    
    Parameters
    ----------
    trades : list
        List of trade dictionaries
    summary_df : pd.DataFrame
        Yearly summary dataframe
    config_name : str
        Name of the configuration used
    """
    print("\n" + "=" * 70)
    print(f"BACKTEST RESULTS - {config_name}")
    print("=" * 70)
    
    if len(trades) == 0:
        print("No trades executed.")
        return
    
    # Overall metrics
    metrics = calculate_performance_metrics(trades)
    
    print("\n" + "-" * 40)
    print("PERFORMANCE SUMMARY")
    print("-" * 40)
    print(f"Total PnL:        ${metrics['total_pnl']:,.2f}")
    print(f"Total Trades:     {metrics['total_trades']}")
    print(f"Win Rate:         {metrics['win_rate']:.1f}%")
    print(f"Avg Winner:       ${metrics['avg_winner']:,.2f}")
    print(f"Avg Loser:        ${metrics['avg_loser']:,.2f}")
    print(f"Profit Factor:    {metrics['profit_factor']:.2f}")
    print(f"Max Drawdown:     ${metrics['max_drawdown']:,.2f}")
    print(f"Avg Holding Days: {metrics['avg_holding_days']:.1f}")
    
    # By exit reason
    print("\n" + "-" * 40)
    print("PnL BY EXIT REASON")
    print("-" * 40)
    by_reason = pnl_by_exit_reason(trades)
    for reason, stats in sorted(by_reason.items(), key=lambda x: x[1]['total_pnl'], reverse=True):
        print(f"{reason}: ${stats['total_pnl']:,.2f} ({stats['trades']} trades, avg ${stats['avg_pnl']:.2f})")
    
    # By sector
    print("\n" + "-" * 40)
    print("PnL BY SECTOR")
    print("-" * 40)
    by_sector = pnl_by_sector(trades)
    for sector, stats in sorted(by_sector.items(), key=lambda x: x[1]['total_pnl'], reverse=True):
        print(f"{sector}: ${stats['total_pnl']:,.2f} ({stats['trades']} trades)")
    
    # Yearly summary
    if len(summary_df) > 0:
        print("\n" + "-" * 40)
        print("YEARLY BREAKDOWN")
        print("-" * 40)
        print(summary_df[['trading_year', 'pairs_selected', 'total_trades', 'win_rate', 'total_pnl']].to_string(index=False))


def save_results(
    trades: List[Dict],
    summary_df: pd.DataFrame,
    config: Any,
    output_dir: Path,
) -> None:
    """
    Save backtest results to files.
    
    Parameters
    ----------
    trades : list
        List of trade dictionaries
    summary_df : pd.DataFrame
        Yearly summary
    config : BacktestConfig
        Configuration used
    output_dir : Path
        Output directory path
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save trades
    if trades:
        trades_df = pd.DataFrame(trades)
        # Convert tuple pairs to string for CSV
        trades_df['pair'] = trades_df['pair'].apply(lambda x: f"{x[0]}_{x[1]}")
        trades_df.to_csv(output_dir / "trades.csv", index=False)
    
    # Save summary
    if len(summary_df) > 0:
        summary_df.to_csv(output_dir / "summary.csv", index=False)
    
    # Save config snapshot
    if hasattr(config, 'to_dict'):
        config_dict = config.to_dict()
    else:
        config_dict = {k: v for k, v in config.__dict__.items() if not k.startswith('_')}
    
    config_dict['_saved_at'] = datetime.now().isoformat()
    
    with open(output_dir / "config_snapshot.yaml", 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
    
    # Save metrics
    if trades:
        metrics = calculate_performance_metrics(trades)
        metrics['by_exit_reason'] = pnl_by_exit_reason(trades)
        metrics['by_sector'] = pnl_by_sector(trades)
        
        with open(output_dir / "metrics.yaml", 'w') as f:
            yaml.dump(metrics, f, default_flow_style=False)
    
    print(f"\nResults saved to: {output_dir}")
