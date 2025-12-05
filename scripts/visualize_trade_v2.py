"""
Trade Visualization V2 - Enhanced Version
Shows comprehensive info for each trade including:
- Capital at trade time
- Full pair statistics (sector, half-life, p-value, hedge ratio)
- Price evolution with LONG/SHORT labels
- Z-score evolution with entry/exit thresholds
- PnL evolution showing contribution from each leg
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import sys
import yaml
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

# Default paths
DATA_PATH = project_root / "data/raw/etf_prices_fresh.csv"
OUTPUT_DIR = project_root / "results/figures"
OUTPUT_DIR.mkdir(exist_ok=True)

# Load ETF metadata for sectors
ETF_METADATA_PATH = project_root / "configs/etf_metadata.yaml"


def load_etf_sectors():
    """Load ETF to sector mapping"""
    with open(ETF_METADATA_PATH, 'r') as f:
        metadata = yaml.safe_load(f)
    
    etf_to_sector = {}
    for sector, etfs in metadata.get('sectors', {}).items():
        for etf in etfs:
            etf_to_sector[etf] = sector
    return etf_to_sector


def load_config_thresholds(trades_path):
    """Load entry/exit thresholds from config_snapshot.yaml in same folder as trades.csv"""
    trades_path = Path(trades_path)
    config_path = trades_path.parent / "config_snapshot.yaml"
    
    # Default values if config not found
    defaults = {
        'entry_threshold_sigma': 2.0,
        'exit_threshold_sigma': 0.5,
        'stop_loss_sigma': 4.0,
        'zscore_lookback': 60
    }
    
    if not config_path.exists():
        print(f"Warning: config_snapshot.yaml not found at {config_path}, using defaults")
        return defaults
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return {
            'entry_threshold_sigma': config.get('entry_threshold_sigma', defaults['entry_threshold_sigma']),
            'exit_threshold_sigma': config.get('exit_threshold_sigma', defaults['exit_threshold_sigma']),
            'stop_loss_sigma': config.get('stop_loss_sigma', defaults['stop_loss_sigma']),
            'zscore_lookback': config.get('zscore_lookback', defaults['zscore_lookback'])
        }
    except Exception as e:
        print(f"Warning: Could not load config: {e}, using defaults")
        return defaults


def find_latest_trades_file():
    """Find most recent trades file from V16b or latest backtest"""
    results_dir = project_root / "results"
    
    # Look for timestamped folders
    timestamped = sorted(results_dir.glob("2025-*_v16b_best"), reverse=True)
    if timestamped:
        trades_file = timestamped[0] / "trades.csv"
        if trades_file.exists():
            return trades_file
    
    # Fallback to other results
    timestamped = sorted(results_dir.glob("2025-*"), reverse=True)
    for folder in timestamped:
        trades_file = folder / "trades.csv"
        if trades_file.exists():
            return trades_file
    
    # Last resort
    return results_dir / "backtest_v4_trades.csv"


def load_data(trades_path=None):
    """Load price data and trades"""
    prices = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    
    if trades_path is None:
        trades_path = find_latest_trades_file()
    else:
        trades_path = Path(trades_path)
    
    print(f"Loading trades from: {trades_path}")
    trades = pd.read_csv(trades_path, parse_dates=['entry_date', 'exit_date'])
    
    return prices, trades, trades_path


def visualize_trade_enhanced(trade_row, prices, capital_at_entry=None, 
                              context_days=30, save=True, output_dir=None,
                              config_thresholds=None):
    """
    Enhanced trade visualization with comprehensive information
    
    Parameters:
    -----------
    trade_row : pd.Series
        Row from trades DataFrame
    prices : pd.DataFrame
        Price data
    capital_at_entry : float, optional
        Capital at time of entry (for context)
    context_days : int
        Days before/after trade to show
    save : bool
        Whether to save figure
    output_dir : Path, optional
        Custom output directory
    config_thresholds : dict, optional
        Dict with entry_threshold_sigma, exit_threshold_sigma, stop_loss_sigma
    """
    # Use default thresholds if not provided
    if config_thresholds is None:
        config_thresholds = {
            'entry_threshold_sigma': 2.0,
            'exit_threshold_sigma': 0.5, 
            'stop_loss_sigma': 4.0,
            'zscore_lookback': 60
        }
    
    etf_sectors = load_etf_sectors()
    
    # Extract trade info
    etf1 = trade_row['leg_x']
    etf2 = trade_row['leg_y']
    entry_date = pd.to_datetime(trade_row['entry_date'])
    exit_date = pd.to_datetime(trade_row['exit_date'])
    direction = trade_row['direction']
    pnl = trade_row['pnl']
    hedge_ratio = trade_row['hedge_ratio']
    exit_reason = trade_row.get('exit_reason', 'unknown')
    entry_z = trade_row.get('entry_z', np.nan)
    exit_z = trade_row.get('exit_z', np.nan)
    
    # Get additional stats if available
    half_life = trade_row.get('half_life', np.nan)
    pvalue = trade_row.get('pvalue', np.nan)
    sector = trade_row.get('sector', etf_sectors.get(etf1, 'UNKNOWN'))
    trading_year = trade_row.get('trading_year', entry_date.year)
    
    # Capital info
    if capital_at_entry is None:
        capital_at_entry = trade_row.get('capital_at_entry', 50000)
    
    # Position sizes - get from CSV if available
    qty_x = trade_row.get('qty_x', np.nan)
    qty_y = trade_row.get('qty_y', np.nan)
    entry_px = trade_row.get('entry_px', np.nan)
    entry_py = trade_row.get('entry_py', np.nan)
    
    # Calculate actual position capital from quantities if available
    if not np.isnan(qty_x) and not np.isnan(entry_px):
        position_capital = abs(qty_x) * entry_px + abs(qty_y) * entry_py
    else:
        position_capital = trade_row.get('position_capital', 15000)
    
    # Determine which ETF is LONG and which is SHORT
    is_long_spread = (direction == "LONG" or direction == 1)
    
    # Get date range with context
    start_date = entry_date - pd.Timedelta(days=context_days)
    end_date = exit_date + pd.Timedelta(days=context_days)
    
    # Get price data
    mask = (prices.index >= start_date) & (prices.index <= end_date)
    p1 = prices.loc[mask, etf1].dropna()
    p2 = prices.loc[mask, etf2].dropna()
    
    # Align dates
    common_idx = p1.index.intersection(p2.index)
    p1 = p1.loc[common_idx]
    p2 = p2.loc[common_idx]
    
    if len(p1) < 5:
        print(f"Not enough data for {etf1}/{etf2}")
        return None
    
    # Find entry and exit indices
    entry_idx = common_idx.get_indexer([entry_date], method='nearest')[0]
    exit_idx = common_idx.get_indexer([exit_date], method='nearest')[0]
    
    # Calculate % change from ENTRY date
    entry_p1 = p1.iloc[entry_idx]
    entry_p2 = p2.iloc[entry_idx]
    pct_change1 = (p1 / entry_p1 - 1) * 100
    pct_change2 = (p2 / entry_p2 - 1) * 100
    
    # Calculate spreads
    log_spread = np.log(p1) - hedge_ratio * np.log(p2)
    
    # Calculate z-score with lookback from config (align with engine's adaptive logic)
    lookback = config_thresholds.get('zscore_lookback', 60)
    if not np.isnan(half_life):
        lookback = int(max(30, min(120, 4 * half_life)))  # clamp to [30, 120]
        lookback = min(lookback, len(log_spread))  # cannot exceed available
    else:
        lookback = min(lookback, len(log_spread) // 2)
    spread_mean = log_spread.rolling(window=lookback, min_periods=10).mean()
    spread_std = log_spread.rolling(window=lookback, min_periods=10).std()
    zscore = (log_spread - spread_mean) / spread_std
    
    # Calculate PnL evolution during trade
    if is_long_spread:
        if not np.isnan(qty_x):
            pnl_x = qty_x * (p1 - entry_p1)
            pnl_y = qty_y * (p2 - entry_p2)
        else:
            notional_x = position_capital / (1 + abs(hedge_ratio))
            notional_y = abs(hedge_ratio) * notional_x
            qty_x = notional_x / entry_p1
            qty_y = -notional_y / entry_p2
            pnl_x = qty_x * (p1 - entry_p1)
            pnl_y = qty_y * (p2 - entry_p2)
    else:
        if not np.isnan(qty_x):
            pnl_x = qty_x * (p1 - entry_p1)
            pnl_y = qty_y * (p2 - entry_p2)
        else:
            notional_x = position_capital / (1 + abs(hedge_ratio))
            notional_y = abs(hedge_ratio) * notional_x
            qty_x = -notional_x / entry_p1
            qty_y = notional_y / entry_p2
            pnl_x = qty_x * (p1 - entry_p1)
            pnl_y = qty_y * (p2 - entry_p2)
    
    total_pnl = pnl_x + pnl_y
    
    # Create figure with better layout - more space for header
    fig = plt.figure(figsize=(16, 20))
    
    # Use gridspec - give much more space to header (4 text lines)
    gs = fig.add_gridspec(4, 2, height_ratios=[0.28, 1, 1, 0.9], 
                          hspace=0.22, wspace=0.25,
                          left=0.06, right=0.94, top=0.96, bottom=0.04)
    
    # ========== ROW 0: Title + Info (4 lines with proper spacing) ==========
    ax_header = fig.add_subplot(gs[0, :])
    ax_header.axis('off')
    
    result_text = "WIN" if pnl > 0 else "LOSS"
    result_color = '#27ae60' if pnl > 0 else '#e74c3c'
    holding_days = (exit_date - entry_date).days
    pnl_pct = (pnl / position_capital) * 100
    hl_ratio = holding_days / half_life if half_life > 0 else 0
    
    # Line 1: Title (top) - y=0.88
    title_str = (f"[{result_text}] {etf1} / {etf2}  |  "
                 f"{entry_date.strftime('%Y-%m-%d')} → {exit_date.strftime('%Y-%m-%d')} ({holding_days}d)  |  "
                 f"PnL: ${pnl:+,.0f} ({pnl_pct:+.1f}%)")
    ax_header.text(0.5, 0.88, title_str, transform=ax_header.transAxes,
                   fontsize=14, fontweight='bold', color=result_color,
                   verticalalignment='center', horizontalalignment='center')
    
    # Line 2: Sector, Direction, Exit - y=0.62
    info_line1 = (f"Sector: {sector}  |  "
                  f"{'LONG' if is_long_spread else 'SHORT'} spread "
                  f"({etf1} {'L' if is_long_spread else 'S'} / {etf2} {'S' if is_long_spread else 'L'})  |  "
                  f"Exit: {exit_reason}")
    ax_header.text(0.5, 0.62, info_line1, transform=ax_header.transAxes,
                   fontsize=10, verticalalignment='center', horizontalalignment='center')
    
    # Line 3: Capital and Hedge - y=0.38
    pvalue_str = f"{pvalue:.3f}" if not np.isnan(pvalue) else "N/A"
    info_line2 = (
        f"Portfolio: ${capital_at_entry:,.0f}  |  Position: ${position_capital:,.0f}  |  "
        f"Hedge Ratio: {hedge_ratio:.3f}  |  P-value: {pvalue_str}  |  Year: {trading_year}"
    )
    ax_header.text(0.5, 0.38, info_line2, transform=ax_header.transAxes,
                   fontsize=9, verticalalignment='center', horizontalalignment='center',
                   color='#444444')
    
    # Line 4: Z-scores and timing - y=0.14
    hl_color = '#c0392b' if hl_ratio > 2.5 else '#555555'
    stats_line = (f"Entry Z: {entry_z:.2f} → Exit Z: {exit_z:.2f}  |  "
                  f"Half-life: {half_life:.1f}d  |  Holding: {holding_days}d ({hl_ratio:.1f}x HL)")
    ax_header.text(0.5, 0.14, stats_line, transform=ax_header.transAxes,
                   fontsize=9, verticalalignment='center', horizontalalignment='center',
                   color=hl_color)
    
    # ========== PANEL 1: ETF1 Price ==========
    ax1 = fig.add_subplot(gs[1, 0])
    
    # Price line
    ax1.plot(p1.index, p1.values, color='#3498db', linewidth=2, label=f'{etf1} Price')
    
    # Entry/Exit markers
    ax1.axvline(entry_date, color='blue', linestyle='--', alpha=0.7, label='Entry')
    ax1.axvline(exit_date, color='purple', linestyle='--', alpha=0.7, label='Exit')
    
    # Shade trade period
    ax1.axvspan(entry_date, exit_date, alpha=0.1, color='blue')
    
    # Entry point
    ax1.scatter([common_idx[entry_idx]], [entry_p1], marker='^', s=150, 
                color='blue', edgecolor='black', linewidth=1.5, zorder=5)
    ax1.scatter([common_idx[exit_idx]], [p1.iloc[exit_idx]], marker='v', s=150, 
                color='purple', edgecolor='black', linewidth=1.5, zorder=5)
    
    position_label = "LONG" if is_long_spread else "SHORT"
    ax1.set_title(f'{etf1} ({position_label})', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Price ($)', fontsize=10)
    ax1.legend(loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # ========== PANEL 2: ETF2 Price ==========
    ax2 = fig.add_subplot(gs[1, 1])
    
    ax2.plot(p2.index, p2.values, color='#e74c3c', linewidth=2, label=f'{etf2} Price')
    
    ax2.axvline(entry_date, color='blue', linestyle='--', alpha=0.7, label='Entry')
    ax2.axvline(exit_date, color='purple', linestyle='--', alpha=0.7, label='Exit')
    ax2.axvspan(entry_date, exit_date, alpha=0.1, color='blue')
    
    ax2.scatter([common_idx[entry_idx]], [entry_p2], marker='^', s=150, 
                color='blue', edgecolor='black', linewidth=1.5, zorder=5)
    ax2.scatter([common_idx[exit_idx]], [p2.iloc[exit_idx]], marker='v', s=150, 
                color='purple', edgecolor='black', linewidth=1.5, zorder=5)
    
    position_label = "SHORT" if is_long_spread else "LONG"
    ax2.set_title(f'{etf2} ({position_label})', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Price ($)', fontsize=10)
    ax2.legend(loc='upper left', fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # ========== PANEL 3: % Change Comparison ==========
    ax3 = fig.add_subplot(gs[2, 0])
    
    if is_long_spread:
        color1, color2 = '#27ae60', '#e74c3c'
        label1, label2 = f'{etf1} (LONG)', f'{etf2} (SHORT)'
    else:
        color1, color2 = '#e74c3c', '#27ae60'
        label1, label2 = f'{etf1} (SHORT)', f'{etf2} (LONG)'
    
    ax3.plot(pct_change1.index, pct_change1.values, label=label1, color=color1, linewidth=2)
    ax3.plot(pct_change2.index, pct_change2.values, label=label2, color=color2, linewidth=2)
    
    ax3.axvspan(entry_date, exit_date, alpha=0.15, color='green' if pnl > 0 else 'red')
    ax3.axhline(0, color='gray', linestyle='--', alpha=0.7)
    
    # Entry/Exit markers
    ax3.scatter([common_idx[entry_idx]], [0], marker='^', s=150, 
                color='blue', edgecolor='black', linewidth=1.5, zorder=5)
    
    exit_y1 = pct_change1.iloc[exit_idx]
    exit_y2 = pct_change2.iloc[exit_idx]
    ax3.scatter([common_idx[exit_idx]], [exit_y1], marker='v', s=100, 
                color=color1, edgecolor='black', linewidth=1, zorder=5)
    ax3.scatter([common_idx[exit_idx]], [exit_y2], marker='v', s=100, 
                color=color2, edgecolor='black', linewidth=1, zorder=5)
    
    # Annotate exit values
    ax3.annotate(f'{exit_y1:+.1f}%', xy=(common_idx[exit_idx], exit_y1),
                 xytext=(5, 5), textcoords='offset points', fontsize=9, color=color1)
    ax3.annotate(f'{exit_y2:+.1f}%', xy=(common_idx[exit_idx], exit_y2),
                 xytext=(5, -10), textcoords='offset points', fontsize=9, color=color2)
    
    ax3.set_title('% Change from Entry', fontsize=11, fontweight='bold')
    ax3.set_ylabel('% Change', fontsize=10)
    ax3.legend(loc='best', fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # ========== PANEL 4: Z-Score ==========
    ax4 = fig.add_subplot(gs[2, 1])
    
    ax4.plot(zscore.index, zscore.values, color='#2c3e50', linewidth=2, label='Z-Score')
    
    # Thresholds from config (FIX: was hardcoded 2.8/0.3/3.0)
    entry_thresh = config_thresholds['entry_threshold_sigma']
    exit_thresh = config_thresholds['exit_threshold_sigma']
    stop_thresh = config_thresholds['stop_loss_sigma']
    
    ax4.axhline(entry_thresh, color='red', linestyle='--', alpha=0.7, label=f'Entry (±{entry_thresh})')
    ax4.axhline(-entry_thresh, color='red', linestyle='--', alpha=0.7)
    ax4.axhline(exit_thresh, color='green', linestyle=':', alpha=0.7, label=f'Exit (±{exit_thresh})')
    ax4.axhline(-exit_thresh, color='green', linestyle=':', alpha=0.7)
    ax4.axhline(0, color='gray', linestyle='-', alpha=0.5)
    ax4.axhline(stop_thresh, color='darkred', linestyle='-.', alpha=0.5, label=f'Stop (±{stop_thresh})')
    ax4.axhline(-stop_thresh, color='darkred', linestyle='-.', alpha=0.5)
    
    ax4.axvspan(entry_date, exit_date, alpha=0.15, color='green' if pnl > 0 else 'red')
    
    # Entry/Exit z markers
    # Use actual z from trades.csv FIRST (these are what engine used),
    # fallback to recalculated rolling z only if missing.
    if not np.isnan(entry_z):
        calc_entry_z = entry_z
    else:
        calc_entry_z = zscore.iloc[entry_idx] if not np.isnan(zscore.iloc[entry_idx]) else 0.0
    
    if not np.isnan(exit_z):
        calc_exit_z = exit_z
    else:
        calc_exit_z = zscore.iloc[exit_idx] if not np.isnan(zscore.iloc[exit_idx]) else 0.0
    
    ax4.scatter([common_idx[entry_idx]], [calc_entry_z], marker='^', s=150, 
                color='blue', edgecolor='black', linewidth=1.5, zorder=5)
    ax4.scatter([common_idx[exit_idx]], [calc_exit_z], marker='v', s=150, 
                color='purple', edgecolor='black', linewidth=1.5, zorder=5)
    
    ax4.annotate(f'{calc_entry_z:.2f}', xy=(common_idx[entry_idx], calc_entry_z),
                 xytext=(5, 5), textcoords='offset points', fontsize=8, color='blue')
    ax4.annotate(f'{calc_exit_z:.2f}', xy=(common_idx[exit_idx], calc_exit_z),
                 xytext=(5, -10), textcoords='offset points', fontsize=8, color='purple')
    
    # Clarify that plotted z-score uses rolling window; engine may use fixed exit params.
    ax4.text(0.01, 0.05, "Z shown = rolling; engine uses entry-time params for exits",
             transform=ax4.transAxes, fontsize=7, color='#555555', ha='left', va='bottom',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.7, edgecolor='none'))
    
    ax4.set_title('Spread Z-Score', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Z-Score', fontsize=10)
    ax4.legend(loc='upper right', fontsize=7, ncol=2)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(-5, 5)
    
    # ========== PANEL 5: PnL Evolution (spans both columns) ==========
    ax5 = fig.add_subplot(gs[3, :])
    
    # Only show PnL from entry onwards
    trade_mask = common_idx >= entry_date
    pnl_during = total_pnl[trade_mask]
    pnl_x_during = pnl_x[trade_mask]
    pnl_y_during = pnl_y[trade_mask]
    
    # Plot individual legs
    ax5.plot(pnl_x_during.index, pnl_x_during.values, 
             label=f'{etf1} leg', color=color1, linewidth=1.5, alpha=0.7, linestyle='--')
    ax5.plot(pnl_y_during.index, pnl_y_during.values, 
             label=f'{etf2} leg', color=color2, linewidth=1.5, alpha=0.7, linestyle='--')
    
    # Total PnL
    ax5.fill_between(pnl_during.index, 0, pnl_during.values, 
                     where=pnl_during.values >= 0, 
                     color='green', alpha=0.3, interpolate=True)
    ax5.fill_between(pnl_during.index, 0, pnl_during.values, 
                     where=pnl_during.values < 0, 
                     color='red', alpha=0.3, interpolate=True)
    ax5.plot(pnl_during.index, pnl_during.values, color='black', linewidth=2.5, label='Total PnL')
    
    ax5.axvspan(entry_date, exit_date, alpha=0.1, color='blue')
    ax5.axhline(0, color='gray', linestyle='-', alpha=0.5)
    
    # Markers
    ax5.scatter([common_idx[entry_idx]], [0], marker='^', s=150, 
                color='blue', edgecolor='black', linewidth=1.5, zorder=5)
    exit_pnl = total_pnl.iloc[exit_idx]
    ax5.scatter([common_idx[exit_idx]], [exit_pnl], marker='v', s=150, 
                color='purple', edgecolor='black', linewidth=1.5, zorder=5)
    
    # Annotation - position based on PnL sign
    ax5.annotate(f'Exit: ${exit_pnl:+,.0f} (net: ${pnl:+,.0f})', 
                 xy=(common_idx[exit_idx], exit_pnl),
                 xytext=(10, 5 if exit_pnl >= 0 else -15), textcoords='offset points',
                 fontsize=9, 
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))
    
    ax5.set_title('PnL Evolution', fontsize=11, fontweight='bold')
    ax5.set_ylabel('PnL ($)', fontsize=10)
    ax5.set_xlabel('Date', fontsize=10)
    ax5.legend(loc='upper left', fontsize=8)
    ax5.grid(True, alpha=0.3)
    
    # Date formatting
    for ax in [ax1, ax2, ax3, ax4, ax5]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    if save:
        if output_dir is None:
            output_dir = OUTPUT_DIR
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        result_str = "WIN" if pnl > 0 else "LOSS"
        filename = f"trade_{result_str}_{etf1}_{etf2}_{entry_date.strftime('%Y%m%d')}.png"
        filepath = output_dir / filename
        plt.savefig(filepath, dpi=150, bbox_inches='tight', 
                    facecolor='white', edgecolor='none')
        print(f"Saved: {filepath}")
    
    plt.show()
    return fig


def show_trades_by_year(year, trades_path=None, max_trades=None, save=True):
    """Show all trades for a specific year"""
    prices, trades, _ = load_data(trades_path)
    
    year_trades = trades[trades['trading_year'] == year] if 'trading_year' in trades.columns else \
                  trades[pd.to_datetime(trades['entry_date']).dt.year == year]
    
    if len(year_trades) == 0:
        print(f"No trades found for year {year}")
        return
    
    if max_trades:
        year_trades = year_trades.head(max_trades)
    
    print(f"\n{'='*60}")
    print(f"TRADES FOR YEAR {year} ({len(year_trades)} trades)")
    print(f"{'='*60}")
    
    for i, (idx, trade) in enumerate(year_trades.iterrows()):
        print(f"\n--- Trade {i+1}/{len(year_trades)}: {trade['leg_x']}/{trade['leg_y']} | "
              f"PnL: ${trade['pnl']:+,.2f} ---")
        visualize_trade_enhanced(trade, prices, save=save)


def show_best_worst_trades(n=3, trades_path=None, save=True):
    """Show n best and n worst trades"""
    prices, trades, _ = load_data(trades_path)
    
    sorted_trades = trades.sort_values('pnl', ascending=False)
    
    print("\n" + "="*60)
    print(f"TOP {n} WINNING TRADES")
    print("="*60)
    
    for i, (idx, trade) in enumerate(sorted_trades.head(n).iterrows()):
        print(f"\n--- Win #{i+1}: {trade['leg_x']}/{trade['leg_y']} | PnL: ${trade['pnl']:+,.2f} ---")
        visualize_trade_enhanced(trade, prices, save=save)
    
    print("\n" + "="*60)
    print(f"TOP {n} LOSING TRADES")
    print("="*60)
    
    for i, (idx, trade) in enumerate(sorted_trades.tail(n).iterrows()):
        print(f"\n--- Loss #{i+1}: {trade['leg_x']}/{trade['leg_y']} | PnL: ${trade['pnl']:+,.2f} ---")
        visualize_trade_enhanced(trade, prices, save=save)


def show_trades_by_exit_reason(reason, trades_path=None, max_trades=5, mix_win_loss=True, save=True):
    """Show trades filtered by exit reason
    
    Args:
        reason: Exit reason to filter
        trades_path: Path to trades CSV
        max_trades: Max trades to show
        mix_win_loss: If True, show mix of winners and losers
    """
    prices, trades, _ = load_data(trades_path)
    
    if 'exit_reason' not in trades.columns:
        print("No exit_reason column in trades file")
        return
    
    filtered = trades[trades['exit_reason'] == reason]
    
    if len(filtered) == 0:
        print(f"No trades with exit reason: {reason}")
        print(f"Available reasons: {trades['exit_reason'].unique().tolist()}")
        return
    
    # Stats
    winners = filtered[filtered['pnl'] > 0]
    losers = filtered[filtered['pnl'] < 0]
    
    print(f"\n{'='*60}")
    print(f"TRADES WITH EXIT REASON: {reason}")
    print(f"{'='*60}")
    print(f"Total: {len(filtered)} | Winners: {len(winners)} | Losers: {len(losers)}")
    print(f"Total PnL: ${filtered['pnl'].sum():+,.2f}")
    print()
    
    if mix_win_loss and max_trades:
        # Show mix of best winners and worst losers
        n_winners = min(len(winners), (max_trades + 1) // 2)
        n_losers = min(len(losers), max_trades - n_winners)
        
        to_show = pd.concat([
            winners.nlargest(n_winners, 'pnl'),
            losers.nsmallest(n_losers, 'pnl')
        ]).sort_values('pnl', ascending=False)
    else:
        to_show = filtered.head(max_trades) if max_trades else filtered
    
    print(f"Showing {len(to_show)} trades:\n")
    
    for i, (idx, trade) in enumerate(to_show.iterrows()):
        result = "WIN" if trade['pnl'] > 0 else "LOSS"
        print(f"--- Trade {i+1} [{result}]: {trade['leg_x']}/{trade['leg_y']} | PnL: ${trade['pnl']:+,.2f} ---")
        visualize_trade_enhanced(trade, prices, save=save)


def list_trades_summary(trades_path=None):
    """Print summary table of all trades"""
    _, trades, path = load_data(trades_path)
    
    print(f"\n{'='*80}")
    print(f"TRADES SUMMARY from {path}")
    print(f"{'='*80}")
    
    # Summary stats
    print(f"\nTotal trades: {len(trades)}")
    print(f"Winners: {len(trades[trades['pnl'] > 0])} ({len(trades[trades['pnl'] > 0])/len(trades)*100:.1f}%)")
    print(f"Total PnL: ${trades['pnl'].sum():,.2f}")
    
    # By exit reason
    if 'exit_reason' in trades.columns:
        print("\nBy Exit Reason:")
        for reason in trades['exit_reason'].unique():
            subset = trades[trades['exit_reason'] == reason]
            print(f"  {reason}: {len(subset)} trades, ${subset['pnl'].sum():+,.2f}")
    
    # Top 10 trades
    print(f"\n{'='*80}")
    print("TOP 10 TRADES BY PnL:")
    print(f"{'='*80}")
    
    top10 = trades.nlargest(10, 'pnl')[['leg_x', 'leg_y', 'entry_date', 'pnl', 'exit_reason']]
    top10['entry_date'] = pd.to_datetime(top10['entry_date']).dt.strftime('%Y-%m-%d')
    print(top10.to_string(index=False))
    
    print(f"\n{'='*80}")
    print("BOTTOM 10 TRADES BY PnL:")
    print(f"{'='*80}")
    
    bottom10 = trades.nsmallest(10, 'pnl')[['leg_x', 'leg_y', 'entry_date', 'pnl', 'exit_reason']]
    bottom10['entry_date'] = pd.to_datetime(bottom10['entry_date']).dt.strftime('%Y-%m-%d')
    print(bottom10.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Enhanced Trade Visualization')
    parser.add_argument('--trades', '-t', type=str, default=None,
                        help='Path to trades CSV file (default: latest V16b)')
    parser.add_argument('--best-worst', '-bw', type=int, default=None,
                        help='Show N best and N worst trades')
    parser.add_argument('--year', '-y', type=int, default=None,
                        help='Show trades for specific year')
    parser.add_argument('--exit-reason', '-e', type=str, default=None,
                        help='Show trades with specific exit reason')
    parser.add_argument('--pair', '-p', nargs=2, default=None,
                        help='Show specific pair (e.g., -p XLF XLI)')
    parser.add_argument('--index', '-i', type=int, default=None,
                        help='Show trade at specific index')
    parser.add_argument('--max', '-m', type=int, default=5,
                        help='Max trades to show per query')
    parser.add_argument('--list', '-l', action='store_true',
                        help='List all trades summary')
    parser.add_argument('--no-save', action='store_true',
                        help='Do not save figures')
    parser.add_argument('--all', '-a', action='store_true',
                        help='Generate visualizations for ALL trades')
    
    args = parser.parse_args()
    
    save = not args.no_save
    
    # Load config thresholds from trades file location
    prices, trades, trades_path = load_data(args.trades)
    config_thresholds = load_config_thresholds(trades_path)
    print(f"Using thresholds: entry=±{config_thresholds['entry_threshold_sigma']}, "
          f"exit=±{config_thresholds['exit_threshold_sigma']}, "
          f"stop=±{config_thresholds['stop_loss_sigma']}")
    
    if args.all:
        # Generate all trades
        print(f"\nGenerating {len(trades)} visualizations...")
        for idx, trade in trades.iterrows():
            try:
                visualize_trade_enhanced(trade, prices, save=save, 
                                        config_thresholds=config_thresholds)
            except Exception as e:
                print(f"Error on trade {idx}: {e}")
        print(f"\nDone! Generated {len(trades)} figures in results/figures/")
    elif args.list:
        list_trades_summary(args.trades)
    elif args.best_worst:
        show_best_worst_trades(args.best_worst, args.trades, save=save)
    elif args.year:
        show_trades_by_year(args.year, args.trades, args.max, save=save)
    elif args.exit_reason:
        show_trades_by_exit_reason(args.exit_reason, args.trades, args.max, save=save)
    elif args.pair:
        etf1, etf2 = args.pair
        mask = ((trades['leg_x'] == etf1) & (trades['leg_y'] == etf2)) | \
               ((trades['leg_x'] == etf2) & (trades['leg_y'] == etf1))
        matching = trades[mask]
        if len(matching) == 0:
            print(f"No trades found for {etf1}/{etf2}")
        else:
            for idx, trade in matching.head(args.max).iterrows():
                visualize_trade_enhanced(trade, prices, save=save,
                                        config_thresholds=config_thresholds)
    elif args.index is not None:
        if 0 <= args.index < len(trades):
            visualize_trade_enhanced(trades.iloc[args.index], prices, save=save,
                                    config_thresholds=config_thresholds)
        else:
            print(f"Index out of range. Total trades: {len(trades)}")
    else:
        # Default: show summary then ask
        list_trades_summary(args.trades)
        print("\n" + "="*60)
        print("Use --help for options, or try:")
        print("  --best-worst 3   : Show 3 best and 3 worst trades")
        print("  --year 2020      : Show trades from 2020")
        print("  --exit-reason convergence : Show convergence trades")
        print("  --index 0        : Show first trade")
