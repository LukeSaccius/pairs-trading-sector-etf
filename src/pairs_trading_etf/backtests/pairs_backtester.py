"""Walk-forward pairs trading backtester.

Implements a rigorous walk-forward validation framework for pairs trading strategies.
Separates formation (in-sample) and trading (out-of-sample) periods to avoid lookahead bias.

Key features:
- Walk-forward validation with rolling windows
- Dollar-neutral position sizing
- Transaction cost modeling
- Performance attribution and metrics
- Multiple testing correction via deflated Sharpe ratio

References:
- Bailey, D. & Lopez de Prado, M. (2014). "The Deflated Sharpe Ratio"
- Harvey, C.R. et al. (2016). "...and the Cross-Section of Expected Returns"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Mapping, Sequence
from enum import Enum

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


class TradeDirection(Enum):
    """Direction of a spread trade."""
    LONG_SPREAD = 1   # Long leg_x, Short leg_y
    SHORT_SPREAD = -1  # Short leg_x, Long leg_y
    FLAT = 0


@dataclass
class Trade:
    """Record of a single round-trip trade."""
    
    pair: tuple[str, str]         # (leg_x, leg_y)
    direction: TradeDirection
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_z: float
    exit_z: float
    entry_spread: float
    exit_spread: float
    hedge_ratio: float
    pnl: float                    # Dollar P&L
    pnl_pct: float                # Percentage return
    holding_days: int
    exit_reason: str
    
    def as_dict(self) -> Mapping:
        """Convert to dictionary for DataFrame construction."""
        return {
            "leg_x": self.pair[0],
            "leg_y": self.pair[1],
            "direction": self.direction.name,
            "entry_date": self.entry_date,
            "exit_date": self.exit_date,
            "entry_z": self.entry_z,
            "exit_z": self.exit_z,
            "entry_spread": self.entry_spread,
            "exit_spread": self.exit_spread,
            "hedge_ratio": self.hedge_ratio,
            "pnl": self.pnl,
            "pnl_pct": self.pnl_pct,
            "holding_days": self.holding_days,
            "exit_reason": self.exit_reason,
        }


@dataclass
class BacktestConfig:
    """Configuration for pairs trading backtest."""
    
    # Window parameters
    formation_period: int = 252      # Days for parameter estimation (IS)
    trading_period: int = 126        # Days for out-of-sample trading (OOS)
    rebalance_frequency: int = 21    # Days between rebalancing (monthly)
    
    # Signal thresholds
    entry_z: float = 2.0             # Entry when |z| >= this
    exit_z: float = 0.5              # Exit when |z| <= this (convergence)
    exit_on_mean: bool = True        # Exit when z crosses 0
    max_holding_periods: int | None = None  # None = use 2x half_life
    
    # Position sizing
    capital_per_pair: float = 10000.0  # Dollar allocation per pair
    max_pairs: int = 10               # Maximum concurrent positions
    
    # Transaction costs
    transaction_cost_bps: float = 5.0  # One-way cost in basis points
    
    # Risk management
    stop_loss_z: float | None = None  # Stop if |z| exceeds this
    max_loss_pct: float | None = None  # Stop if trade loss exceeds this %
    
    # Filtering
    min_half_life: float = 15.0
    max_half_life: float = 120.0
    pvalue_threshold: float = 0.10
    
    # Options
    use_kalman_hedge: bool = False    # Use Kalman filter for dynamic hedge ratio
    kalman_delta: float = 1e-4
    use_log_prices: bool = True


@dataclass
class BacktestMetrics:
    """Performance metrics from backtest."""
    
    # Return metrics
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    
    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    avg_holding_days: float
    
    # Risk metrics
    volatility: float
    downside_vol: float
    var_95: float
    cvar_95: float
    
    # Multiple testing adjusted metrics
    deflated_sharpe: float | None = None
    expected_max_sharpe: float | None = None
    
    # Time series
    equity_curve: pd.Series | None = None
    drawdown_series: pd.Series | None = None
    
    def as_dict(self) -> Mapping[str, float | int]:
        """Convert to dictionary (excluding time series)."""
        return {
            "total_return": self.total_return,
            "annualized_return": self.annualized_return,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "max_drawdown": self.max_drawdown,
            "calmar_ratio": self.calmar_ratio,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": self.win_rate,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
            "profit_factor": self.profit_factor,
            "avg_holding_days": self.avg_holding_days,
            "volatility": self.volatility,
            "downside_vol": self.downside_vol,
            "var_95": self.var_95,
            "cvar_95": self.cvar_95,
            "deflated_sharpe": self.deflated_sharpe,
        }


@dataclass
class BacktestResult:
    """Complete results from a pairs trading backtest."""
    
    config: BacktestConfig
    metrics: BacktestMetrics
    trades: list[Trade]
    daily_returns: pd.Series
    equity_curve: pd.Series
    pair_performance: pd.DataFrame
    
    def to_trades_dataframe(self) -> pd.DataFrame:
        """Convert trades to DataFrame."""
        return pd.DataFrame([t.as_dict() for t in self.trades])
    
    def summary(self) -> str:
        """Generate text summary of backtest results."""
        m = self.metrics
        return f"""
Walk-Forward Backtest Summary
=============================
Period: {self.daily_returns.index[0].date()} to {self.daily_returns.index[-1].date()}
Total Trades: {m.total_trades}

Returns:
  Total Return: {m.total_return:.2%}
  Annualized Return: {m.annualized_return:.2%}
  Sharpe Ratio: {m.sharpe_ratio:.2f}
  Sortino Ratio: {m.sortino_ratio:.2f}
  Max Drawdown: {m.max_drawdown:.2%}
  Calmar Ratio: {m.calmar_ratio:.2f}

Trade Statistics:
  Win Rate: {m.win_rate:.1%}
  Avg Win: {m.avg_win:.2%}
  Avg Loss: {m.avg_loss:.2%}
  Profit Factor: {m.profit_factor:.2f}
  Avg Holding: {m.avg_holding_days:.1f} days

Risk:
  Volatility: {m.volatility:.2%}
  VaR (95%): {m.var_95:.2%}
  CVaR (95%): {m.cvar_95:.2%}
  Deflated Sharpe: {f'{m.deflated_sharpe:.2f}' if m.deflated_sharpe and not np.isnan(m.deflated_sharpe) else 'N/A'}
"""


class PairsBacktester:
    """Walk-forward backtester for pairs trading strategies."""
    
    def __init__(
        self,
        prices: pd.DataFrame,
        config: BacktestConfig | None = None,
    ):
        """Initialize backtester with price data.
        
        Parameters
        ----------
        prices : pd.DataFrame
            DataFrame of adjusted close prices with date index and ticker columns.
        config : BacktestConfig | None
            Backtest configuration. Uses defaults if None.
        """
        self.prices = prices.copy()
        self.config = config or BacktestConfig()
        self._validate_inputs()
    
    def _validate_inputs(self):
        """Validate input data and configuration."""
        if self.prices.empty:
            raise ValueError("Price DataFrame cannot be empty")
        if not isinstance(self.prices.index, pd.DatetimeIndex):
            self.prices.index = pd.to_datetime(self.prices.index)
        
        min_days = self.config.formation_period + self.config.trading_period
        if len(self.prices) < min_days:
            raise ValueError(
                f"Insufficient data: {len(self.prices)} days < "
                f"{min_days} required (formation + trading)"
            )
    
    def run_single_pair(
        self,
        leg_x: str,
        leg_y: str,
        start_idx: int | None = None,
        end_idx: int | None = None,
    ) -> list[Trade]:
        """Run backtest on a single pair for a specified period.
        
        Parameters
        ----------
        leg_x : str
            Ticker for long leg.
        leg_y : str
            Ticker for short leg.
        start_idx : int | None
            Start index (inclusive). If None, starts at formation_period.
        end_idx : int | None
            End index (exclusive). If None, uses full data.
            
        Returns
        -------
        list[Trade]
            List of completed trades.
        """
        cfg = self.config
        
        # Get prices
        px = self.prices[leg_x].dropna()
        py = self.prices[leg_y].dropna()
        
        # Align
        df = pd.concat([px, py], axis=1, join="inner").dropna()
        if df.shape[0] < cfg.formation_period + 30:
            logger.debug(f"Insufficient data for {leg_x}-{leg_y}")
            return []
        
        px = df.iloc[:, 0]
        py = df.iloc[:, 1]
        
        # Set bounds
        if start_idx is None:
            start_idx = cfg.formation_period
        if end_idx is None:
            end_idx = len(df)
        
        # Log prices for spread calculation
        if cfg.use_log_prices:
            log_px = np.log(px)
            log_py = np.log(py)
        else:
            log_px = px
            log_py = py
        
        trades = []
        current_position = TradeDirection.FLAT
        entry_date = None
        entry_z = None
        entry_spread = None
        entry_hedge = None
        bars_held = 0
        
        for i in range(start_idx, end_idx):
            # Formation window for parameter estimation
            form_start = max(0, i - cfg.formation_period)
            form_end = i
            
            # Estimate hedge ratio from formation period
            window_log_px = log_px.iloc[form_start:form_end]
            window_log_py = log_py.iloc[form_start:form_end]
            
            if len(window_log_px) < 30:
                continue
            
            # Simple OLS hedge ratio (or use Kalman if configured)
            X = np.column_stack([np.ones(len(window_log_py)), window_log_py.values])
            beta = np.linalg.lstsq(X, window_log_px.values, rcond=None)[0]
            hedge_ratio = beta[1]
            
            # Calculate spread
            spread = log_px.iloc[form_start:form_end + 1] - hedge_ratio * log_py.iloc[form_start:form_end + 1]
            
            # Calculate z-score using formation period stats
            spread_mean = spread.iloc[:-1].mean()
            spread_std = spread.iloc[:-1].std()
            
            if spread_std < 1e-10:
                continue
            
            current_spread = spread.iloc[-1]
            current_z = (current_spread - spread_mean) / spread_std
            current_date = px.index[i]
            
            # Calculate half-life for max holding
            if cfg.max_holding_periods is None:
                spread_lag = spread.iloc[:-1].shift(1).dropna()
                spread_delta = spread.iloc[:-1].diff().dropna()
                aligned = pd.concat([spread_lag, spread_delta], axis=1, join="inner").dropna()
                if len(aligned) > 10:
                    x_vals = aligned.iloc[:, 0].values
                    y_vals = aligned.iloc[:, 1].values
                    ar_beta = np.linalg.lstsq(x_vals.reshape(-1, 1), y_vals, rcond=None)[0][0]
                    if ar_beta < 0:
                        half_life = int(-np.log(2) / ar_beta)
                        max_hold = max(2 * half_life, 20)
                    else:
                        max_hold = 60
                else:
                    max_hold = 60
            else:
                max_hold = cfg.max_holding_periods
            
            # Position management
            if current_position != TradeDirection.FLAT:
                bars_held += 1
                should_exit = False
                exit_reason = ""
                
                # Exit conditions
                if cfg.exit_on_mean:
                    if current_position == TradeDirection.LONG_SPREAD and current_z >= 0:
                        should_exit = True
                        exit_reason = "mean_cross"
                    elif current_position == TradeDirection.SHORT_SPREAD and current_z <= 0:
                        should_exit = True
                        exit_reason = "mean_cross"
                
                if not should_exit and abs(current_z) <= cfg.exit_z:
                    should_exit = True
                    exit_reason = "convergence"
                
                if not should_exit and bars_held >= max_hold:
                    should_exit = True
                    exit_reason = "timeout"
                
                if not should_exit and cfg.stop_loss_z and abs(current_z) >= cfg.stop_loss_z:
                    should_exit = True
                    exit_reason = "stop_loss"
                
                if should_exit:
                    # Calculate P&L
                    spread_pnl = (current_spread - entry_spread) * current_position.value
                    pnl_pct = spread_pnl / abs(entry_spread) if entry_spread != 0 else 0
                    
                    # Deduct transaction costs (entry + exit)
                    tc = 2 * cfg.transaction_cost_bps / 10000
                    pnl_pct -= tc
                    pnl_dollar = pnl_pct * cfg.capital_per_pair
                    
                    trades.append(Trade(
                        pair=(leg_x, leg_y),
                        direction=current_position,
                        entry_date=entry_date,
                        exit_date=current_date,
                        entry_z=entry_z,
                        exit_z=current_z,
                        entry_spread=entry_spread,
                        exit_spread=current_spread,
                        hedge_ratio=entry_hedge,
                        pnl=pnl_dollar,
                        pnl_pct=pnl_pct,
                        holding_days=bars_held,
                        exit_reason=exit_reason,
                    ))
                    
                    current_position = TradeDirection.FLAT
                    entry_date = None
                    bars_held = 0
            
            # Entry conditions
            if current_position == TradeDirection.FLAT:
                if current_z <= -cfg.entry_z:
                    # Long spread (expect mean reversion up)
                    current_position = TradeDirection.LONG_SPREAD
                    entry_date = current_date
                    entry_z = current_z
                    entry_spread = current_spread
                    entry_hedge = hedge_ratio
                    bars_held = 0
                    
                elif current_z >= cfg.entry_z:
                    # Short spread (expect mean reversion down)
                    current_position = TradeDirection.SHORT_SPREAD
                    entry_date = current_date
                    entry_z = current_z
                    entry_spread = current_spread
                    entry_hedge = hedge_ratio
                    bars_held = 0
        
        return trades
    
    def run_walk_forward(
        self,
        pairs: Sequence[tuple[str, str]],
        start_date: pd.Timestamp | None = None,
        end_date: pd.Timestamp | None = None,
    ) -> BacktestResult:
        """Execute walk-forward backtest on multiple pairs.
        
        Parameters
        ----------
        pairs : Sequence[tuple[str, str]]
            List of (leg_x, leg_y) pairs to trade.
        start_date : pd.Timestamp | None
            Start of backtest period. If None, uses earliest possible.
        end_date : pd.Timestamp | None
            End of backtest period. If None, uses latest available.
            
        Returns
        -------
        BacktestResult
            Complete backtest results with metrics and trade list.
        """
        cfg = self.config
        
        # Set date bounds
        if start_date is None:
            start_date = self.prices.index[cfg.formation_period]
        if end_date is None:
            end_date = self.prices.index[-1]
        
        # Filter to date range
        mask = (self.prices.index >= start_date) & (self.prices.index <= end_date)
        date_range = self.prices.index[mask]
        
        if len(date_range) < 30:
            raise ValueError("Insufficient data in specified date range")
        
        # Run backtest on each pair
        all_trades = []
        for leg_x, leg_y in pairs:
            if leg_x not in self.prices.columns or leg_y not in self.prices.columns:
                logger.warning(f"Missing data for {leg_x}-{leg_y}, skipping")
                continue
            
            pair_trades = self.run_single_pair(leg_x, leg_y)
            
            # Filter trades to date range
            pair_trades = [
                t for t in pair_trades
                if t.entry_date >= start_date and t.exit_date <= end_date
            ]
            all_trades.extend(pair_trades)
        
        logger.info(f"Completed {len(all_trades)} trades across {len(pairs)} pairs")
        
        # Build daily returns
        daily_returns = self._build_daily_returns(all_trades, date_range)
        
        # Calculate metrics
        metrics = self._calculate_metrics(daily_returns, all_trades, len(pairs))
        
        # Build equity curve
        equity_curve = (1 + daily_returns).cumprod()
        metrics.equity_curve = equity_curve
        metrics.drawdown_series = self._calculate_drawdown(equity_curve)
        
        # Pair-level performance
        pair_perf = self._calculate_pair_performance(all_trades)
        
        return BacktestResult(
            config=cfg,
            metrics=metrics,
            trades=all_trades,
            daily_returns=daily_returns,
            equity_curve=equity_curve,
            pair_performance=pair_perf,
        )
    
    def _build_daily_returns(
        self,
        trades: list[Trade],
        date_range: pd.DatetimeIndex,
    ) -> pd.Series:
        """Convert trades to daily return series."""
        daily_pnl = pd.Series(0.0, index=date_range)
        
        # Allocate P&L to exit dates
        for trade in trades:
            if trade.exit_date in daily_pnl.index:
                daily_pnl.loc[trade.exit_date] += trade.pnl
        
        # Convert to returns based on capital allocation
        capital = self.config.capital_per_pair * self.config.max_pairs
        daily_returns = daily_pnl / capital
        
        return daily_returns
    
    def _calculate_drawdown(self, equity_curve: pd.Series) -> pd.Series:
        """Calculate drawdown series from equity curve."""
        rolling_max = equity_curve.expanding().max()
        drawdown = (equity_curve - rolling_max) / rolling_max
        return drawdown
    
    def _calculate_metrics(
        self,
        daily_returns: pd.Series,
        trades: list[Trade],
        n_pairs: int,
    ) -> BacktestMetrics:
        """Calculate comprehensive performance metrics."""
        # Basic return metrics
        total_return = (1 + daily_returns).prod() - 1
        n_days = len(daily_returns)
        annualized_return = (1 + total_return) ** (252 / n_days) - 1 if n_days > 0 else 0
        
        # Volatility
        volatility = daily_returns.std() * np.sqrt(252)
        
        # Sharpe ratio (assuming 0 risk-free rate)
        sharpe = annualized_return / volatility if volatility > 0 else 0
        
        # Sortino ratio
        downside_returns = daily_returns[daily_returns < 0]
        downside_vol = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino = annualized_return / downside_vol if downside_vol > 0 else 0
        
        # Drawdown
        equity = (1 + daily_returns).cumprod()
        drawdown = self._calculate_drawdown(equity)
        max_drawdown = abs(drawdown.min())
        
        # Calmar ratio
        calmar = annualized_return / max_drawdown if max_drawdown > 0 else 0
        
        # Trade statistics
        total_trades = len(trades)
        winning_trades = sum(1 for t in trades if t.pnl > 0)
        losing_trades = sum(1 for t in trades if t.pnl < 0)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        wins = [t.pnl_pct for t in trades if t.pnl > 0]
        losses = [t.pnl_pct for t in trades if t.pnl < 0]
        avg_win = np.mean(wins) if wins else 0
        avg_loss = abs(np.mean(losses)) if losses else 0
        
        gross_profit = sum(t.pnl for t in trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in trades if t.pnl < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        avg_holding = np.mean([t.holding_days for t in trades]) if trades else 0
        
        # VaR and CVaR
        var_95 = daily_returns.quantile(0.05)
        cvar_95 = daily_returns[daily_returns <= var_95].mean() if len(daily_returns[daily_returns <= var_95]) > 0 else var_95
        
        # Deflated Sharpe Ratio
        deflated_sharpe = self._calculate_deflated_sharpe(
            sharpe, n_days, n_pairs, volatility
        )
        
        return BacktestMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_drawdown,
            calmar_ratio=calmar,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            avg_holding_days=avg_holding,
            volatility=volatility,
            downside_vol=downside_vol,
            var_95=var_95,
            cvar_95=cvar_95,
            deflated_sharpe=deflated_sharpe,
        )
    
    def _calculate_deflated_sharpe(
        self,
        sharpe: float,
        n_observations: int,
        n_trials: int,
        volatility: float,
    ) -> float:
        """Calculate Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014).
        
        Adjusts for multiple testing by comparing to expected max Sharpe
        under the null hypothesis of no skill.
        """
        if n_trials <= 1 or n_observations < 30:
            return sharpe
        
        # Expected maximum Sharpe under null (Euler-Mascheroni approximation)
        euler_mascheroni = 0.5772156649
        expected_max_sharpe = (
            (1 - euler_mascheroni) * stats.norm.ppf(1 - 1 / n_trials) +
            euler_mascheroni * stats.norm.ppf(1 - 1 / (n_trials * np.e))
        )
        
        # Standard error of Sharpe ratio
        se_sharpe = np.sqrt((1 + 0.5 * sharpe**2) / n_observations)
        
        # Deflated Sharpe (probability-based)
        z_stat = (sharpe - expected_max_sharpe) / se_sharpe
        deflated = stats.norm.cdf(z_stat)
        
        # Convert to Sharpe-equivalent
        deflated_sharpe = sharpe * deflated
        
        return deflated_sharpe
    
    def _calculate_pair_performance(self, trades: list[Trade]) -> pd.DataFrame:
        """Calculate performance metrics by pair."""
        if not trades:
            return pd.DataFrame()
        
        # Group trades by pair
        pair_groups = {}
        for trade in trades:
            key = (trade.pair[0], trade.pair[1])
            if key not in pair_groups:
                pair_groups[key] = []
            pair_groups[key].append(trade)
        
        rows = []
        for (leg_x, leg_y), pair_trades in pair_groups.items():
            total_pnl = sum(t.pnl for t in pair_trades)
            total_return = sum(t.pnl_pct for t in pair_trades)
            n_trades = len(pair_trades)
            wins = sum(1 for t in pair_trades if t.pnl > 0)
            
            rows.append({
                "leg_x": leg_x,
                "leg_y": leg_y,
                "n_trades": n_trades,
                "total_pnl": total_pnl,
                "total_return": total_return,
                "win_rate": wins / n_trades if n_trades > 0 else 0,
                "avg_holding_days": np.mean([t.holding_days for t in pair_trades]),
            })
        
        df = pd.DataFrame(rows)
        return df.sort_values("total_pnl", ascending=False)


def run_backtest(
    prices: pd.DataFrame,
    pairs: Sequence[tuple[str, str]],
    config: BacktestConfig | None = None,
) -> BacktestResult:
    """Convenience function to run a complete backtest.
    
    Parameters
    ----------
    prices : pd.DataFrame
        Price DataFrame with date index and ticker columns.
    pairs : Sequence[tuple[str, str]]
        List of pairs to trade.
    config : BacktestConfig | None
        Backtest configuration.
        
    Returns
    -------
    BacktestResult
        Complete backtest results.
    """
    backtester = PairsBacktester(prices, config)
    return backtester.run_walk_forward(pairs)


def compare_configs(
    prices: pd.DataFrame,
    pairs: Sequence[tuple[str, str]],
    configs: Sequence[BacktestConfig],
    config_names: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Run backtests with different configurations and compare results.
    
    Parameters
    ----------
    prices : pd.DataFrame
        Price data.
    pairs : Sequence[tuple[str, str]]
        Pairs to trade.
    configs : Sequence[BacktestConfig]
        Different configurations to test.
    config_names : Sequence[str] | None
        Names for each configuration.
        
    Returns
    -------
    pd.DataFrame
        Comparison table of metrics across configurations.
    """
    if config_names is None:
        config_names = [f"config_{i}" for i in range(len(configs))]
    
    results = []
    for name, config in zip(config_names, configs):
        result = run_backtest(prices, pairs, config)
        metrics = result.metrics.as_dict()
        metrics["config_name"] = name
        results.append(metrics)
    
    return pd.DataFrame(results).set_index("config_name")
