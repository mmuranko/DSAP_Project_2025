"""
Portfolio Analytics Module.

Provides the PortfolioAnalyser class for comparative performance analysis between 
realised portfolio results, control baselines, and Monte Carlo simulation paths. 
Calculates standard financial risk metrics (CAGR, Sharpe Ratio, Drawdown) and 
generates visualisation assets for reporting.
"""
import time
from typing import Optional, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import seaborn as sns
import matplotlib.ticker as mtick
from .config import RISK_FREE_RATE

# Global plot configuration
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 13,
    'axes.titlesize': 16,
    'axes.titleweight': 'bold',
    'axes.labelsize': 14,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'legend.fontsize': 13,
    'figure.titlesize': 18
})

class PortfolioAnalyser:
    """
    Comparative portfolio performance analysis and visualisation.

    Aligns three distinct data series (Real, Control, and Simulated) to a common 
    timeline and generates statistical summaries and plots to assess algorithmic 
    skill versus random chance.
    """

    def __init__(self, real_nav: pd.Series, control_nav: pd.Series, sim_paths_df: pd.DataFrame, flow_series: Optional[pd.Series] = None) -> None:
        """
        Initializes the analyser and aligns all input series to a shared date index.

        Args:
            real_nav (pd.Series): Net Asset Value (NAV) series derived from actual 
                historical execution records.
            control_nav (pd.Series): NAV series of the Control Portfolio, representing 
                historical trades processed through the simulation engine to 
                normalise for fees and execution timing.
            sim_paths_df (pd.DataFrame): DataFrame where each column represents a distinct 
                Monte Carlo simulation path (NAV) and the index represents dates.
            flow_series (Optional[pd.Series]): External cash flows (deposits/withdrawals)
                used for return adjustments.
        """
        self.real = real_nav
        self.control = control_nav
        self.sims = sim_paths_df
        self.rf = RISK_FREE_RATE
        
        # Intersects indices to ensure metrics cover identical time periods for all datasets.
        # Prevents result skewing caused by missing start/end dates in a single series.
        common_idx = self.real.index.intersection(self.sims.index).intersection(self.control.index)
        
        self.real = self.real.loc[common_idx]
        self.control = self.control.loc[common_idx]
        self.sims = self.sims.loc[common_idx]

        # Aligns flows to the timeline (fills missing days with 0.0)
        if flow_series is not None:
            if not isinstance(flow_series.index, pd.DatetimeIndex):
                flow_series.index = pd.to_datetime(flow_series.index)
            flow_series.index = flow_series.index.normalize()
            
            self.flows = flow_series.reindex(common_idx).fillna(0.0)
        else:
            self.flows = pd.Series(0.0, index=common_idx)
        
    # ==========================================
    # INTERNAL CALCULATIONS
    # ==========================================
    def _get_adjusted_returns_and_index(self, nav_series: pd.Series) -> tuple[pd.Series, pd.Series]:
        """
        Calculates daily returns adjusted for external cash flows (Deposits/Withdrawals).
        
        Returns:
            tuple[pd.Series, pd.Series]: Daily return series and cumulative Wealth Index.
        
        Formula: r_t = (NAV_t - Flow_t) / NAV_{t-1} - 1
        """
        if nav_series.empty: 
            return pd.Series(dtype=float), pd.Series(dtype=float)

        # Subtracts flow from ending NAV to derive "Organic End Value".
        # Assumes flows occur before measurement or calculation.
        organic_nav_end = nav_series - self.flows
        
        prev_nav = nav_series.shift(1)
        
        # Calculates Return.
        # Handles division by zero for the initial day or empty accounts.
        returns = (organic_nav_end / prev_nav) - 1.0
        returns = returns.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        
        # Constructs Wealth Index (Start at 1.0).
        # Normalizes t=0 to 1.0.
        wealth_index = (1 + returns).cumprod()
        wealth_index.iloc[0] = 1.0 
        
        return returns, wealth_index
    
    def _calculate_metrics(self, series: pd.Series) -> dict[str, float]:
        """
        Calculates key financial performance indicators for a NAV series.

        Metrics:
            - CAGR: Compound Annual Growth Rate.
            - Volatility: Annualised standard deviation of daily returns.
            - Sharpe Ratio: Risk-adjusted return relative to the risk-free rate.
            - Maximum Drawdown: Largest peak-to-trough decline.

        Args:
            series (pd.Series): Time-series of Net Asset Values.

        Returns:
            dict[str, float]: Calculated metrics. Returns empty dict if series 
            is empty or length is insufficient.
        """
        if series.empty: return {}
        
        # Calculates daily percentage returns.
        returns, wealth_index = self._get_adjusted_returns_and_index(series)
        
        # 1. CAGR (Geometric Mean)
        # Uses total return over the period, annualised by day count.
        days = (series.index[-1] - series.index[0]).days
        if days <= 0: return {}
        
        total_ret = wealth_index.iloc[-1] / wealth_index.iloc[0] - 1
        cagr = (1 + total_ret) ** (365 / days) - 1
        
        # 2. Volatility (Annualised)
        # Scales standard deviation by sqrt(252) to approximate annual trading days.
        vol = returns.std() * np.sqrt(252)
        
        # 3. Sharpe Ratio (Annualised)
        # Uses epsilon (1e-9) to prevent division by zero in zero-volatility cases.
        excess_ret = cagr - self.rf
        sharpe = excess_ret / vol if vol > 1e-9 else 0.0
        
        # 4. Maximum Drawdown
        # Computes rolling maximum to find minimum percentage deviation.
        roll_max = wealth_index.cummax()
        drawdown = (wealth_index - roll_max) / roll_max
        max_dd = drawdown.min()
        
        return {
            'CAGR': cagr,
            'Volatility': vol,
            'Sharpe': sharpe,
            'Max_DD': max_dd,
            'Final_NAV': series.iloc[-1] # Raw NAV for display
        }

    def get_summary_table(self) -> tuple[pd.DataFrame, pd.DataFrame, object | None]:
        """
        Aggregates performance metrics for Real, Control, and Simulated portfolios.

        Calculates metrics individually for each simulation path before averaging to 
        derive the "Expected Value" of the strategy under random asset selection.

        Returns:
            tuple: 
                - summary_df (pd.DataFrame): Comparison table of averaged metrics.
                - raw_sim_stats (pd.DataFrame): Raw metrics for every simulation path.
                - styled_df (Styler | None): Formatted summary table (None if dependency missing).
        """
        stats_real = self._calculate_metrics(self.real)
        stats_control = self._calculate_metrics(self.control)
        
        # Computes metrics per path to preserve outcome distribution.
        # Averaging NAVs prior to metric calculation underestimates volatility.
        sim_metrics = []
        for col in self.sims.columns:
            sim_metrics.append(self._calculate_metrics(self.sims[col]))
        
        df_sim_stats = pd.DataFrame(sim_metrics)
        stats_sim_avg = df_sim_stats.mean().to_dict()
        
        # Constructs Summary DataFrame
        df = pd.DataFrame([stats_real, stats_control, stats_sim_avg], 
                          index=['Real Portfolio', 'Control Portfolio', 'Simulated (Avg)'])
        
        format_map: dict[str, Any] = {
            'CAGR': '{:.2%}',
            'Volatility': '{:.2%}',
            'Sharpe': '{:.2f}',
            'Max_DD': '{:.2%}',
            'Final_NAV': '{:,.0f}'
        }
        
        # Applies pandas styling if available
        try:
            df_styled = df.style.format(format_map)
        except (ImportError, AttributeError):
            df_styled = None 
        
        return df, df_sim_stats, df_styled

    # ==========================================
    # VISUALISATION METHODS
    # ==========================================

    def plot_confidence_intervals(self, save_path: Optional[str] = None) -> None:
        """ 
        Generates plot comparing Real and Control portfolios against 
        Monte Carlo simulation confidence intervals.

        Displays:
        - 90% Confidence Interval (5th-95th percentile)
        - 50% Confidence Interval (25th-75th percentile)
        - Median Simulated Path
        """
        plt.figure(figsize=(12, 6))
        
        # Calculates Percentiles across simulation paths (axis=1)
        median = self.sims.median(axis=1)
        p95 = self.sims.quantile(0.95, axis=1)
        p05 = self.sims.quantile(0.05, axis=1)
        p75 = self.sims.quantile(0.75, axis=1)
        p25 = self.sims.quantile(0.25, axis=1)
        
        # Renders control and actual portfolio
        plt.plot(self.real, color='red', linewidth=2, label='Real Portfolio')
        plt.plot(self.control, color='blue', linewidth=1.5, linestyle='-.', label='Control Portfolio')

        # Renders confidence bands
        plt.fill_between(self.sims.index, p05, p95, color='gray', alpha=0.15, label='90% Confidence Interval')
        plt.fill_between(self.sims.index, p25, p75, color='gray', alpha=0.30, label='50% Confidence Interval')
        plt.plot(median, color='gray', linestyle='--', alpha=0.6, label='Median Simulated Portfolio')
        
        # plt.title('Monte Carlo Analysis: Confidence Intervals')
        plt.ylabel('Net Asset Value [CHF]')
        plt.legend(loc='upper left', framealpha=1.0, facecolor='white')
        plt.grid(True, alpha=0.3)
        plt.xlim(self.sims.index[0], self.sims.index[-1])
        plt.tight_layout()

        if save_path:
            print(f" [>] Saving plot to {save_path}")
            time.sleep(0.5)
            plt.savefig(save_path, dpi=600, bbox_inches='tight')

        plt.show()


    def plot_simulation_traces(self, num_paths: int = 150, save_path: Optional[str] = None) -> None:
        """
        Visualizes a subset of individual simulation traces to illustrate path variance.

        Args:
            num_paths (int): Maximum number of random paths to overlay. 
                Defaults to 150 to prevent overplotting.
        """
        plt.figure(figsize=(12, 6))
        
        cols_to_plot = self.sims.columns[:min(len(self.sims.columns), num_paths)]
        plt.plot(self.sims[cols_to_plot], color='gray', linewidth=0.5, alpha=0.2)

        plt.plot(self.real, color='red', linewidth=2.5, label='Real Portfolio')
        plt.plot(self.control, color='blue', linewidth=2, label='Control Portfolio')
        
        # plt.title(f'Monte Carlo Trace Analysis ({len(cols_to_plot)} paths)')
        plt.ylabel('Net Asset Value [CHF]')
        plt.legend(framealpha=1.0, facecolor='white')
        plt.grid(True, alpha=0.3)
        plt.xlim(self.sims.index[0], self.sims.index[-1])
        plt.tight_layout()

        if save_path:
            print(f" [>] Saving plot to {save_path}")
            time.sleep(0.5)
            plt.savefig(save_path, dpi=600, bbox_inches='tight')

        plt.show()


    def plot_drawdown_profile(self, save_path: Optional[str] = None) -> None:
        """ 
        Plots historical drawdown profile over time.

        Compares drawdown depth and recovery duration of Real and Control portfolios 
        against the median drawdown of simulated portfolios.
        """
        plt.figure(figsize=(12, 4))
        
        # Calculates drawdown from Wealth Index
        def _get_dd(s):
            _, wi = self._get_adjusted_returns_and_index(s)
            roll = wi.cummax()
            return (wi - roll) / roll

        dd_real = _get_dd(self.real)
        dd_control = _get_dd(self.control)
        
        # Computes median NAV path first, then derives wealth index
        median_sim_nav = self.sims.median(axis=1)
        dd_sim = _get_dd(median_sim_nav)

        plt.plot(dd_real, color='red', linewidth=2, label='Real Portfolio')
        plt.plot(dd_control, color='blue', linestyle='--', linewidth=1.5, label='Control Portfolio')
        plt.fill_between(dd_real.index, dd_real, 0, color='red', alpha=0.1)
        
        plt.plot(dd_sim, color='gray', linestyle=':', linewidth=1.5, label='Median Simulated Portfolio')
        plt.fill_between(dd_sim.index, dd_sim, 0, color='gray', alpha=0.1)

        plt.title('Portfolio Drawdown Profile')
        plt.ylabel('Drawdown [%]')
        plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0, symbol=''))
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim(self.sims.index[0], self.sims.index[-1])
        plt.tight_layout()

        if save_path:
            print(f" [>] Saving plot to {save_path}")
            time.sleep(0.5)
            plt.savefig(save_path, dpi=600, bbox_inches='tight')

        plt.show()

    def plot_distributions_NAV(self, sim_raw_stats: pd.DataFrame, save_path: Optional[str] = None) -> None:
        """Generates histogram for Final Net Asset Value (NAV)."""
        fig, ax = plt.subplots(figsize=(10, 6))

        mean_val = sim_raw_stats['Final_NAV'].mean()

        # Reduced target_ticks to 5 to prevent label overlap on large currency values
        bins, stride = self._get_dynamic_bins_and_stride(sim_raw_stats['Final_NAV'], target_ticks=5)
        
        sns.histplot(data=sim_raw_stats, x='Final_NAV', bins=bins, kde=True, ax=ax, color='skyblue')
        self._add_lines(ax, self.real.iloc[-1], self.control.iloc[-1], mean_val)
        
        ax.set_xlabel("Net Asset Value [CHF]")
        ax.set_ylabel("Count")
        
        # Apply the calculated stride
        ax.set_xticks(bins[::stride])
        ax.xaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))

        plt.tight_layout()
        if save_path:
            print(f" [>] Saving NAV plot to {save_path}")
            plt.savefig(save_path, dpi=600, bbox_inches='tight')
        plt.show()

    def plot_distributions_maxdd(self, sim_raw_stats: pd.DataFrame, save_path: Optional[str] = None) -> None:
        """Generates histogram for Maximum Drawdown."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        real_stats = self._calculate_metrics(self.real)
        control_stats = self._calculate_metrics(self.control)
        mean_val = sim_raw_stats['Max_DD'].mean()

        bins, stride = self._get_dynamic_bins_and_stride(sim_raw_stats['Max_DD'])
        
        sns.histplot(data=sim_raw_stats, x='Max_DD', bins=bins, kde=True, ax=ax, color='salmon')
        self._add_lines(ax, real_stats['Max_DD'], control_stats['Max_DD'], mean_val)
        
        ax.set_xlabel("Drawdown [%]")
        ax.set_ylabel("Count")
        ax.set_xticks(bins[::stride])
        ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0, symbol=''))

        plt.tight_layout()
        if save_path:
            print(f" [>] Saving MaxDD plot to {save_path}")
            plt.savefig(save_path, dpi=600, bbox_inches='tight')
        plt.show()

    def plot_distributions_volatility(self, sim_raw_stats: pd.DataFrame, save_path: Optional[str] = None) -> None:
        """Generates histogram for Volatility."""
        fig, ax = plt.subplots(figsize=(10, 6))

        real_stats = self._calculate_metrics(self.real)
        control_stats = self._calculate_metrics(self.control)
        mean_val = sim_raw_stats['Volatility'].mean()

        bins, stride = self._get_dynamic_bins_and_stride(sim_raw_stats['Volatility'])
        
        sns.histplot(data=sim_raw_stats, x='Volatility', bins=bins, kde=True, ax=ax, color='lightgreen')
        self._add_lines(ax, real_stats['Volatility'], control_stats['Volatility'], mean_val)
        
        ax.set_xlabel("Annualised Volatility [%]")
        ax.set_ylabel("Count")
        ax.set_xticks(bins[::stride])
        ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0, symbol=''))

        plt.tight_layout()
        if save_path:
            print(f" [>] Saving Volatility plot to {save_path}")
            plt.savefig(save_path, dpi=600, bbox_inches='tight')
        plt.show()

    def plot_distributions_CAGR(self, sim_raw_stats: pd.DataFrame, save_path: Optional[str] = None) -> None:
        """Generates histogram for CAGR."""
        fig, ax = plt.subplots(figsize=(10, 6))

        real_stats = self._calculate_metrics(self.real)
        control_stats = self._calculate_metrics(self.control)
        mean_val = sim_raw_stats['CAGR'].mean()

        bins, stride = self._get_dynamic_bins_and_stride(sim_raw_stats['CAGR'])
        
        sns.histplot(data=sim_raw_stats, x='CAGR', bins=bins, kde=True, ax=ax, color='orchid')
        self._add_lines(ax, real_stats['CAGR'], control_stats['CAGR'], mean_val)
        
        ax.set_xlabel("Compound Annual Growth Rate [%]")
        ax.set_ylabel("Count")
        ax.set_xticks(bins[::stride])
        ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0, symbol=''))

        plt.tight_layout()
        if save_path:
            print(f" [>] Saving CAGR plot to {save_path}")
            plt.savefig(save_path, dpi=600, bbox_inches='tight')
        plt.show()
    
    def plot_distributions_sharpe(self, sim_raw_stats: pd.DataFrame, save_path: Optional[str] = None) -> None:
        """Generates histogram for Sharpe Ratio."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        real_stats = self._calculate_metrics(self.real)
        control_stats = self._calculate_metrics(self.control)
        mean_val = sim_raw_stats['Sharpe'].mean()

        bins, stride = self._get_dynamic_bins_and_stride(sim_raw_stats['Sharpe'])
        
        sns.histplot(data=sim_raw_stats, x='Sharpe', bins=bins, kde=True, ax=ax, color='gold')
        self._add_lines(ax, real_stats['Sharpe'], control_stats['Sharpe'], mean_val)
        
        ax.set_xlabel("Sharpe Ratio")
        ax.set_ylabel("Count")
        ax.set_xticks(bins[::stride])
        ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))

        plt.tight_layout()
        if save_path:
            print(f" [>] Saving Sharpe plot to {save_path}")
            plt.savefig(save_path, dpi=600, bbox_inches='tight')
        plt.show()

    def _add_lines(self, ax: Axes, real_val: float, control_val: float, mean_val: float) -> None:
        """Adds vertical reference lines to histograms."""
        ax.axvline(real_val, color='red', linestyle='-', linewidth=2, label='Real Portfolio')
        ax.axvline(control_val, color='blue', linestyle='--', linewidth=2, label='Control Portfolio')
        ax.axvline(mean_val, color='black', linestyle=':', linewidth=2, label='Distribution Mean') 
        ax.legend(loc='upper right', framealpha=1.0, facecolor='white')

    def _get_dynamic_bins_and_stride(self, data: pd.Series, target_bins: int = 40, target_ticks: int = 8) -> tuple[np.ndarray, int]:
        """
        Calculates optimal histogram bin edges and tick stride.
        Ensures ticks fall on nice integers (multiples of 1, 2, 5, 10).
        """
        val_range = data.max() - data.min()
        if val_range <= 0: 
            val_range = abs(data.mean()) * 0.1 if abs(data.mean()) > 1e-9 else 1.0

        # 1. Determine "Nice" Tick Interval
        raw_tick_step = val_range / target_ticks
        magnitude = 10 ** np.floor(np.log10(raw_tick_step))
        residual = raw_tick_step / magnitude
        
        if residual <= 1.5: nice_mult = 1.0
        elif residual <= 3.5: nice_mult = 2.0
        elif residual <= 7.5: nice_mult = 5.0
        else: nice_mult = 10.0
        
        tick_step = nice_mult * magnitude
        
        # 2. Determine Stride (Bins per Tick) to match target_bins count
        approx_stride = tick_step / (val_range / target_bins)
        valid_strides = np.array([1, 2, 4, 5, 10, 20])
        stride = int(valid_strides[np.argmin(np.abs(valid_strides - approx_stride))])
        
        # 3. Generate Bins aligned to the Tick Grid
        bin_step = tick_step / stride
        lower_bound = np.floor(data.min() / tick_step) * tick_step
        upper_bound = np.ceil(data.max() / tick_step) * tick_step
        
        bins = np.arange(lower_bound, upper_bound + (bin_step * 0.5), bin_step)
        
        # Ensure max value is covered
        if bins[-1] < data.max():
             bins = np.concatenate([bins, [bins[-1] + bin_step]])
            
        return bins, stride