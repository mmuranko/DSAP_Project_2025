"""
Portfolio Analytics Module.

This module provides the PortfolioAnalyser class, which is responsible for comparative
performance analysis between realised portfolio results, control baselines, and 
Monte Carlo simulation paths. It handles the calculation of standard financial risk 
metrics (CAGR, Sharpe Ratio, Drawdown) and generates visualisation assets for reporting.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import seaborn as sns
import matplotlib.ticker as mtick
import time
from typing import Optional, Any
from .config import RISK_FREE_RATE

# Set the global font to be consistent across all plots
# 'font.serif': 'DejaVu Serif', 'Times New Roman', 'Georgia', 'Garamond'
# 'font.sans-serif': 'DejaVu Sans', 'Arial', 'Helvetica', 'Calibri'
# 'font.monospace': 'DejaVu Sans Mono', 'Courier New', 'Consolas'
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,                  # Base font size
    'axes.titlesize': 16,             # Title size
    'axes.titleweight': 'bold',       # Title weight
    'axes.labelsize': 14,             # X/Y label size
    'xtick.labelsize': 12,            # X-axis tick size
    'ytick.labelsize': 12,            # Y-axis tick size
    'legend.fontsize': 12,            # Legend text size
    'figure.titlesize': 18            # Super title size (if used)
})

class PortfolioAnalyser:
    """
    Encapsulates logic for comparative portfolio performance analysis and visualisation.
    
    This class aligns three distinct data series (Real, Control, and Simulated) to a common 
    timeline and generates statistical summaries and plots to assess skill versus luck.
    """

    def __init__(self, real_nav: pd.Series, control_nav: pd.Series, sim_paths_df: pd.DataFrame, flow_series: Optional[pd.Series] = None) -> None:
        """
        Initializes the analyser by aligning all input series to a shared date index.

        Args:
            real_nav (pd.Series): The Net Asset Value (NAV) series derived from actual 
                historical execution records (IBKR data).
            control_nav (pd.Series): The NAV series of the Control Portfolio, representing 
                the same historical trades processed through the simulation engine to 
                normalisefor fees and execution timing.
            sim_paths_df (pd.DataFrame): A DataFrame where each column represents a distinct 
                Monte Carlo simulation path (NAV) and the index represents dates.
        """
        self.real = real_nav
        self.control = control_nav
        self.sims = sim_paths_df
        self.rf = RISK_FREE_RATE
        
        # Data Alignment:
        # An intersection of indices is performed to ensure that metrics are calculated 
        # over the exact same time period for all datasets. This prevents skewing results 
        # due to missing start/end dates in any single series.
        common_idx = self.real.index.intersection(self.sims.index).intersection(self.control.index)
        
        self.real = self.real.loc[common_idx]
        self.control = self.control.loc[common_idx]
        self.sims = self.sims.loc[common_idx]

        # Align flows to the same timeline (fill missing days with 0)
        if flow_series is not None:
            # Ensure index is datetime and normalized
            if not isinstance(flow_series.index, pd.DatetimeIndex):
                flow_series.index = pd.to_datetime(flow_series.index)
            flow_series.index = flow_series.index.normalize()
            
            # Reindex to match the common simulation dates
            self.flows = flow_series.reindex(common_idx).fillna(0.0)
        else:
            self.flows = pd.Series(0.0, index=common_idx)
        
    # ==========================================
    # INTERNAL CALCULATIONS
    # ==========================================
    def _get_adjusted_returns_and_index(self, nav_series: pd.Series) -> tuple[pd.Series, pd.Series]:
        """
        Calculates daily returns adjusted for external cash flows (Deposits/Withdrawals).
        Returns both the daily return series and the cumulative Wealth Index (Unit Price).
        
        Formula: r_t = (NAV_t - Flow_t) / NAV_{t-1} - 1
        """
        if nav_series.empty: 
            return pd.Series(dtype=float), pd.Series(dtype=float)

        # Subtract today's flow from today's ending NAV to get "Organic End Value"
        # This assumes flows happen during the day or at end-of-day before measurement
        organic_nav_end = nav_series - self.flows
        
        # Previous day's NAV
        prev_nav = nav_series.shift(1)
        
        # Calculate Return
        # Handle division by zero for the very first day or empty accounts
        returns = (organic_nav_end / prev_nav) - 1.0
        returns = returns.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        
        # Construct Wealth Index (Start at 1.0)
        # We assume the first day (t=0) has 0 return or is the base
        wealth_index = (1 + returns).cumprod()
        wealth_index.iloc[0] = 1.0 # normalize start
        
        return returns, wealth_index
    
    def _calculate_metrics(self, series: pd.Series) -> dict[str, float]:
        """
        Calculates key financial performance indicators for a given NAV series.

        Metrics calculated:
            - CAGR: Compound Annual Growth Rate.
            - Volatility: Annualised standard deviation of daily returns.
            - Sharpe Ratio: Risk-adjusted return relative to the risk-free rate.
            - Maximum Drawdown: The largest peak-to-trough decline.

        Args:
            series (pd.Series): A time-series of Net Asset Values.

        Returns:
            Dict[str, float]: A dictionary containing the calculated metrics. Returns 
            an empty dict if the input series is empty or insufficient length.
        """
        if series.empty: return {}
        
        # Calculate daily percentage returns, dropping the first NaN value
        returns, wealth_index = self._get_adjusted_returns_and_index(series)
        
        # 1. CAGR (Geometric Mean)
        # Calculated using the total return over the period, annualised by day count.
        days = (series.index[-1] - series.index[0]).days
        if days <= 0: return {}
        
        total_ret = wealth_index.iloc[-1] / wealth_index.iloc[0] - 1
        cagr = (1 + total_ret) ** (365 / days) - 1
        
        # 2. Volatility (Annualised)
        # Standard deviation is scaled by sqrt(252) to approximate annual trading days.
        vol = returns.std() * np.sqrt(252)
        
        # 3. Sharpe Ratio (Annualised)
        # Measures excess return per unit of risk. A small epsilon (1e-9) is used 
        # to prevent division by zero in zero-volatility cases (e.g., cash only).
        excess_ret = cagr - self.rf
        sharpe = excess_ret / vol if vol > 1e-9 else 0.0
        
        # 4. Maximum Drawdown
        # Calculated by computing the rolling maximum and finding the minimum percentage 
        # deviation from that maximum.
        roll_max = wealth_index.cummax()
        drawdown = (wealth_index - roll_max) / roll_max
        max_dd = drawdown.min()
        
        return {
            'CAGR': cagr,
            'Volatility': vol,
            'Sharpe': sharpe,
            'Max_DD': max_dd,
            'Final_NAV': series.iloc[-1] # Keep raw NAV for display
        }

    def get_summary_table(self) -> tuple[pd.DataFrame, pd.DataFrame, object | None]:
        """
        Aggregates performance metrics for the Real, Control, and Simulated portfolios.

        For the simulated portfolios, metrics are calculated individually for each path 
        first, and then averaged. This provides the "Expected Value" of the strategy 
        under random asset selection.

        Returns:
            Tuple containing:
                - summary_df (pd.DataFrame): Comparison table of averaged metrics.
                - raw_sim_stats (pd.DataFrame): The raw metrics for every individual simulation path.
                - styled_df (Styler object or None): A formatted version of the summary table 
                  for display in Jupyter/HTML environments (returns None if dependency missing).
        """
        stats_real = self._calculate_metrics(self.real)
        stats_control = self._calculate_metrics(self.control)
        
        # Metrics are computed per path to preserve the distribution of outcomes.
        # Averaging NAVs first and then calculating metrics would underestimate volatility.
        sim_metrics = []
        for col in self.sims.columns:
            sim_metrics.append(self._calculate_metrics(self.sims[col]))
        
        df_sim_stats = pd.DataFrame(sim_metrics)
        stats_sim_avg = df_sim_stats.mean().to_dict()
        
        # Construct Summary DataFrame
        df = pd.DataFrame([stats_real, stats_control, stats_sim_avg], 
                          index=['Real Portfolio', 'Control Portfolio', 'Simulated (Avg)'])
        
        # Define display formatting
        format_map: dict[str, Any] = {
            'CAGR': '{:.2%}',
            'Volatility': '{:.2%}',
            'Sharpe': '{:.2f}',
            'Max_DD': '{:.2%}',
            'Final_NAV': '{:,.0f}'
        }
        
        # Attempt to apply pandas styling
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
        Generates a plot comparing the Real and Control portfolios against the 
        confidence intervals derived from the Monte Carlo simulation.

        The plot displays:
        - 90% Confidence Interval (5th-95th percentile)
        - 50% Confidence Interval (25th-75th percentile)
        - Median Simulated Path
        """
        plt.figure(figsize=(12, 6))
        
        # Calculate Percentiles across simulation paths (axis=1)
        median = self.sims.median(axis=1)
        p95 = self.sims.quantile(0.95, axis=1)
        p05 = self.sims.quantile(0.05, axis=1)
        p75 = self.sims.quantile(0.75, axis=1)
        p25 = self.sims.quantile(0.25, axis=1)
        
        # Render confidence bands
        plt.fill_between(self.sims.index, p05, p95, color='gray', alpha=0.15, label='90% Confidence Interval')
        plt.fill_between(self.sims.index, p25, p75, color='gray', alpha=0.30, label='50% Confidence Interval')
        plt.plot(median, color='gray', linestyle='--', alpha=0.6, label='Median Simulated Portfolio')
        
        # Render control and actual portfolio
        plt.plot(self.control, color='blue', linewidth=1.5, linestyle='-.', label='Control Portfolio')
        plt.plot(self.real, color='red', linewidth=2, label='Real Portfolio')
        
        plt.title('Monte Carlo Analysis: Confidence Intervals')
        plt.ylabel('Net Asset Value [CHF]')
        plt.legend(loc='upper left')
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
            num_paths (int): The maximum number of random paths to overlay. 
                             Defaults to 150 to prevent overplotting/performance issues.
        """
        plt.figure(figsize=(12, 6))
        
        # Select a subset of columns to maintain plot readability
        cols_to_plot = self.sims.columns[:min(len(self.sims.columns), num_paths)]
        plt.plot(self.sims[cols_to_plot], color='gray', linewidth=0.5, alpha=0.2)
        
        # Highlight real and control portfolio
        plt.plot(self.control, color='blue', linewidth=2, label='Control Portfolio')
        plt.plot(self.real, color='red', linewidth=2.5, label='Real Portfolio')
        
        plt.title(f'Monte Carlo Trace Analysis ({len(cols_to_plot)} paths)')
        plt.ylabel('Net Asset Value [CHF]')
        plt.legend()
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
        Plots the historical drawdown profile over time.

        This visualisation compares the drawdown depth and recovery duration of the 
        Real and Control portfolios against the median drawdown experience of the 
        simulated universe.
        """
        plt.figure(figsize=(12, 4))
        
        # Helper lambda to get drawdown from Wealth Index
        def _get_dd(s):
            _, wi = self._get_adjusted_returns_and_index(s)
            roll = wi.cummax()
            return (wi - roll) / roll

        dd_real = _get_dd(self.real)
        dd_control = _get_dd(self.control)
        
        # For simulation, we compute the median NAV path first, then its wealth index
        # (Approximation: Median of paths vs Path of Medians. Path of Medians is safer here)
        median_sim_nav = self.sims.median(axis=1)
        dd_sim = _get_dd(median_sim_nav)
        
        # Plot Median Simulated Portfolio
        plt.plot(dd_sim, color='gray', linestyle=':', linewidth=1.5, label='Median Simulated Portfolio Drawdown')
        plt.fill_between(dd_sim.index, dd_sim, 0, color='gray', alpha=0.1)

        # Plot Real and Control Portfolio
        plt.plot(dd_real, color='red', linewidth=2, label='Real Portfolio Drawdown')
        plt.plot(dd_control, color='blue', linestyle='--', linewidth=1.5, label='Control Portfolio Drawdown')
        plt.fill_between(dd_real.index, dd_real, 0, color='red', alpha=0.1)
        
        plt.title('Portfolio Drawdown Profile')
        plt.ylabel('Drawdown [%]')
        plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
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

        # Calculate metrics
        mean_val = sim_raw_stats['Final_NAV'].mean()

        # Distribution
        bins, stride = self._get_dynamic_bins_and_stride(sim_raw_stats['Final_NAV'])
        
        sns.histplot(data=sim_raw_stats, x='Final_NAV', bins=bins, kde=True, ax=ax, color='skyblue')
        self._add_lines(ax, self.real.iloc[-1], self.control.iloc[-1], mean_val)
        
        # ax.set_title('Final NAV Distribution')
        ax.set_xlabel("Net Asset Value [CHF]")
        ax.set_ylabel("Count")
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
        
        # Calculate metrics
        real_stats = self._calculate_metrics(self.real)
        control_stats = self._calculate_metrics(self.control)
        mean_val = sim_raw_stats['Max_DD'].mean()

        # Distribution
        bins, stride = self._get_dynamic_bins_and_stride(sim_raw_stats['Max_DD'])
        
        sns.histplot(data=sim_raw_stats, x='Max_DD', bins=bins, kde=True, ax=ax, color='salmon')
        self._add_lines(ax, real_stats['Max_DD'], control_stats['Max_DD'], mean_val)
        
        # ax.set_title('Maximum Drawdown Distribution')
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

        # Calculate metrics
        real_stats = self._calculate_metrics(self.real)
        control_stats = self._calculate_metrics(self.control)
        mean_val = sim_raw_stats['Volatility'].mean()

        # Distribution
        bins, stride = self._get_dynamic_bins_and_stride(sim_raw_stats['Volatility'])
        
        sns.histplot(data=sim_raw_stats, x='Volatility', bins=bins, kde=True, ax=ax, color='lightgreen')
        self._add_lines(ax, real_stats['Volatility'], control_stats['Volatility'], mean_val)
        
        # ax.set_title('Volatility Distribution')
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

        # Calculate metrics
        real_stats = self._calculate_metrics(self.real)
        control_stats = self._calculate_metrics(self.control)
        mean_val = sim_raw_stats['CAGR'].mean()

        # Distribution
        bins, stride = self._get_dynamic_bins_and_stride(sim_raw_stats['CAGR'])
        
        sns.histplot(data=sim_raw_stats, x='CAGR', bins=bins, kde=True, ax=ax, color='orchid')
        self._add_lines(ax, real_stats['CAGR'], control_stats['CAGR'], mean_val)
        
        # ax.set_title('CAGR Distribution')
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
        
        # Calculate metrics
        real_stats = self._calculate_metrics(self.real)
        control_stats = self._calculate_metrics(self.control)
        mean_val = sim_raw_stats['Sharpe'].mean()

        # Distribution
        bins, stride = self._get_dynamic_bins_and_stride(sim_raw_stats['Sharpe'])
        
        sns.histplot(data=sim_raw_stats, x='Sharpe', bins=bins, kde=True, ax=ax, color='gold')
        self._add_lines(ax, real_stats['Sharpe'], control_stats['Sharpe'], mean_val)
        
        # ax.set_title('Sharpe Ratio Distribution')
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
        """Helper to add vertical reference lines to histograms."""
        ax.axvline(real_val, color='red', linestyle='-', linewidth=2, label='Real')
        ax.axvline(control_val, color='blue', linestyle='--', linewidth=2, label='Control')
        ax.axvline(mean_val, color='black', linestyle=':', linewidth=2, label='Average') 
        ax.legend(loc='upper right', framealpha=1.0, facecolor='white')

    def _get_dynamic_bins_and_stride(self, data: pd.Series, target_bins: int = 40, target_ticks: int = 8) -> tuple[np.ndarray, int]:
        """
        Calculates optimal histogram bin edges and a tick stride for x-axis labelling.
        Dynamically adapts to the data range using the "1-2-5" rule.
        """
        val_range = data.max() - data.min()

        # Handle edge cases where variance is zero or negligible
        if val_range <= 0: val_range = abs(data.mean()) * 0.1 
        if val_range == 0: val_range = 1.0
            
        # 1. Estimate raw step size required to achieve the target bin count
        raw_step = val_range / target_bins
            
        # 2. Snap to "Nice" Human Numbers (1, 2, 2.5, 5, 10)
        magnitude = 10 ** np.floor(np.log10(raw_step))
        residual = raw_step / magnitude
            
        if residual <= 1.0: nice_step = 1.0 * magnitude
        elif residual <= 2.0: nice_step = 2.0 * magnitude
        elif residual <= 2.5: nice_step = 2.5 * magnitude
        elif residual <= 5.0: nice_step = 5.0 * magnitude
        else: nice_step = 10.0 * magnitude
            
        # 3. Generate Bins aligned to the "nice" step (Inlined Logic)
        if nice_step <= 0: 
            bins = np.array([data.min(), data.max()])
        else:
            lower = np.floor(data.min() / nice_step) * nice_step
            upper = np.ceil(data.max() / nice_step) * nice_step
            bins = np.arange(lower, upper + nice_step + (nice_step/1000), nice_step)
            
        # 4. Calculate Stride to maintain clean x-axis visualisation
        total_bins = len(bins)
        stride = max(1, total_bins // target_ticks)
            
        return bins, stride