"""
Portfolio Analytics Module.

Defines the PortfolioAnalyser class for comparative performance analysis between 
realised portfolio results, control baselines, and Monte Carlo simulation paths. 
Calculates standard financial risk metrics (TWRR, Sharpe Ratio, Drawdown) and 
generates visualisation plots for reporting.
"""
from typing import Optional, Any, Tuple, Dict
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
    Performs comparative portfolio performance analysis and visualisation.

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
        
        # Intersect indices to ensure metrics cover identical time periods for all datasets.
        # prevents result skewing caused by missing start/end dates in a single series.
        common_idx = self.real.index.intersection(self.sims.index).intersection(self.control.index)
        
        self.real = self.real.loc[common_idx]
        self.control = self.control.loc[common_idx]
        self.sims = self.sims.loc[common_idx]

        # Align flows to the timeline (fills missing days with 0.0).
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
    def _get_adjusted_returns_and_index(self, nav_series: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """
        Calculates daily returns adjusted for external cash flows.

        Args:
            nav_series (pd.Series): Daily Net Asset Value series.

        Returns:
            Tuple[pd.Series, pd.Series]:
                - Adjusted daily returns.
                - Wealth index (cumulative product of returns starting at 1.0).
        """
        if nav_series.empty: 
            return pd.Series(dtype=float), pd.Series(dtype=float)

        # 1. Align flows: Subtract flow specifically for the correct day.
        # Handles cases where 'nav_series' is a subset of the full timeline.
        relevant_flows = self.flows.reindex(nav_series.index).fillna(0.0)

        # 2. Derive "Organic End Value".
        # Logic: Subtracts flow from ending NAV to isolate performance impact.
        organic_nav_end = nav_series - relevant_flows
        
        prev_nav = nav_series.shift(1)
        
        # 3. Calculate Daily Return.
        returns = (organic_nav_end / prev_nav) - 1.0
        returns = returns.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        
        # 4. Construct Wealth Index (Start at 1.0).
        wealth_index = (1 + returns).cumprod()
        wealth_index.iloc[0] = 1.0 
        
        return returns, wealth_index
    
    def _calculate_metrics(self, series: pd.Series) -> Dict[str, float]:
        """
        Calculates performance metrics with handling for sub-year durations.

        Args:
            series (pd.Series): Daily Net Asset Value series.

        Returns:
            Dict[str, float]: Dictionary containing TWRR, Volatility, Sharpe Ratio,
            and Max Drawdown.
        """
        if series.empty: return {}
        
        returns, wealth_index = self._get_adjusted_returns_and_index(series)
        
        days = (series.index[-1] - series.index[0]).days
        if days <= 0: return {}
        
        # 1. Period TWRR (Cumulative Return).
        period_twr = wealth_index.iloc[-1] / wealth_index.iloc[0] - 1
        
        # 2. Annualized Return (CAGR).
        ann_factor = 365 / days
        annualized_ret = (1 + period_twr) ** ann_factor - 1
        
        # 3. Volatility (Annualized).
        vol = returns.std() * np.sqrt(252)
        
        # 4. Geometric Sharpe Ratio.
        # Uses CAGR (Geometric Mean) rather than Arithmetic Mean.
        excess_ret = annualized_ret - self.rf
        sharpe = excess_ret / vol if vol > 1e-9 else 0.0
        
        # 5. Maximum Drawdown.
        roll_max = wealth_index.cummax()
        drawdown = (wealth_index - roll_max) / roll_max
        max_dd = drawdown.min()
        
        return {
            'Final NAV': series.iloc[-1],
            'Period TWRR': period_twr,
            'Annualised TWRR': annualized_ret,
            'Annualised Volatility': vol,
            'Geometric Sharpe Ratio': sharpe,
            'Maximum Drawdown': max_dd
        }
    
    def get_simulation_distributions(self) -> pd.DataFrame:
        """
        Calculates performance metrics for every individual simulation path.
        
        Returns:
            pd.DataFrame: DataFrame where rows correspond to simulation paths and
            columns correspond to performance metrics (Final NAV, Sharpe, etc.),
            formatted for plotting functions.
        """
        # Calculate metrics for all simulation paths
        sim_metrics = [self._calculate_metrics(self.sims[col]) for col in self.sims.columns]
        df_sim = pd.DataFrame(sim_metrics)
        
        return df_sim
    
    def get_statistics_summary(self, sim_stats: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Generates a comprehensive statistical summary comparing Real, Control, 
        and Simulated portfolios.
        
        Args:
            sim_stats (Optional[pd.DataFrame]): Pre-calculated simulation distributions. 
                Calculates automatically if None.

        Returns:
            pd.DataFrame: Summary table containing Medians and Percentile Ranks.
        """
        # 1. Retrieve Distribution Data.
        if sim_stats is None:
            df_sim = self.get_simulation_distributions()
        else:
            df_sim = sim_stats

        # 2. Calculate Single-Point Metrics for Real/Control.
        real_stats = self._calculate_metrics(self.real)
        control_stats = self._calculate_metrics(self.control)
        
        # 3. Map display names to data columns.
        metric_map = {
            "Final NAV": ("Final NAV", "Final NAV"),
            "Period TWRR": ("Period TWRR", "Period TWRR"),
            "Annualised TWRR": ("Annualised TWRR", "Annualised TWRR"),
            "Annualised Volatility": ("Annualised Volatility", "Annualised Volatility"),
            "Geometric Sharpe Ratio": ("Geometric Sharpe Ratio", "Geometric Sharpe Ratio"),
            "Maximum Drawdown": ("Maximum Drawdown", "Maximum Drawdown")
        }
        
        data = {}
        
        for row_name, (sim_col, single_col) in metric_map.items():
            # Extract data arrays.
            sim_dist = df_sim[sim_col].to_numpy(dtype=float)
            
            # Calculate Median.
            sim_median = np.median(sim_dist)
            
            # Calculate Percentiles.
            ctrl_val = control_stats[single_col]
            real_val = real_stats[single_col]

            # Calculate percentile rank (percentage of sims <= target).
            ctrl_pct = (sim_dist <= ctrl_val).mean() * 100
            real_pct = (sim_dist <= real_val).mean() * 100
            
            data[row_name] = {
                "Median Simulated Portfolio": sim_median,
                "Control Portfolio": ctrl_val,
                "Control Portfolio Percentile": ctrl_pct,
                "Real Portfolio": real_val,
                "Real Portfolio Percentile": real_pct
            }
            
        df = pd.DataFrame(data).T
        return df

    # ==========================================
    # VISUALISATION METHODS
    # ==========================================
    def plot_confidence_intervals(self, save_path: Optional[str] = None) -> None:
        """ 
        Generates plot comparing Real and Control portfolios against 
        Monte Carlo simulation confidence intervals.

        Visualizes 90% and 50% confidence intervals alongside the median 
        simulated path.

        Args:
            save_path (Optional[str]): File path to export the plot image.
        """
        plt.figure(figsize=(12, 6))

        total_n = self.sims.shape[1]

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

        # Add Dynamic N label
        plt.gca().text(0.98, 0.02, f'N = {total_n}', 
                       transform=plt.gca().transAxes, 
                       horizontalalignment='right', 
                       verticalalignment='bottom', 
                       fontsize=12,
                       fontweight='normal',
                       bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
        
        # plt.title('Monte Carlo Analysis: Confidence Intervals')
        plt.ylabel('Net Asset Value [CHF]')
        plt.legend(loc='upper left', framealpha=1.0, facecolor='white')
        plt.grid(True, alpha=0.3)
        plt.xlim(self.sims.index[0], self.sims.index[-1])
        plt.tight_layout()

        if save_path:
            print(f" [>] Saving plot to {save_path}")
            plt.savefig(save_path, dpi=600, bbox_inches='tight')

        plt.show()

    def plot_simulation_traces(self, num_paths: int = 150, save_path: Optional[str] = None) -> None:
        """
        Visualizes a subset of individual simulation traces to illustrate path variance.

        Args:
            num_paths (int): Maximum number of random paths to overlay. 
                Defaults to 150 to prevent overplotting.
            save_path (Optional[str]): File path to export the plot image.
        """
        plt.figure(figsize=(12, 6))

        total_n = self.sims.shape[1]
        
        cols_to_plot = self.sims.columns[:min(len(self.sims.columns), num_paths)]
        # Plot actual simulation data (no label to avoid duplicate entries)
        plt.plot(self.real, color='red', linewidth=2.5, label='Real Portfolio')
        plt.plot(self.control, color='blue', linewidth=2, label='Control Portfolio')
        plt.plot(self.sims[cols_to_plot], color='gray', linewidth=0.5, alpha=0.2)

        # Add a dummy line (empty data) just to create the single legend entry
        plt.plot([], [], color='gray', linewidth=0.5, label='Simulated Portfolios')

        # Add Dynamic N label
        plt.gca().text(0.98, 0.02, f'N = {total_n}', 
                       transform=plt.gca().transAxes, 
                       horizontalalignment='right', 
                       verticalalignment='bottom', 
                       fontsize=12,
                       fontweight='normal',
                       bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
        
        # plt.title(f'Monte Carlo Trace Analysis ({len(cols_to_plot)} paths)')
        plt.ylabel('Net Asset Value [CHF]')
        plt.legend(framealpha=1.0, facecolor='white')
        plt.grid(True, alpha=0.3)
        plt.xlim(self.sims.index[0], self.sims.index[-1])
        plt.tight_layout()

        if save_path:
            print(f" [>] Saving plot to {save_path}")
            plt.savefig(save_path, dpi=600, bbox_inches='tight')

        plt.show()

    def plot_drawdown_profile(self, save_path: Optional[str] = None) -> None:
        """ 
        Plots historical drawdown profile over time.

        Compares drawdown depth and recovery duration of Real and Control portfolios 
        against the median drawdown of simulated portfolios.

        Args:
            save_path (Optional[str]): File path to export the plot image.
        """
        plt.figure(figsize=(12, 4))

        total_n = self.sims.shape[1]

        # Calculates drawdown from wealth index
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

        # Add Dynamic N label
        loc_config = {'x': 0.02, 'y': 0.02, 'ha': 'left', 'va': 'bottom'}
        plt.gca().text(loc_config['x'], loc_config['y'], f'N = {total_n}', 
               transform=plt.gca().transAxes, 
               horizontalalignment=loc_config['ha'], 
               verticalalignment=loc_config['va'], 
               fontsize=12,
               fontweight='normal',
               bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

        plt.ylabel('Drawdown [%]')
        plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0, symbol=''))
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim(self.sims.index[0], self.sims.index[-1])
        plt.tight_layout()

        if save_path:
            print(f" [>] Saving plot to {save_path}")
            plt.savefig(save_path, dpi=600, bbox_inches='tight')

        plt.show()

    def plot_distributions_NAV(self, sim_raw_stats: pd.DataFrame, save_path: Optional[str] = None) -> None:
        """
        Generates histogram for Final Net Asset Value (NAV).

        Args:
            sim_raw_stats (pd.DataFrame): DataFrame containing 'Final NAV' column.
            save_path (Optional[str]): File path to export the plot image.
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        total_n = self.sims.shape[1]
        mean_val = sim_raw_stats['Final NAV'].mean()

        # Reduced target_ticks to 5 to prevent label overlap on large currency values
        bins, stride = self._get_dynamic_bins_and_stride(sim_raw_stats['Final NAV'], target_ticks=5)
        
        sns.histplot(data=sim_raw_stats, x='Final NAV', bins=bins, kde=True, ax=ax, color='skyblue')
        self._add_lines(ax, self.real.iloc[-1], self.control.iloc[-1], mean_val)

        # Add Dynamic N label
        loc_config = {'x': 0.02, 'y': 0.98, 'ha': 'left', 'va': 'top'}
        plt.gca().text(loc_config['x'], loc_config['y'], f'N = {total_n}', 
               transform=plt.gca().transAxes, 
               horizontalalignment=loc_config['ha'], 
               verticalalignment=loc_config['va'], 
               fontsize=12,
               fontweight='normal',
               bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
        
        ax.set_xlabel("Final Net Asset Value [CHF]")
        ax.set_ylabel("Count")
        
        # Apply the calculated stride
        ax.set_xticks(bins[::stride])
        ax.xaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))

        plt.tight_layout()
        if save_path:
            print(f" [>] Saving Final NAV plot to {save_path}")
            plt.savefig(save_path, dpi=600, bbox_inches='tight')
        plt.show()

    def plot_distributions_maxdd(self, sim_raw_stats: pd.DataFrame, save_path: Optional[str] = None) -> None:
        """
        Generates histogram for Maximum Drawdown.

        Args:
            sim_raw_stats (pd.DataFrame): DataFrame containing 'Maximum Drawdown' column.
            save_path (Optional[str]): File path to export the plot image.
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        total_n = self.sims.shape[1]
        real_stats = self._calculate_metrics(self.real)
        control_stats = self._calculate_metrics(self.control)
        mean_val = sim_raw_stats['Maximum Drawdown'].mean()

        bins, stride = self._get_dynamic_bins_and_stride(sim_raw_stats['Maximum Drawdown'])
        
        sns.histplot(data=sim_raw_stats, x='Maximum Drawdown', bins=bins, kde=True, ax=ax, color='salmon')
        self._add_lines(ax, real_stats['Maximum Drawdown'], control_stats['Maximum Drawdown'], mean_val)

        # Add Dynamic N label
        loc_config = {'x': 0.02, 'y': 0.98, 'ha': 'left', 'va': 'top'}
        plt.gca().text(loc_config['x'], loc_config['y'], f'N = {total_n}', 
               transform=plt.gca().transAxes, 
               horizontalalignment=loc_config['ha'], 
               verticalalignment=loc_config['va'], 
               fontsize=12,
               fontweight='normal',
               bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
        
        ax.set_xlabel("Drawdown [%]")
        ax.set_ylabel("Count")
        ax.set_xticks(bins[::stride])
        ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0, symbol=''))

        plt.tight_layout()
        if save_path:
            print(f" [>] Saving Maximum Drawdown plot to {save_path}")
            plt.savefig(save_path, dpi=600, bbox_inches='tight')
        plt.show()

    def plot_distributions_volatility(self, sim_raw_stats: pd.DataFrame, save_path: Optional[str] = None) -> None:
        """
        Generates histogram for Volatility.

        Args:
            sim_raw_stats (pd.DataFrame): DataFrame containing 'Annualised Volatility' column.
            save_path (Optional[str]): File path to export the plot image.
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        total_n = self.sims.shape[1]
        real_stats = self._calculate_metrics(self.real)
        control_stats = self._calculate_metrics(self.control)
        mean_val = sim_raw_stats['Annualised Volatility'].mean()

        bins, stride = self._get_dynamic_bins_and_stride(sim_raw_stats['Annualised Volatility'])
        
        sns.histplot(data=sim_raw_stats, x='Annualised Volatility', bins=bins, kde=True, ax=ax, color='lightgreen')
        self._add_lines(ax, real_stats['Annualised Volatility'], control_stats['Annualised Volatility'], mean_val)

        # Add Dynamic N label
        loc_config = {'x': 0.02, 'y': 0.98, 'ha': 'left', 'va': 'top'}
        plt.gca().text(loc_config['x'], loc_config['y'], f'N = {total_n}', 
               transform=plt.gca().transAxes, 
               horizontalalignment=loc_config['ha'], 
               verticalalignment=loc_config['va'], 
               fontsize=12,
               fontweight='normal',
               bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
        
        ax.set_xlabel("Annualised Volatility [%]")
        ax.set_ylabel("Count")
        ax.set_xticks(bins[::stride])
        ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0, symbol=''))

        plt.tight_layout()
        if save_path:
            print(f" [>] Saving Volatility plot to {save_path}")
            plt.savefig(save_path, dpi=600, bbox_inches='tight')
        plt.show()

    def plot_distributions_TWRR(self, sim_raw_stats: pd.DataFrame, save_path: Optional[str] = None) -> None:
        """
        Generates histogram for Period TWRR (Cumulative Return).

        Args:
            sim_raw_stats (pd.DataFrame): DataFrame containing 'Period TWRR' column.
            save_path (Optional[str]): File path to export the plot image.
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        total_n = self.sims.shape[1]
        
        # Calculate single-point metrics for Real and Control to place the vertical lines
        real_stats = self._calculate_metrics(self.real)
        control_stats = self._calculate_metrics(self.control)
        
        mean_val = sim_raw_stats['Period TWRR'].mean()
        bins, stride = self._get_dynamic_bins_and_stride(sim_raw_stats['Period TWRR'])
        
        sns.histplot(data=sim_raw_stats, x='Period TWRR', bins=bins, kde=True, ax=ax, color='orchid')
        
        self._add_lines(ax, real_stats['Period TWRR'], control_stats['Period TWRR'], mean_val)

        # Add Dynamic N label (Standard boilerplate)
        loc_config = {'x': 0.02, 'y': 0.98, 'ha': 'left', 'va': 'top'}
        plt.gca().text(loc_config['x'], loc_config['y'], f'N = {total_n}', 
               transform=plt.gca().transAxes, 
               horizontalalignment=loc_config['ha'], 
               verticalalignment=loc_config['va'], 
               fontsize=12,
               fontweight='normal',
               bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
        
        ax.set_xlabel("Period TWRR [%]") 
        ax.set_ylabel("Count")
        ax.set_xticks(bins[::stride])
        ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0, symbol=''))

        plt.tight_layout()
        if save_path:
            print(f" [>] Saving Period TWRR plot to {save_path}")
            plt.savefig(save_path, dpi=600, bbox_inches='tight')
        plt.show()
    
    def plot_distributions_sharpe(self, sim_raw_stats: pd.DataFrame, save_path: Optional[str] = None) -> None:
        """
        Generates histogram for Sharpe Ratio.

        Args:
            sim_raw_stats (pd.DataFrame): DataFrame containing 'Geometric Sharpe Ratio' column.
            save_path (Optional[str]): File path to export the plot image.
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        total_n = self.sims.shape[1]
        real_stats = self._calculate_metrics(self.real)
        control_stats = self._calculate_metrics(self.control)
        mean_val = sim_raw_stats['Geometric Sharpe Ratio'].mean()

        bins, stride = self._get_dynamic_bins_and_stride(sim_raw_stats['Geometric Sharpe Ratio'])
        
        sns.histplot(data=sim_raw_stats, x='Geometric Sharpe Ratio', bins=bins, kde=True, ax=ax, color='gold')
        self._add_lines(ax, real_stats['Geometric Sharpe Ratio'], control_stats['Geometric Sharpe Ratio'], mean_val)

        # Add Dynamic N label
        loc_config = {'x': 0.02, 'y': 0.98, 'ha': 'left', 'va': 'top'}
        plt.gca().text(loc_config['x'], loc_config['y'], f'N = {total_n}', 
               transform=plt.gca().transAxes, 
               horizontalalignment=loc_config['ha'], 
               verticalalignment=loc_config['va'], 
               fontsize=12,
               fontweight='normal',
               bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
        
        ax.set_xlabel("Geometric Sharpe Ratio")
        ax.set_ylabel("Count")
        ax.set_xticks(bins[::stride])
        ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))

        plt.tight_layout()
        if save_path:
            print(f" [>] Saving Sharpe Ratio plot to {save_path}")
            plt.savefig(save_path, dpi=600, bbox_inches='tight')
        plt.show()

    def _add_lines(self, ax: Axes, real_val: float, control_val: float, mean_val: float) -> None:
        """
        Adds vertical reference lines to histograms.

        Args:
            ax (Axes): Matplotlib Axes object.
            real_val (float): Value for the Real Portfolio.
            control_val (float): Value for the Control Portfolio.
            mean_val (float): Mean value of the simulation distribution.
        """
        ax.axvline(real_val, color='red', linestyle='-', linewidth=2, label='Real Portfolio')
        ax.axvline(control_val, color='blue', linestyle='--', linewidth=2, label='Control Portfolio')
        ax.axvline(mean_val, color='black', linestyle=':', linewidth=2, label='Distribution Mean') 
        ax.legend(loc='upper right', framealpha=1.0, facecolor='white')

    def _get_dynamic_bins_and_stride(self, data: pd.Series, target_bins: int = 40, target_ticks: int = 8) -> tuple[np.ndarray, int]:
        """
        Calculates optimal histogram bin edges and tick stride.
        Ensures ticks fall on integers or clean decimals (multiples of 1, 2, 5, 10).

        Args:
            data (pd.Series): The dataset to bin.
            target_bins (int): Desired number of histogram bins.
            target_ticks (int): Desired number of x-axis ticks.

        Returns:
            Tuple[np.ndarray, int]:
                - Array of bin edges.
                - Integer stride for tick labeling.
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