import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mtick

class PortfolioAnalyzer:
    def __init__(self, real_nav, benchmark_nav, sim_paths_df, risk_free_rate=0.02):
        """
        Initializes the Analyzer with the three key datasets.
        
        :param real_nav: Series (The faithful reconstruction from IBKR data)
        :param benchmark_nav: Series (The 'Shadow' simulation of actual trades)
        :param sim_paths_df: DataFrame (Columns = Path IDs, Index = Dates)
        :param risk_free_rate: Float (Annualized risk-free rate, e.g. 0.02 for 2%)
        """
        self.real = real_nav
        self.benchmark = benchmark_nav
        self.sims = sim_paths_df
        self.rf = risk_free_rate
        
        # 1. Alignment: Ensure all series share the exact same dates
        common_idx = self.real.index.intersection(self.sims.index).intersection(self.benchmark.index)
        
        self.real = self.real.loc[common_idx]
        self.benchmark = self.benchmark.loc[common_idx]
        self.sims = self.sims.loc[common_idx]
        
    # ==========================================
    # INTERNAL CALCULATIONS
    # ==========================================
    def _calculate_metrics(self, series):
        """ Calculates CAGR, Volatility, Sharpe, and MaxDD for a single price series. """
        if series.empty: return {}
        
        # Daily Returns
        returns = series.pct_change().dropna()
        
        # 1. CAGR (Geometric Mean)
        days = (series.index[-1] - series.index[0]).days
        if days <= 0: return {}
        
        total_ret = series.iloc[-1] / series.iloc[0] - 1
        cagr = (1 + total_ret) ** (365 / days) - 1
        
        # 2. Volatility (Annualized)
        vol = returns.std() * np.sqrt(252)
        
        # 3. Sharpe Ratio (Annualized)
        excess_ret = cagr - self.rf
        sharpe = excess_ret / vol if vol > 1e-9 else 0.0
        
        # 4. Max Drawdown
        roll_max = series.cummax()
        drawdown = (series - roll_max) / roll_max
        max_dd = drawdown.min()
        
        return {
            'CAGR': cagr,
            'Volatility': vol,
            'Sharpe': sharpe,
            'Max_DD': max_dd,
            'Final_NAV': series.iloc[-1]
        }

    def get_summary_table(self):
        """ Returns a DataFrame comparing Real, Benchmark, and Sim Averages. """
        stats_real = self._calculate_metrics(self.real)
        stats_bench = self._calculate_metrics(self.benchmark)
        
        # For simulations, we calculate metrics for EVERY path then average them
        sim_metrics = []
        for col in self.sims.columns:
            sim_metrics.append(self._calculate_metrics(self.sims[col]))
        
        df_sim_stats = pd.DataFrame(sim_metrics)
        stats_sim_avg = df_sim_stats.mean().to_dict()
        
        # Create Summary DataFrame
        df = pd.DataFrame([stats_real, stats_bench, stats_sim_avg], 
                          index=['Real Portfolio', 'Shadow Benchmark', 'Simulated (Avg)'])
        
        # Format for readability (Dictionary only)
        format_map = {
            'CAGR': '{:.2%}',
            'Volatility': '{:.2%}',
            'Sharpe': '{:.2f}',
            'Max_DD': '{:.2%}',
            'Final_NAV': '{:,.0f}'
        }
        
        # Robust Styling Logic
        try:
            df_styled = df.style.format(format_map)
        except (ImportError, AttributeError):
            df_styled = None 
        
        return df, df_sim_stats, df_styled

    def _get_drawdown_series(self, series):
        roll_max = series.cummax()
        return (series - roll_max) / roll_max

    # ==========================================
    # VISUALIZATION METHODS
    # ==========================================
    def plot_confidence_intervals(self):
        """ Plots Real vs Benchmark vs 5th-95th Percentile Confidence Intervals. """
        plt.figure(figsize=(12, 6))
        
        # Calculate Percentiles across the rows (axis=1)
        median = self.sims.median(axis=1)
        p95 = self.sims.quantile(0.95, axis=1)
        p05 = self.sims.quantile(0.05, axis=1)
        p75 = self.sims.quantile(0.75, axis=1)
        p25 = self.sims.quantile(0.25, axis=1)
        
        # Plot Envelopes
        plt.fill_between(self.sims.index, p05, p95, color='gray', alpha=0.15, label='90% Confidence Interval')
        plt.fill_between(self.sims.index, p25, p75, color='gray', alpha=0.30, label='50% Confidence Interval')
        plt.plot(median, color='gray', linestyle='--', alpha=0.6, label='Median Simulation')
        
        # Plot Actuals
        plt.plot(self.benchmark, color='blue', linewidth=1.5, linestyle='-.', label='Shadow Benchmark')
        plt.plot(self.real, color='red', linewidth=2, label='Real Portfolio')
        
        plt.title('Monte Carlo Analysis: Confidence Intervals')
        plt.ylabel('NAV (Base Currency)')
        plt.legend(loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_simulation_traces(self, num_paths=150):
        """ Visualizes individual simulation traces to illustrate path variance. """
        plt.figure(figsize=(12, 6))
        
        # Plot subset of sims
        cols_to_plot = self.sims.columns[:min(len(self.sims.columns), num_paths)]
        plt.plot(self.sims[cols_to_plot], color='gray', linewidth=0.5, alpha=0.2)
        
        # Highlight Real/Benchmark
        plt.plot(self.benchmark, color='blue', linewidth=2, label='Shadow Benchmark')
        plt.plot(self.real, color='red', linewidth=2.5, label='Real Portfolio')
        
        plt.title(f'Monte Carlo Trace Analysis ({len(cols_to_plot)} paths)')
        plt.ylabel('NAV')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    def plot_distributions(self, sim_raw_stats):
        """ 
        Plots histograms for Final NAV, Max Drawdown, and Volatility.
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Helper to plot vertical lines
        def add_lines(ax, real_val, bench_val):
            ax.axvline(real_val, color='red', linestyle='-', linewidth=2, label='Real')
            ax.axvline(bench_val, color='blue', linestyle='--', linewidth=2, label='Benchmark')
            ax.legend()

        # A. Final NAV
        sns.histplot(sim_raw_stats['Final_NAV'], kde=True, ax=axes[0], color='skyblue')
        add_lines(axes[0], self.real.iloc[-1], self.benchmark.iloc[-1])
        axes[0].set_title('Distribution of Final NAV')

        # B. Max Drawdown (Risk)
        sns.histplot(sim_raw_stats['Max_DD'], kde=True, ax=axes[1], color='salmon')
        real_stats = self._calculate_metrics(self.real)
        bench_stats = self._calculate_metrics(self.benchmark)
        add_lines(axes[1], real_stats['Max_DD'], bench_stats['Max_DD'])
        axes[1].set_title('Distribution of Max Drawdowns')
        axes[1].xaxis.set_major_formatter(mtick.PercentFormatter(1.0))

        # C. Volatility (Variance)
        sns.histplot(sim_raw_stats['Volatility'], kde=True, ax=axes[2], color='lightgreen')
        add_lines(axes[2], real_stats['Volatility'], bench_stats['Volatility'])
        axes[2].set_title('Distribution of Volatility')
        axes[2].xaxis.set_major_formatter(mtick.PercentFormatter(1.0))

        plt.tight_layout()
        plt.show()

    def plot_drawdown_profile(self):
        """ Plots the Drawdown Profile comparing Real, Benchmark, and Median Simulation. """
        plt.figure(figsize=(12, 4))
        
        dd_real = self._get_drawdown_series(self.real)
        dd_bench = self._get_drawdown_series(self.benchmark)
        
        # Calculate Median Simulation Drawdown
        # We calculate the median NAV series first, then its drawdown. 
        # This represents the "Typical Path" experience.
        median_sim_nav = self.sims.median(axis=1)
        dd_sim = self._get_drawdown_series(median_sim_nav)
        
        # Plot Median Sim (Baseline)
        plt.plot(dd_sim, color='gray', linestyle=':', linewidth=1.5, label='Median Sim Drawdown')
        plt.fill_between(dd_sim.index, dd_sim, 0, color='gray', alpha=0.1)

        # Plot Benchmark
        plt.plot(dd_bench, color='blue', linestyle='--', linewidth=1.5, label='Shadow Drawdown')
        
        # Plot Real (Highlighted)
        plt.plot(dd_real, color='red', linewidth=2, label='Real Drawdown')
        plt.fill_between(dd_real.index, dd_real, 0, color='red', alpha=0.1)
        
        plt.title('Portfolio Drawdown Profile')
        plt.ylabel('Drawdown %')
        plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()