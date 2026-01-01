# Monte Carlo Portfolio Reconstruction

---

## Original Project Proposal

The proposed project aims to develop a flexible framework for reconstructing, simulating, and analysing investment portfolios using account data exported from Interactive Brokers (IBKR). The programme will be designed to handle any IBKR data imports, allowing users to upload their own CSV files and automatically generate a historic portfolio reconstruction. All main transactions, including trades, dividend payments, and margin interest payments, will be processed while maintaining currency separation until the final valuation in Swiss francs (CHF).

A key component of the project is the implementation of a Monte Carlo simulation to explore alternative portfolio evolutions: For each historical trade, the simulation will randomly replace the traded asset with another asset from the actual portfolio universe, maintaining equivalent trade values adjusted for currency exchange rates. Repeating this process across thousands of simulated paths will produce a distribution of potential portfolio outcomes in terms of return and volatility. This enables a direct comparison between realised portfolio performance and the range of possible alternative trajectories.

The analysis will address practical complexities such as exchange rate conversions, transaction fees, dividend payments, margin interest across multiple currencies. The final output will include statistical summaries and visualisations of the actual historical performance compared to the simulated outcome distributions.

As a potential extension, the random replacement mechanism of the Monte Carlo simulation could be substituted with an optimisation-based approach: In this variant, replacement trades would be selected to maximise the portfolio’s instantaneous Sharpe ratio at each transaction point. This method would rely on the covariance matrix and expected returns of all assets within the portfolio universe, derived from daily historic returns of all assets up to that point in time. Comparing this Sharpe ratio-optimised scenario to both the realised and randomised simulations would provide deeper insight into the efficiency and opportunity cost of past investment decisions.

---

**Author:** Marvin Muranko  
**Student ID:** 21955182  
**Course:** Advanced Programming 2025  
**Date:** October 13, 2025