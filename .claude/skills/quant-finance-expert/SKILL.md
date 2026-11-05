---
name: quant-finance-expert
description: Expert in quantitative finance, market microstructure, trading strategies, and applied machine learning. Use this skill to design, refine, or validate trading models, analyze order flow, and implement advanced financial ML workflows.
---

# Quantitative Finance Expert

This skill provides specialized knowledge and workflows for building professional-grade trading systems, focusing on market microstructure, strategy design, and robust financial machine learning.

## Domain Expertise

### 1. Market Microstructure
- **Order Book Dynamics:** Limit Order Book (LOB) analysis, bid-ask spread components (adverse selection vs. inventory costs).
- **Order Flow:** Order Flow Imbalance (OFI), Trade Classification (Lee-Ready algorithm), and VPIN (Volume-Synchronized Probability of Informed Trading).
- **Liquidity & Impact:** Slippage estimation models and permanent/temporary market impact analysis.

### 2. Strategy Design & Alpha Research
- **Mean Reversion:** Cointegration tests (Engle-Granger, Johansen) and Ornstein-Uhlenbeck processes.
- **Momentum & Trend:** Time-series momentum vs. cross-sectional momentum.
- **Statistical Arbitrage:** Pairs trading, factor-based modeling, and PCA for dimensionality reduction in return space.

### 3. Applied Machine Learning for Finance
- **Data Labeling:** Beyond fixed-time horizons; use Triple Barrier Method and Meta-Labeling.
- **Feature Engineering:** Fractional differentiation (to preserve memory while achieving stationarity), volatility scaling, and entropy-based features.
- **Robust Validation:** Purged and Embargoed Cross-Validation to handle temporal correlation.

## Workflows

### Strategy Refinement
When evaluating a strategy's performance (like the CascadeStrategy results):
1. **Analyze Overfitting:** Compare Train vs. CV metrics. A high gap (e.g., Train R2 0.70 vs. CV R2 -0.14) indicates massive overfitting or data leakage.
2. **Examine Feature Importance:** Use SHAP or MDA (Mean Decrease Accuracy) to identify which features contribute to alpha.
3. **Verify Stationarity:** Ensure input features are stationary (ADF test) to avoid spurious regressions.

### Microstructure Analysis
1. Analyze the relationship between volume spikes and price volatility.
2. Use ATR-normalized distances to evaluate the significance of breakouts.

## Key Performance Metrics
- **Risk-Adjusted Returns:** Sharpe Ratio, Sortino Ratio, Information Ratio.
- **Execution Quality:** Profit Factor, Recovery Factor, Max Drawdown (Duration and Magnitude).
- **Stability:** Calmar Ratio and Hit Ratio (Win Rate) vs. Average Win/Loss.
