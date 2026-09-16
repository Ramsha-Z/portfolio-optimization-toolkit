# Mean-Variance Portfolio Optimization

A from-scratch implementation of Markowitz's mean-variance framework — the efficient frontier, the minimum-variance portfolio, and the maximum-Sharpe-ratio portfolio — applied to a 6-stock portfolio spanning technology, finance, consumer staples, energy, and healthcare.

This is independent, individually-built work: fresh market data, and every line of the optimization written and tested from scratch.

## What it shows

Using daily closing prices for **AAPL, JPM, KO, XOM, JNJ, MSFT** (2013–2017, sourced from public market data):

- Annualized return and volatility for each stock individually
- The **global minimum-variance portfolio** — the lowest-risk combination possible
- The **maximum-Sharpe-ratio portfolio** — the best risk-adjusted return
- The full **efficient frontier**, solved with constrained optimization (`scipy.optimize`, SLSQP)

**Key result:** the maximum-Sharpe portfolio achieves a better risk-adjusted return (Sharpe ≈ 1.26) than any single stock in the set — including MSFT, the best individual performer. That's the core insight mean-variance optimization is built to demonstrate: diversification improves the risk/return trade-off itself, not just the average outcome.

## Repo structure

```
portfolio-optimization-toolkit/
├── src/portfolio_optimization.py   # Core optimization functions, importable
├── notebooks/                      # Worked example with plots and commentary
├── data/                           # Market data used
├── tests/                          # Automated tests (pytest)
├── images/                         # Saved plots
└── requirements.txt
```

## Running it

```bash
pip install -r requirements.txt
jupyter notebook notebooks/02_portfolio_optimization.ipynb
```

Or run the tests directly:

```bash
pytest tests/
```

## Method

For a long-only, fully-invested portfolio with weights **w**, expected returns **μ**, and covariance matrix **Σ**:

- Portfolio return: `w'μ`
- Portfolio volatility: `sqrt(w'Σw)`
- Efficient frontier: minimize `w'Σw` subject to `w'μ = target_return`, `sum(w) = 1`, `w ≥ 0`, solved across a range of target returns
- Maximum Sharpe: maximize `(w'μ − r_f) / sqrt(w'Σw)` under the same constraints

## What I learned

- How to implement Markowitz mean-variance optimization from scratch
- Using `scipy.optimize` (SLSQP) to solve constrained portfolio problems
- Why diversification improves the risk/return trade-off itself, not just the average return
- How to structure a small quant project so it's testable and reproducible, not just a notebook

## Future improvements

- Add more asset classes (bonds, commodities) for broader diversification
- Include transaction costs in the optimization
- Add rolling-window backtesting to see how weights would have performed out-of-sample
- Extend to the Black-Litterman model to incorporate market views

## Author

Ramsha Zainab — MSc Economics and Finance, University of Navarra
[GitHub](https://github.com/Ramsha-Z) |  [LinkedIn](https://linkedin.com/in/ramsha-zainab)
