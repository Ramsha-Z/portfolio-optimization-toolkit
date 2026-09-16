"""
Mean-Variance Portfolio Optimization (Markowitz)
==================================================
Build the efficient frontier, find the minimum-variance and
maximum-Sharpe-ratio portfolios, and solve for a portfolio that hits a
target expected return with minimum risk.

Data: 6 large-cap US stocks across different sectors (AAPL, JPM, KO,
XOM, JNJ, MSFT), 2013-2017 daily closing prices, sourced from public
market data.

Author: Ramsha Zainab
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize


TRADING_DAYS_PER_YEAR = 252


def daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple daily percentage returns from a wide price DataFrame."""
    return prices.pct_change().dropna()


def annualize_returns(returns: pd.DataFrame) -> pd.Series:
    """Annualized mean return for each asset."""
    return returns.mean() * TRADING_DAYS_PER_YEAR


def annualize_covariance(returns: pd.DataFrame) -> pd.DataFrame:
    """Annualized covariance matrix."""
    return returns.cov() * TRADING_DAYS_PER_YEAR


def portfolio_performance(weights: np.ndarray, mean_returns: pd.Series, cov_matrix: pd.DataFrame):
    """Expected annual return and volatility for a given set of weights."""
    port_return = float(np.dot(weights, mean_returns))
    port_vol = float(np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))))
    return port_return, port_vol


def negative_sharpe(weights, mean_returns, cov_matrix, risk_free_rate):
    port_return, port_vol = portfolio_performance(weights, mean_returns, cov_matrix)
    return -(port_return - risk_free_rate) / port_vol


def portfolio_volatility(weights, mean_returns, cov_matrix):
    return portfolio_performance(weights, mean_returns, cov_matrix)[1]


def _long_only_constraints(n_assets):
    bounds = tuple((0.0, 1.0) for _ in range(n_assets))
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    return bounds, constraints


def max_sharpe_portfolio(mean_returns, cov_matrix, risk_free_rate=0.02):
    """Find the long-only portfolio that maximizes the Sharpe ratio."""
    n = len(mean_returns)
    bounds, constraints = _long_only_constraints(n)
    init_guess = np.repeat(1 / n, n)

    result = minimize(
        negative_sharpe,
        init_guess,
        args=(mean_returns, cov_matrix, risk_free_rate),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    return result.x


def min_variance_portfolio(mean_returns, cov_matrix):
    """Find the long-only global minimum-variance portfolio."""
    n = len(mean_returns)
    bounds, constraints = _long_only_constraints(n)
    init_guess = np.repeat(1 / n, n)

    result = minimize(
        portfolio_volatility,
        init_guess,
        args=(mean_returns, cov_matrix),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    return result.x


def efficient_portfolio_for_target_return(mean_returns, cov_matrix, target_return):
    """Minimum-variance long-only portfolio that achieves a target return."""
    n = len(mean_returns)
    bounds, base_constraints = _long_only_constraints(n)
    constraints = base_constraints + [
        {"type": "eq", "fun": lambda w: np.dot(w, mean_returns) - target_return}
    ]
    init_guess = np.repeat(1 / n, n)

    result = minimize(
        portfolio_volatility,
        init_guess,
        args=(mean_returns, cov_matrix),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    return result.x if result.success else None


def efficient_frontier(mean_returns, cov_matrix, n_points=50):
    """
    Trace the efficient frontier by solving the minimum-variance problem
    for a range of target returns.

    Returns a DataFrame with columns: target_return, volatility, weights.
    """
    target_returns = np.linspace(mean_returns.min(), mean_returns.max(), n_points)
    records = []
    for target in target_returns:
        weights = efficient_portfolio_for_target_return(mean_returns, cov_matrix, target)
        if weights is None:
            continue
        _, vol = portfolio_performance(weights, mean_returns, cov_matrix)
        records.append({"target_return": target, "volatility": vol, "weights": weights})
    return pd.DataFrame(records)
