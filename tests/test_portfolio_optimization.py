"""
Sanity tests for the mean-variance portfolio optimization module.
"""

import sys
import os
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from portfolio_optimization import (
    daily_returns, annualize_returns, annualize_covariance,
    portfolio_performance, max_sharpe_portfolio, min_variance_portfolio,
)


@pytest.fixture
def market_data():
    path = os.path.join(os.path.dirname(__file__), "..", "data",
                         "portfolio_stock_prices_2013_2017.csv")
    prices = pd.read_csv(path, index_col="Date", parse_dates=True)
    returns = daily_returns(prices)
    return annualize_returns(returns), annualize_covariance(returns)


def test_portfolio_weights_sum_to_one(market_data):
    mean_returns, cov_matrix = market_data
    weights = max_sharpe_portfolio(mean_returns, cov_matrix)
    assert round(weights.sum(), 4) == 1.0
    assert (weights >= -1e-6).all()  # long-only


def test_min_variance_has_lowest_volatility(market_data):
    mean_returns, cov_matrix = market_data
    w_minvar = min_variance_portfolio(mean_returns, cov_matrix)
    _, v_minvar = portfolio_performance(w_minvar, mean_returns, cov_matrix)

    # Min-variance portfolio should have lower volatility than every
    # individual asset held alone.
    for i in range(len(mean_returns)):
        single_asset_weights = np.zeros(len(mean_returns))
        single_asset_weights[i] = 1.0
        _, v_single = portfolio_performance(single_asset_weights, mean_returns, cov_matrix)
        assert v_minvar <= v_single + 1e-9


def test_max_sharpe_beats_every_individual_stock(market_data):
    mean_returns, cov_matrix = market_data
    w_sharpe = max_sharpe_portfolio(mean_returns, cov_matrix, risk_free_rate=0.02)
    r_sharpe, v_sharpe = portfolio_performance(w_sharpe, mean_returns, cov_matrix)
    sharpe_ratio = (r_sharpe - 0.02) / v_sharpe

    for i, ticker in enumerate(mean_returns.index):
        r_single = mean_returns.iloc[i]
        v_single = np.sqrt(cov_matrix.iloc[i, i])
        single_sharpe = (r_single - 0.02) / v_single
        assert sharpe_ratio >= single_sharpe - 1e-9
