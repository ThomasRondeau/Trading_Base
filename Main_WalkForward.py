import logging

import pandas as pd

from Connections.DataManager import DataManager
from Engines.WalkForwardOptimizer import WalkForwardOptimizer
from Strategies.TrendFollowingStrategy import TrendFollowingStrategy


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # ~55 jours de barres 2 minutes (limite intraday de yfinance : 60 jours)
    data = DataManager().collect_data_yfinance("AAPL", 55, 0)

    optimizer = WalkForwardOptimizer(
        strategy_factory=TrendFollowingStrategy,
        param_grid={"sma_fast": [5, 10, 20], "sma_slow": [30, 50, 100]},
        train_size=3000,
        validation_size=1000,
        test_size=1000,
        embargo=30,
        objective="total_pnl_net_percent",
        min_trades=5,
    )
    result = optimizer.run(data)

    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(result.summary())
    print("\nStabilité des paramètres retenus :")
    print(result.param_stability())
    print(f"\nScore OOS moyen ({result.objective}) : {result.mean_test_score:.4f}")


if __name__ == "__main__":
    main()
