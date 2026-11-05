"""Candlestick chart of one IBKR session with VWAP bands and the top support / resistance lines."""

import logging
import threading
import time
from typing import Any

import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import pandas as pd

from quantlab.data.schema import validate_ohlcv
from quantlab.execution.ibkr.trading_app import TradingApp
from quantlab.features.levels import Resistance, Support
from quantlab.features.technical import VWAP

logger = logging.getLogger(__name__)

N_LINES = 3  # support / resistance lines drawn


def _line_series(index: pd.Index, line: pd.Series) -> pd.Series:
    """Series that is NaN everywhere except on the bars covered by a support/resistance line."""
    series = pd.Series(index=index, data=np.nan)
    for idx in range(line["idxStart"], min(line["idxEnd"] + 1, len(index))):
        if idx in line["points"]:
            series.iloc[idx] = line["points"][idx]
    return series


def _connect(app: TradingApp, timeout: float = 10) -> bool:
    app.connect("127.0.0.1", 7497, clientId=5)
    threading.Thread(target=app.run, daemon=True).start()
    start_time = time.time()
    while not app.connected and (time.time() - start_time) < timeout:
        logger.info("Waiting for the TWS/IB Gateway connection...")
        time.sleep(1)
    return bool(app.connected)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    app = TradingApp.get_instance()
    if not _connect(app):
        logger.error(
            "Cannot connect to TWS/IB Gateway. Check that TWS or IB Gateway is running, the API is enabled "
            "(Configure > API > Enable ActiveX and Socket Clients), the port is 7497 (TWS) or 4002 (Gateway) "
            "and client id 5 is free."
        )
        return

    try:
        spy = TradingApp.get_contract("SPY")
        data = validate_ohlcv(app.get_historical_data(99, spy, end_date="20250711 20:00:00 US/Eastern"))
        if data.empty:
            logger.warning("No data received for SPY")
            return
        logger.info("SPY data received: %d bars", len(data))

        vwap, support, resistance = VWAP(data), Support(data), Resistance(data)
        for indicator in (vwap, support, resistance):
            indicator.calculate()

        plots: list[Any] = [
            mpf.make_addplot(vwap.point_of_control, color="orange", width=2, label="VWAP"),
            mpf.make_addplot(vwap.superior_limit, color="green", linestyle="--", label="Upper Band"),
            mpf.make_addplot(vwap.far_superior_limit, color="green", linestyle=":", label="Far Upper Band"),
            mpf.make_addplot(vwap.inferior_limit, color="green", linestyle="--", label="Lower Band"),
            mpf.make_addplot(vwap.far_inferior_limit, color="green", linestyle=":", label="Far Lower Band"),
        ]
        for lines, color, name in (
            (resistance.resistances, "red", "Resistance"),
            (support.supports, "blue", "Support"),
        ):
            if isinstance(lines, pd.DataFrame):
                for rank, (_, line) in enumerate(lines.head(N_LINES).iterrows(), start=1):
                    plots.append(
                        mpf.make_addplot(
                            _line_series(data.index, line),
                            color=color,
                            width=1.5,
                            linestyle="-",
                            label=f"{name} {rank}",
                        )
                    )

        chart_data = data.rename(columns=str.title)  # mplfinance expects Open/High/Low/Close/Volume
        _, axes = mpf.plot(
            chart_data,
            type="candle",
            style="yahoo",
            figsize=(12, 8),
            title="SPY",
            ylabel="Price ($)",
            addplot=plots,
            volume=True,
            show_nontrading=False,
            tight_layout=True,
            returnfig=True,
        )
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            axes[0].legend(handles, labels, loc="upper left", fontsize=8)
        plt.show()

    except Exception:
        logger.exception("Chart failed")
    finally:
        if app.connected:
            app.disconnect()
            logger.info("Disconnected")


if __name__ == "__main__":
    main()
