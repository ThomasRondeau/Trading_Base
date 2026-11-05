"""Candlestick chart of one IBKR session with VWAP bands and the top support / resistance lines."""

import logging
from typing import Any

import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import pandas as pd

from quantlab.execution.ibkr.client import IBKRClient, stock
from quantlab.features.levels import resistance_lines, support_lines
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


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    client = IBKRClient(auto_reconnect=False)
    try:
        client.connect()
    except ConnectionError as exc:
        logger.error("%s", exc)
        return

    try:
        data = client.historical_bars(stock("SPY"), end="20250711 20:00:00 US/Eastern", use_rth=True)
        if data.empty:
            logger.warning("No data received for SPY")
            return
        logger.info("SPY data received: %d bars", len(data))

        vwap = VWAP().compute(data, {})
        plots: list[Any] = [mpf.make_addplot(vwap["vwap"], color="orange", width=2, label="VWAP")]
        for column, style, label in (
            ("vwap_upper_3", "--", "Upper Band"),
            ("vwap_far_upper_3", ":", "Far Upper Band"),
            ("vwap_lower_3", "--", "Lower Band"),
            ("vwap_far_lower_3", ":", "Far Lower Band"),
        ):
            plots.append(mpf.make_addplot(vwap[column], color="green", linestyle=style, label=label))

        for lines, color, name in (
            (resistance_lines(data), "red", "Resistance"),
            (support_lines(data), "blue", "Support"),
        ):
            for rank, (_, line) in enumerate(lines.head(N_LINES).iterrows(), start=1):
                plots.append(
                    mpf.make_addplot(
                        _line_series(data.index, line), color=color, width=1.5, linestyle="-", label=f"{name} {rank}"
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
        client.disconnect()


if __name__ == "__main__":
    main()
