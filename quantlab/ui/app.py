"""
quantlab graphical interface (Streamlit): everything the CLI does, plus charts.

    py -3.10 -m quantlab ui            (or: py -3.10 -m streamlit run quantlab/ui/app.py)

Pages: Backtest, Optimization (walk-forward), Live session (shadow / paper / live on IBKR),
Sessions (journal monitoring and replay), Chart (VWAP + support / resistance).
The pages call the same library functions as the CLI; the labels are in French.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import io
import logging
import sys
import tempfile
from datetime import date
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:  # `streamlit run` puts only this file's folder on the path
    sys.path.insert(0, str(_ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from quantlab.cli import parse_value  # noqa: E402
from quantlab.config import ACCOUNTS, IBKRSettings  # noqa: E402
from quantlab.data import loading  # noqa: E402
from quantlab.execution.risk import RiskLimits  # noqa: E402
from quantlab.strategies.base import EXIT_PARAMS, Strategy  # noqa: E402
from quantlab.strategies.registry import STRATEGIES, build_strategy  # noqa: E402
from quantlab.ui import charts  # noqa: E402

logger = logging.getLogger("quantlab.ui")

SOURCE_LABELS = {
    "yfinance": "yfinance (2 min, ~60 jours)",
    "massive": "Massive (1 min, MASSIVE_API_KEY)",
    "ibkr": "IBKR (1 min, TWS requis)",
    "file": "Fichier (parquet / csv)",
}
STRATEGY_LABELS = {
    "trend": "Trend following",
    "mean_reversion": "Mean reversion",
    "arima": "ARIMA",
    "ml_hybrid": "ML Hybrid",
    "cascade": "Cascade",
    "pair": "Pair trading (2 symboles)",
}
OBJECTIVES = ("total_return_pct", "sharpe_ratio", "sortino_ratio", "profit_factor", "total_pnl_net", "win_rate")
EXCLUDED_PARAMS = {"self", "symbol", "symbol_a", "symbol_b", "strategy_id"}
SESSIONS_DIR = Path("sessions")


# --- shared widgets ----------------------------------------------------------------------------------------


#: exit parameters of every strategy (base class): label, help, kind; 0 disables a level or the time exit
EXIT_WIDGETS: dict[str, tuple[str, str, type]] = {
    "stop_loss_pct": ("Stop (% du prix d'entrée)", "0 = aucun stop : l'entrée est alors refusée.", float),
    "take_profit_pct": ("Objectif (% du prix d'entrée)", "0 = aucun objectif.", float),
    "holding_bars": ("Durée max (barres)", "Clôture au marché après ce nombre de barres ; 0 = aucune.", int),
    "stop_atr": (
        "Stop en ATR",
        "Distance du stop en multiples d'ATR, mesurée au signal ; remplace le %. 0 = non.",
        float,
    ),
    "take_profit_atr": (
        "Objectif en ATR",
        "Distance de l'objectif en multiples d'ATR ; remplace le %. 0 = non.",
        float,
    ),
    "atr_window": ("Fenêtre de l'ATR", "Barres de l'ATR utilisé par les niveaux en ATR.", int),
}
#: strategies whose stop / target apply to a group (pair: combined P&L, no ATR level)
GROUP_STRATEGIES = {"pair"}


def strategy_signature(name: str) -> dict[str, Any]:
    """Tunable constructor parameters of a strategy (its own, then the exit parameters) and their defaults."""
    params: dict[str, Any] = {}
    signatures = (inspect.signature(STRATEGIES[name].factory), inspect.signature(Strategy.__init__))
    for signature in signatures:
        for param in signature.parameters.values():
            if param.name in EXCLUDED_PARAMS or param.kind in (param.VAR_KEYWORD, param.VAR_POSITIONAL):
                continue
            params.setdefault(param.name, None if param.default is inspect.Parameter.empty else param.default)
    return params


def exit_inputs(key: str, group: bool = False) -> dict[str, Any]:
    """Exit parameters (stop, target, holding period); only values that differ from the default are returned."""
    defaults = inspect.signature(Strategy.__init__).parameters
    names = [n for n in EXIT_WIDGETS if not (group and n in ("stop_atr", "take_profit_atr", "atr_window"))]
    chosen: dict[str, Any] = {}
    columns = st.columns(3)
    for i, param in enumerate(names):
        label, help_text, kind = EXIT_WIDGETS[param]
        if group and param in ("stop_loss_pct", "take_profit_pct"):
            label = label.replace("du prix d'entrée", "du notionnel de la paire")
        default = defaults[param].default or 0
        widget_key = f"{key}_exit_{param}"
        with columns[i % 3]:
            if kind is int:
                value: Any = int(st.number_input(label, min_value=0, value=int(default), step=1, key=widget_key,
                                                 help=help_text))  # fmt: skip
            else:
                value = float(st.number_input(label, min_value=0.0, value=float(default), format="%g",
                                              key=widget_key, help=help_text))  # fmt: skip
        if value != default:
            chosen[param] = value
    return chosen


def parse_extra(text: str) -> dict[str, Any]:
    """Free `key=value` lines (same syntax as `--param`)."""
    extra: dict[str, Any] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        key, sep, value = line.partition("=")
        if not sep:
            raise ValueError(f"Paramètre invalide (attendu clé=valeur) : {line!r}")
        extra[key.strip()] = parse_value(value.strip())
    return extra


def param_inputs(name: str, key: str) -> dict[str, Any]:
    """Inputs for every parameter of a strategy; only values that differ from the default are returned."""
    chosen: dict[str, Any] = {}
    defaults = {k: v for k, v in strategy_signature(name).items() if k not in EXIT_PARAMS}
    columns = st.columns(3)
    for i, (param, default) in enumerate(defaults.items()):
        with columns[i % 3]:
            widget_key = f"{key}_{name}_{param}"
            if isinstance(default, bool):
                value: Any = st.checkbox(param, value=default, key=widget_key)
            elif isinstance(default, int):
                value = int(st.number_input(param, value=default, step=1, key=widget_key))
            elif isinstance(default, float):
                value = float(st.number_input(param, value=default, format="%g", key=widget_key))
            else:
                raw = st.text_input(param, value=repr(default), key=widget_key, help="Valeur Python, ex. (2, 1, 0)")
                value = ast.literal_eval(raw) if raw.strip() else default
            if value != default:
                chosen[param] = value
    group = name in GROUP_STRATEGIES
    st.markdown(
        "**Sorties** (par position)"
        + (" : stop et objectif sur le P&L combiné des deux jambes, clôturées ensemble" if group else "")
    )
    chosen.update(exit_inputs(f"{key}_{name}", group=group))
    extra = st.text_area(
        "Autres paramètres (clé=valeur, un par ligne)", key=f"{key}_{name}_extra", height=68, placeholder="sma_fast=10"
    )
    chosen.update(parse_extra(extra))
    return chosen


def data_inputs(key: str, multi_symbol: bool = True, default_days: int = 55) -> dict[str, Any]:
    columns = st.columns([2, 2, 1])
    with columns[0]:
        symbols_text = st.text_input(
            "Symbole(s)",
            value="AAPL",
            key=f"{key}_symbols",
            help="Séparés par des virgules ; deux symboles pour le pair trading (ex. AAPL, MSFT).",
        )
    with columns[1]:
        source = st.selectbox(
            "Source", list(loading.SOURCES), format_func=SOURCE_LABELS.__getitem__, key=f"{key}_source"
        )
    with columns[2]:
        days = int(st.number_input("Jours d'historique", min_value=1, value=default_days, key=f"{key}_days"))
    file_path: str | None = None
    if source == "file":
        uploaded = st.file_uploader("Fichier de barres", type=["parquet", "csv"], key=f"{key}_upload")
        typed = st.text_input("…ou chemin du fichier", key=f"{key}_path", placeholder="tests/fixtures/aapl_1m.parquet")
        if uploaded is not None:
            file_path = save_upload(uploaded.getvalue(), uploaded.name)
        elif typed.strip():
            file_path = typed.strip()
    symbols = [s.strip().upper() for s in symbols_text.split(",") if s.strip()]
    if not multi_symbol:
        symbols = symbols[:1]
    return {"symbols": symbols, "source": source, "days": days, "file": file_path}


def save_upload(content: bytes, name: str) -> str:
    """Uploaded files are written once (by content hash) to a temporary folder."""
    digest = hashlib.sha256(content).hexdigest()[:16]
    path = Path(tempfile.gettempdir()) / "quantlab_uploads" / f"{digest}{Path(name).suffix}"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return str(path)


def execution_inputs(key: str, live: bool = False) -> dict[str, Any]:
    """Capital, simulated costs and risk limits (the paper defaults are the CLI's conservative ones)."""
    with st.expander("Exécution et risque", expanded=False):
        c1, c2, c3 = st.columns(3)
        capital = c1.number_input("Capital ($)", min_value=1.0, value=100_000.0, step=10_000.0, key=f"{key}_capital")
        fee_bps = c2.number_input("Frais (bp par côté)", min_value=0.0, value=1.0, key=f"{key}_fee")
        slippage_bps = c3.number_input("Slippage simulé (bp)", min_value=0.0, value=1.0, key=f"{key}_slip")
        st.caption("Stop, objectif et durée de détention sont des paramètres de chaque stratégie (section Sorties).")
        c1, c2, c3 = st.columns(3)
        max_order = c1.number_input(
            "Max actions par ordre (0 = illimité)", min_value=0, value=10 if live else 0, key=f"{key}_maxo"
        )
        max_position = c2.number_input(
            "Max actions par symbole (0 = illimité)", min_value=0, value=50 if live else 0, key=f"{key}_maxp"
        )
        max_loss = c3.number_input(
            "Coupe-circuit perte jour (%, 0 = aucun)", min_value=0.0, value=1.0 if live else 0.0, key=f"{key}_dl"
        )
    return {
        "capital": float(capital),
        "fee_bps": float(fee_bps),
        "slippage_bps": float(slippage_bps),
        "risk": RiskLimits(
            max_order_quantity=int(max_order) or None,
            max_position_quantity=int(max_position) or None,
            max_daily_loss_pct=max_loss / 100 or None,
        ),
    }


@st.cache_data(show_spinner=False, ttl=3600)
def cached_bars(
    symbols: tuple[str, ...], source: str, days: int, file: str | None, account: str | None = None
) -> pd.DataFrame:
    return loading.load_bars(list(symbols), source, days, file, ibkr_account=account)


def get_bars(data: dict[str, Any]) -> pd.DataFrame:
    if not data["symbols"]:
        raise ValueError("Indiquez au moins un symbole.")
    with st.spinner(f"Chargement des données {data['source']}…"):
        account = current_account() if data["source"] == "ibkr" else None
        return cached_bars(tuple(data["symbols"]), data["source"], data["days"], data["file"], account)


def current_account() -> str:
    """IBKR session chosen on the Configuration page (default: IBKR_ACCOUNT, else paper)."""
    if "ibkr_account" not in st.session_state:
        st.session_state["ibkr_account"] = IBKRSettings.from_env().account
    account: str = st.session_state["ibkr_account"]
    return account


def account_label(account: str) -> str:
    settings = IBKRSettings.from_env(account)
    kind = "Paper" if account == "paper" else "Réel"
    return f"{kind} (port {settings.port})"


def metric_tiles(metrics: dict[str, float]) -> None:
    tiles = [
        ("Rendement", f"{metrics['total_return_pct']:+.3f} %"),
        ("P&L net", f"{metrics['total_pnl_net']:+,.2f} $"),
        ("Trades", f"{metrics['total_trades']:.0f}"),
        ("Réussite", f"{metrics['win_rate']:.1f} %"),
        ("Profit factor", f"{metrics['profit_factor']:.2f}"),
        ("Sharpe", f"{metrics['sharpe_ratio']:.2f}"),
        ("Drawdown max", f"{metrics['max_drawdown_pct']:.3f} %"),
    ]
    show_tiles(tiles)


def show_tiles(tiles: list[tuple[str, str]], per_row: int = 4) -> None:
    """Metric tiles, at most `per_row` per row so the values are never truncated."""
    for start in range(0, len(tiles), per_row):
        row = tiles[start : start + per_row]
        for column, (label, value) in zip(st.columns(per_row), row, strict=False):
            column.metric(label, value)


def show_error(exc: Exception) -> None:
    logger.exception("UI action failed")
    st.error(f"{type(exc).__name__} : {exc}")


# --- pages -------------------------------------------------------------------------------------------------


def page_backtest() -> None:
    st.header("Backtest")
    st.caption(
        "Entraîne les stratégies sur le début des données puis les fait trader barre par barre sur la fin, "
        "avec le même moteur que le live (exécution à l'open suivant, frais, slippage, stop / objectif par position)."
    )
    names = st.multiselect(
        "Stratégie(s)",
        list(STRATEGIES),
        default=["trend"],
        format_func=STRATEGY_LABELS.__getitem__,
        help="Plusieurs stratégies = un portefeuille.",
    )
    data = data_inputs("bt")
    c1, c2, c3 = st.columns(3)
    split_mode = c1.radio("Bloc de test", ["Derniers jours", "Dernière fraction"], key="bt_split", horizontal=True)
    if split_mode == "Derniers jours":
        test_days, test_fraction = c2.number_input("Jours de test", min_value=0.1, value=5.0), None
    else:
        test_days, test_fraction = 5.0, c2.slider("Fraction de test", 0.05, 0.9, 0.2)
    each = c3.checkbox("Aussi chaque stratégie seule", value=True, disabled=len(names) < 2)
    bar_by_bar = c3.checkbox("Chemin live barre à barre (plus lent)")
    params: dict[str, dict[str, Any]] = {}
    for name in names:
        with st.expander(f"Paramètres : {STRATEGY_LABELS[name]}"):
            params[name] = param_inputs(name, "bt")
    execution = execution_inputs("bt")

    if st.button("Lancer le backtest", type="primary", disabled=not names):
        try:
            st.session_state["backtest"] = run_backtests(names, params, data, test_days, test_fraction, each,
                                                         bar_by_bar, execution)  # fmt: skip
        except Exception as exc:
            show_error(exc)
    results = st.session_state.get("backtest")
    if results:
        show_backtest(results)


def run_backtests(
    names: list[str],
    params: dict[str, dict[str, Any]],
    data: dict[str, Any],
    test_days: float,
    test_fraction: float | None,
    each: bool,
    bar_by_bar: bool,
    execution: dict[str, Any],
) -> dict[str, Any]:
    from quantlab.backtest.run import backtest

    bars = get_bars(data)
    train, test = loading.split(bars, test_days, test_fraction)
    if train.empty or test.empty:
        raise ValueError(f"Découpage vide : {len(train)} barres d'entraînement, {len(test)} de test.")
    runs = [[n] for n in names] if each and len(names) > 1 else []
    runs.append(list(names))
    results = {}
    progress = st.progress(0.0, text="Backtest…")
    for i, run_names in enumerate(runs):
        label = " + ".join(STRATEGY_LABELS[n].split(" (")[0] for n in run_names)
        progress.progress(i / len(runs), text=f"Backtest : {label}")
        strategies = [build_strategy(n, data["symbols"], params[n]) for n in run_names]
        results[label] = backtest(
            strategies,
            train,
            test,
            capital=execution["capital"],
            fee_bps=execution["fee_bps"],
            slippage_bps=execution["slippage_bps"],
            risk_limits=execution["risk"],
            precompute_features=not bar_by_bar,
        )
    progress.empty()
    return {"results": results, "train": (train.index[0], train.index[-1], len(train)),
            "test": (test.index[0], test.index[-1], len(test))}  # fmt: skip


def show_backtest(state: dict[str, Any]) -> None:
    results = state["results"]
    t0, t1, n_train = state["train"]
    s0, s1, n_test = state["test"]
    st.caption(f"Entraînement : {t0} → {t1} ({n_train} barres) · Test : {s0} → {s1} ({n_test} barres)")
    labels = list(results)
    main = labels[-1]
    st.subheader(main)
    metric_tiles(results[main].metrics)
    st.altair_chart(charts.equity_chart({label: results[label].equity for label in labels}), width="stretch")
    st.altair_chart(charts.drawdown_chart(results[main].equity), width="stretch")
    if len(labels) > 1:
        st.dataframe(pd.DataFrame({k: v.metrics for k, v in results.items()}).T, width="stretch")

    trades = results[main].trades_frame()
    st.subheader(f"Trades ({len(trades)})")
    if trades.empty:
        st.info("Aucun trade.")
        return
    st.altair_chart(charts.trade_pnl_chart(trades), width="stretch")
    st.dataframe(trades, width="stretch", height=260, hide_index=True)
    c1, c2 = st.columns(2)
    c1.download_button("Trades (CSV)", trades.to_csv(index=False), "trades.csv", "text/csv")
    c2.download_button("Courbe de capital (CSV)", results[main].equity.to_csv(), "equity.csv", "text/csv")


def page_optimize() -> None:
    st.header("Optimisation walk-forward")
    st.caption(
        "Chaque fold : [entraînement][embargo][validation][embargo][test]. La meilleure combinaison sur la "
        "validation est ré-entraînée puis évaluée une seule fois sur le test."
    )
    name = st.selectbox("Stratégie", list(STRATEGIES), format_func=STRATEGY_LABELS.__getitem__, key="wf_strategy")
    data = data_inputs("wf")
    st.markdown("**Grille** : valeurs séparées par des virgules ; laisser vide pour garder la valeur par défaut.")
    grid: dict[str, list[Any]] = {}
    defaults = strategy_signature(name)
    columns = st.columns(3)
    for i, (param, default) in enumerate(defaults.items()):
        raw = columns[i % 3].text_input(f"{param} (défaut {default!r})", key=f"wf_grid_{name}_{param}")
        if raw.strip():
            grid[param] = [parse_value(v.strip()) for v in raw.split(",") if v.strip()]
    fixed = parse_extra(st.text_area("Paramètres fixes (clé=valeur)", key=f"wf_fixed_{name}", height=68))
    c1, c2, c3, c4 = st.columns(4)
    train_size = int(c1.number_input("Barres d'entraînement", min_value=10, value=3000, step=100))
    validation_size = int(c2.number_input("Barres de validation", min_value=10, value=1000, step=100))
    test_size = int(c3.number_input("Barres de test", min_value=10, value=1000, step=100))
    embargo = int(c4.number_input("Embargo (barres)", min_value=0, value=30))
    c1, c2, c3, c4 = st.columns(4)
    step = int(c1.number_input("Pas entre folds (0 = taille du test)", min_value=0, value=0))
    objective = c2.selectbox("Objectif", OBJECTIVES)
    min_trades = int(c3.number_input("Trades minimum", min_value=0, value=5))
    anchored = c4.checkbox("Entraînement ancré (fenêtre croissante)")

    if st.button("Lancer l'optimisation", type="primary", disabled=not grid):
        try:
            from quantlab.backtest.walk_forward import WalkForwardOptimizer

            bars = get_bars(data)
            optimizer = WalkForwardOptimizer(
                strategy_factory=lambda **p: build_strategy(name, data["symbols"], {**fixed, **p}),
                param_grid=grid,
                train_size=train_size,
                validation_size=validation_size,
                test_size=test_size,
                step=step or None,
                embargo=embargo,
                anchored=anchored,
                objective=objective,
                min_trades=min_trades,
            )
            n = len(optimizer.candidates)
            with st.spinner(f"{n} combinaisons par fold…"):
                st.session_state["optimize"] = optimizer.run(bars)
        except Exception as exc:
            show_error(exc)
    if not grid:
        st.info("Renseignez au moins un paramètre de la grille.")
    result = st.session_state.get("optimize")
    if result is not None:
        summary = result.summary()
        st.metric(f"Score moyen hors échantillon ({result.objective})", f"{result.mean_test_score:,.4f}")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Score de test par fold**")
            st.altair_chart(charts.value_bars(summary["test_score"], result.objective, "Fold"), width="stretch")
        with c2:
            st.markdown("**Stabilité des paramètres choisis** (nombre de folds)")
            stability = result.param_stability()
            st.altair_chart(charts.value_bars(stability, "Folds", "Paramètres", fmt="d"), width="stretch")
        st.dataframe(summary, width="stretch")


def page_live() -> None:
    from quantlab.execution.ibkr.live import LiveConfig

    st.header("Session live IBKR")
    account = current_account()
    settings = IBKRSettings.from_env(account)
    mode = "shadow : barres live, exécutions simulées, aucun ordre envoyé" if settings.read_only else (
        "ordres envoyés à IBKR (paper sur un compte DU…, réel seulement avec confirmation)")  # fmt: skip
    st.info(
        f"Compte {account_label(account)} · TWS / IB Gateway {settings.host}:{settings.port} "
        f"(client id {settings.client_id}) · "
        f"IBKR_READ_ONLY={'true' if settings.read_only else 'false'} → mode {mode}. "
        "Compte à changer dans la page Configuration, réglages dans le fichier .env."
    )
    registry = live_sessions()
    with st.form("live_form"):
        c1, c2 = st.columns(2)
        name = c1.selectbox("Stratégie", list(STRATEGIES), format_func=STRATEGY_LABELS.__getitem__)
        symbols_text = c2.text_input("Symbole(s)", value="AAPL", help="Deux symboles pour le pair trading.")
        c1, c2, c3 = st.columns(3)
        session_name = c1.text_input("Nom de session", placeholder="défaut : <stratégie>_<symboles>_<date>")
        resume = c2.checkbox("Reprendre une session existante")
        rth = c3.checkbox("Heures régulières seulement")
        c1, c2 = st.columns(2)
        bar_seconds = int(c1.number_input("Durée des barres (s)", min_value=5, value=60, step=5))
        history = c2.text_input("Historique de préchauffage IBKR", placeholder="défaut selon la stratégie, ex. 2 D")
        st.markdown("**Sorties** (par position ; pour le pair trading : sur le P&L combiné, sans ATR)")
        exits = exit_inputs("live")
        params_text = st.text_area("Autres paramètres de la stratégie (clé=valeur)", height=68)
        execution = execution_inputs("live", live=True)
        st.markdown("**Compte réel** : jamais par défaut.")
        c1, c2 = st.columns(2)
        allow_live = c1.checkbox("Autoriser un compte réel (équivalent de `--live`)")
        confirm = c2.text_input("Identifiant du compte, pour confirmer", type="password")
        submitted = st.form_submit_button("Démarrer la session", type="primary")
    if submitted:
        try:
            symbols = [s.strip().upper() for s in symbols_text.split(",") if s.strip()]
            spec = STRATEGIES[name]
            session = session_name.strip() or f"{name}_{'_'.join(symbols)}_{date.today():%Y%m%d}"
            config = LiveConfig(
                symbols=symbols,
                bar_seconds=bar_seconds,
                history_duration=history.strip() or spec.live_history,
                use_rth=rth,
                capital=execution["capital"],
                risk=execution["risk"],
                fee_bps=execution["fee_bps"],
                slippage_bps=execution["slippage_bps"],
            )
            params = {**exits, **parse_extra(params_text)}
            registry.start(session, SESSIONS_DIR / session, name, symbols, params, config,
                           resume=resume, allow_live=allow_live, confirm_account=confirm,
                           account=account)  # fmt: skip
            st.success(f"Session {session} lancée.")
        except Exception as exc:
            show_error(exc)
    live_monitor()


@st.cache_resource
def live_sessions() -> Any:
    from quantlab.ui.live_runner import LiveSessions

    return LiveSessions()


@st.fragment(run_every=5)
def live_monitor() -> None:
    entries = live_sessions().all()
    if not entries:
        st.caption("Aucune session lancée depuis cette interface.")
        return
    for entry in reversed(entries):
        with st.container(border=True):
            c1, c2 = st.columns([4, 1])
            c1.markdown(
                f"**{entry.name}** · {STRATEGY_LABELS[entry.strategy]} sur {', '.join(entry.symbols)} · "
                f"statut **{entry.status}**" + (f" · mode {entry.mode}" if entry.mode else "")
            )
            if entry.alive and c2.button("Arrêter", key=f"stop_{entry.name}_{entry.started.value}"):
                entry.stop()
                st.toast(f"Arrêt de {entry.name} demandé")
            if entry.error:
                st.error(entry.error)
            journal = entry.path / "journal.sqlite"
            if journal.exists():
                journal_overview(journal, compact=True)


def journal_overview(path: Path, compact: bool = False) -> None:
    from quantlab.runtime.journal import read_journal

    journal = read_journal(path)
    equity = journal.kind("equity")
    trades = journal.kind("trade")
    if equity.empty:
        st.caption("Pas encore de barre.")
        return
    last = equity.iloc[-1]
    latency = equity["latency_ms"].dropna().astype(float)
    tiles = [
        ("Barres", f"{len(journal.bars)}"),
        ("Capital", f"{last['equity']:,.2f} $"),
        ("Trades", f"{len(trades)}"),
        ("P&L net", f"{trades['pnl_net'].sum() if not trades.empty else 0:+,.2f} $"),
        ("Positions", ", ".join(f"{k} {v:+d}" for k, v in (last["positions"] or {}).items()) or "aucune"),
        ("Latence p50", f"{latency.median():.1f} ms" if len(latency) else "-"),
        ("Coupe-circuit", "ACTIF" if last["halted"] else "non"),
    ]
    show_tiles(tiles)
    curve = pd.Series(equity["equity"].astype(float).to_numpy(), index=pd.DatetimeIndex(equity["bar_ts"]))
    st.altair_chart(charts.equity_chart({"Capital": curve}, height=200), width="stretch")
    if compact:
        events = journal.events[journal.events["kind"].isin(["signal", "order", "fill", "status", "trade"])]
        if not events.empty:
            st.dataframe(readable(events.tail(8))[["bar_ts", "kind", "strategy_id", "data"]], width="stretch")
        return
    if len(latency):
        st.markdown("**Latence par barre** (ms, barre reçue → ordres envoyés)")
        latency_times = equity.loc[latency.index, "bar_ts"]
        st.altair_chart(charts.time_line(latency_times, latency, "Latence (ms)"), width="stretch")
    if not trades.empty:
        st.markdown(f"**Trades ({len(trades)})**")
        st.altair_chart(charts.trade_pnl_chart(trades), width="stretch")
        st.dataframe(trades.drop(columns=["wall_ts"]), width="stretch", height=240)
    kinds = sorted(journal.events["kind"].unique())
    shown = st.multiselect("Événements", kinds, default=[k for k in ("signal", "fill", "trade") if k in kinds])
    events = journal.events[journal.events["kind"].isin(shown)]
    st.dataframe(readable(events)[["bar_ts", "wall_ts", "kind", "strategy_id", "data"]], width="stretch", height=260)


def readable(events: pd.DataFrame) -> pd.DataFrame:
    """Journal events with their JSON payload as text (tables cannot show nested dicts)."""
    return events.assign(data=events["data"].astype(str))


def session_names() -> list[str]:
    if not SESSIONS_DIR.exists():
        return []
    return sorted((p.name for p in SESSIONS_DIR.iterdir() if (p / "journal.sqlite").exists()), reverse=True)


def page_sessions() -> None:
    st.header("Sessions")
    names = session_names()
    if not names:
        st.info(f"Aucune session dans {SESSIONS_DIR.resolve()}. Lancez-en une depuis la page « Session live ».")
        return
    name = st.selectbox("Session", names)
    path = SESSIONS_DIR / name
    meta_path = path / "session.json"
    if meta_path.exists():
        from quantlab.runtime.session import SessionFiles

        meta = SessionFiles(path).meta()
        st.caption(
            f"Mode {meta.get('mode')} · {', '.join(meta.get('symbols', []))} · stratégies "
            f"{', '.join(meta.get('strategies', []))} · capital {meta.get('capital', 0):,.0f} $"
        )
    journal_tab, replay_tab = st.tabs(["Journal", "Replay (live vs backtest)"])
    with journal_tab:
        try:
            journal_overview(path / "journal.sqlite")
        except Exception as exc:
            show_error(exc)
    with replay_tab:
        replay_section(path)


def replay_section(path: Path) -> None:
    st.caption(
        "Relance les stratégies de la session, telles qu'avant la première barre live, sur les barres "
        "journalisées avec le broker simulé, puis compare signaux, prix d'exécution, features et P&L."
    )
    override = st.checkbox("Remplacer le slippage simulé de la session")
    slippage = st.number_input("Slippage simulé (bp)", min_value=0.0, value=1.0, disabled=not override)
    if st.button("Lancer le replay", type="primary"):
        try:
            from quantlab.analysis.replay import replay_session

            with st.spinner("Replay…"):
                st.session_state[f"replay_{path.name}"] = replay_session(path, slippage_bps=slippage if override
                                                                         else None)  # fmt: skip
        except Exception as exc:
            show_error(exc)
    report = st.session_state.get(f"replay_{path.name}")
    if report is None:
        return
    s = report.summary
    tiles = [
        ("Accord des signaux", f"{s['signal_agreement_pct']:.1f} %"),
        ("Signaux live / replay", f"{s['signals_live']:.0f} / {s['signals_replay']:.0f}"),
        ("Slippage d'entrée", f"{s['entry_slippage_bps_mean']:+.2f} bp"),
        ("P&L live", f"{s['pnl_net_live']:+,.2f} $"),
        ("P&L replay", f"{s['pnl_net_replay']:+,.2f} $"),
        ("Écart max features", f"{s['max_feature_diff']:.2g}"),
    ]
    show_tiles(tiles)
    if not report.entry_fills.empty:
        st.markdown("**Slippage d'entrée par signal** (bp, positif = le live a payé plus cher)")
        fills = report.entry_fills.reset_index(drop=True)
        values = pd.Series(fills["slippage_bps"].to_numpy(), index=[f"{i + 1}. {r.side} {r.symbol}" for i, r in
                                                                    fills.iterrows()])  # fmt: skip
        st.altair_chart(charts.value_bars(values, "Slippage (bp)", "Signal", fmt="+.2f"), width="stretch")
    if not report.feature_diffs.empty and report.feature_diffs.max() > 0:
        st.markdown("**Plus grands écarts de features live / replay**")
        st.altair_chart(charts.value_bars(report.feature_diffs.head(20), "Écart max", "Feature", fmt=".3g"),
                        width="stretch")  # fmt: skip
    st.markdown("**Signaux**")
    st.dataframe(report.signals, width="stretch", height=260)
    st.dataframe(pd.Series(s, name="valeur"), width="stretch")


def page_chart() -> None:
    from quantlab.analysis.plot import levels_figure, session_bars

    st.header("Graphique de séance")
    st.caption(
        "Bougies, volume, VWAP et ses bandes, et les meilleures lignes de support / résistance. Ces lignes "
        "utilisent des barres futures : elles servent à l'analyse visuelle, jamais de signal."
    )
    data = data_inputs("chart", multi_symbol=False, default_days=3)
    c1, c2 = st.columns(2)
    day = c1.date_input("Séance (vide = la dernière)", value=None)
    n_lines = int(c2.number_input("Lignes de support / résistance", min_value=0, max_value=10, value=3))
    if st.button("Tracer", type="primary"):
        try:
            bars, chosen = session_bars(get_bars(data), day.isoformat() if day else None)
            with st.spinner("Calcul des niveaux…"):
                figure = levels_figure(bars, f"{data['symbols'][0]} {chosen} (New York time)", n_lines)
            buffer = io.BytesIO()
            figure.savefig(buffer, format="png", dpi=120)
            st.session_state["chart"] = (figure, buffer.getvalue(), f"{data['symbols'][0]}_{chosen}.png")
        except Exception as exc:
            show_error(exc)
    if "chart" in st.session_state:
        figure, png, filename = st.session_state["chart"]
        st.pyplot(figure)
        st.download_button("Télécharger (PNG)", png, filename, "image/png")


def page_config() -> None:
    st.header("Configuration")
    st.subheader("Compte IBKR")
    st.caption(
        "Choisit la session TWS / IB Gateway utilisée par toutes les pages : données IBKR, sessions live et "
        "graphiques. Ports lus dans .env : IBKR_PORT_PAPER pour le paper, IBKR_PORT pour le compte réel."
    )
    accounts = list(ACCOUNTS)
    chosen = st.radio(
        "Compte",
        accounts,
        index=accounts.index(current_account()),
        format_func=account_label,
        horizontal=True,
    )
    st.session_state["ibkr_account"] = chosen
    settings = IBKRSettings.from_env(chosen)
    if chosen == "live":
        st.warning(
            "Compte réel sélectionné. Tant que IBKR_READ_ONLY=true, aucun ordre ne part (mode shadow). Sinon, "
            "une session exige en plus la case « Autoriser un compte réel » et la saisie de l'identifiant du compte."
        )
    st.dataframe(
        pd.Series(
            {
                "Hôte": settings.host,
                "Port": str(settings.port),
                "Client id": str(settings.client_id),
                "Lecture seule (IBKR_READ_ONLY)": "oui" if settings.read_only else "non",
                "Compte réel autorisé (IBKR_ALLOW_LIVE)": "oui" if settings.allow_live_account else "non",
            },
            name="valeur",
        ),
        width="stretch",
    )
    if st.button("Tester la connexion (données uniquement)"):
        try:
            with st.spinner("Connexion…"), loading.ibkr_client(account=chosen) as client:
                accounts_found = ", ".join(client.accounts) or "inconnu"
                kind = "paper" if client.is_paper else "réel"
            st.success(f"Connecté à {settings.host}:{settings.port} · compte {accounts_found} ({kind}).")
        except Exception as exc:
            show_error(exc)


def main() -> None:
    st.set_page_config(page_title="quantlab", page_icon=":chart_with_upwards_trend:", layout="wide")
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    navigation = st.navigation(
        [
            st.Page(page_backtest, title="Backtest", icon=":material/query_stats:", default=True),
            st.Page(page_optimize, title="Optimisation", icon=":material/tune:", url_path="optimisation"),
            st.Page(page_live, title="Session live", icon=":material/sensors:", url_path="live"),
            st.Page(page_sessions, title="Sessions & replay", icon=":material/history:", url_path="sessions"),
            st.Page(page_chart, title="Graphique", icon=":material/candlestick_chart:", url_path="graphique"),
            st.Page(page_config, title="Configuration", icon=":material/settings:", url_path="configuration"),
        ]
    )
    st.sidebar.caption(f"Compte IBKR : **{account_label(current_account())}**")
    navigation.run()


if __name__ == "__main__":
    main()
