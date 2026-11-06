"""
Live sessions started from the UI run in background threads (the UI script reruns on every interaction).
A `LiveSessions` registry lives as long as the UI process and keeps their status, errors and stop handle.
The session itself is the same `open_live_session` / `LiveSession.run` as the `paper` command.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from quantlab.config import IBKRSettings
from quantlab.execution.ibkr.client import IBKRClient
from quantlab.execution.ibkr.live import (
    LiveConfig,
    LiveSession,
    authorize_live_account,
    needs_live_confirmation,
    open_live_session,
    session_mode,
)
from quantlab.strategies.registry import STRATEGIES

logger = logging.getLogger(__name__)


@dataclass
class RunningSession:
    name: str
    path: Path
    strategy: str
    symbols: list[str]
    started: pd.Timestamp = field(default_factory=lambda: pd.Timestamp.now(tz="UTC"))
    status: str = "connexion"
    mode: str = ""
    accounts: list[str] = field(default_factory=list)
    error: str | None = None
    session: LiveSession | None = None
    thread: threading.Thread | None = None
    _stop: threading.Event = field(default_factory=threading.Event)

    @property
    def alive(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def stop(self) -> None:
        self._stop.set()
        if self.session is not None:
            self.session.stop()


class LiveSessions:
    def __init__(self, client_factory: Callable[[str], IBKRClient] | None = None) -> None:
        """`client_factory(account)` builds the IBKR client of the "paper" or "live" session."""
        self.client_factory = client_factory or (
            lambda account: IBKRClient(IBKRSettings.from_env(account), auto_reconnect=True)
        )
        self._lock = threading.Lock()
        self._sessions: dict[str, RunningSession] = {}

    def all(self) -> list[RunningSession]:
        with self._lock:
            return list(self._sessions.values())

    def start(
        self,
        name: str,
        path: Path,
        strategy: str,
        symbols: Sequence[str],
        params: Mapping[str, Any],
        config: LiveConfig,
        resume: bool = False,
        allow_live: bool = False,
        confirm_account: str | None = None,
        account: str = "paper",
    ) -> RunningSession:
        with self._lock:
            current = self._sessions.get(name)
            if current is not None and current.alive:
                raise RuntimeError(f"Session {name!r} is already running.")
            entry = RunningSession(name, path, strategy, list(symbols))
            self._sessions[name] = entry
        entry.thread = threading.Thread(
            target=self._run,
            kwargs={"client_factory": lambda: self.client_factory(account)},
            args=(entry, params, config, resume, allow_live, confirm_account),
            name=f"live-{name}",
            daemon=True,
        )
        entry.thread.start()
        return entry

    @staticmethod
    def _run(
        entry: RunningSession,
        params: Mapping[str, Any],
        config: LiveConfig,
        resume: bool,
        allow_live: bool,
        confirm_account: str | None,
        client_factory: Callable[[], IBKRClient],
    ) -> None:
        spec = STRATEGIES[entry.strategy]
        client = client_factory()
        try:
            client.connect()
            entry.mode, entry.accounts = session_mode(client), list(client.accounts)
            if needs_live_confirmation(client):
                if not allow_live:
                    raise PermissionError(
                        f"{', '.join(client.accounts)} is a real-money account: tick 'allow a real-money account'."
                    )
                authorize_live_account(client, confirm_account)
            entry.status = "préchauffage"
            entry.session = open_live_session(
                client, lambda: [spec.build(entry.symbols, dict(params))], entry.path, config, resume
            )
            if entry._stop.is_set():
                entry.session.stop()
            entry.status = "en cours"
            entry.session.run()
            entry.status = "arrêtée"
        except Exception as exc:
            logger.exception("Live session %s failed", entry.name)
            entry.status, entry.error = "erreur", f"{type(exc).__name__}: {exc}"
        finally:
            client.disconnect()
