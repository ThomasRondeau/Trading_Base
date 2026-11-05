"""Runtime settings read from the environment (or a .env file at the repo root)."""

from __future__ import annotations

import os
from dataclasses import dataclass

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # python-dotenv is optional
    pass

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in _TRUE:
        return True
    if value in _FALSE:
        return False
    raise ValueError(f"{name}={raw!r} is not a boolean (use true/false).")


@dataclass(frozen=True)
class IBKRSettings:
    host: str = "127.0.0.1"
    port: int = 7497  # TWS paper trading default
    client_id: int = 5
    # Safety first: orders are refused unless IBKR_READ_ONLY=false is set explicitly.
    read_only: bool = True
    #: orders are refused on a real-money account unless IBKR_ALLOW_LIVE=true (paper accounts are DU...)
    allow_live_account: bool = False

    @classmethod
    def from_env(cls) -> IBKRSettings:
        return cls(
            host=os.environ.get("IBKR_HOST", cls.host),
            port=int(os.environ.get("IBKR_PORT", cls.port)),
            client_id=int(os.environ.get("IBKR_CLIENT_ID", cls.client_id)),
            read_only=_env_bool("IBKR_READ_ONLY", cls.read_only),
            allow_live_account=_env_bool("IBKR_ALLOW_LIVE", cls.allow_live_account),
        )
