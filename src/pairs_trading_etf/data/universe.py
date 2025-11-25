"""Utilities for managing ETF universes and metadata."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

from pairs_trading_etf.utils.config import ConfigError, load_yaml_config


@dataclass(frozen=True)
class ETFMetadata:
    """Rich description of an ETF used for filtering and reporting."""

    ticker: str
    name: str
    sector: str
    region: str | None = None
    issuer: str | None = None
    expense_ratio: float | None = None
    benchmark: str | None = None
    description: str | None = None
    inception: str | None = None

    @classmethod
    def from_mapping(cls, ticker: str, payload: Mapping[str, Any]) -> "ETFMetadata":
        """Construct metadata from a mapping loaded from YAML."""

        ticker_upper = ticker.upper()
        expense = payload.get("expense_ratio")
        expense_ratio = float(expense) if expense is not None else None

        return cls(
            ticker=ticker_upper,
            name=str(payload.get("name", ticker_upper)),
            sector=str(payload.get("sector", "Unknown")),
            region=payload.get("region"),
            issuer=payload.get("issuer"),
            expense_ratio=expense_ratio,
            benchmark=payload.get("benchmark"),
            description=payload.get("description"),
            inception=payload.get("inception"),
        )


@dataclass(frozen=True)
class ETFUniverse:
    """Represents a concrete ETF universe with optional metadata."""

    name: str
    tickers: tuple[str, ...]
    description: str | None = None
    sectors: tuple[str, ...] | None = None
    metadata: Mapping[str, ETFMetadata] | None = None

    def as_list(self) -> list[str]:
        """Return tickers as a mutable list."""

        return list(self.tickers)

    def missing_metadata(self) -> list[str]:
        """Return tickers that lack metadata entries."""

        if not self.metadata:
            return []
        return [ticker for ticker in self.tickers if ticker not in self.metadata]

    def require_metadata(self) -> None:
        """Ensure every ticker has metadata, raising ConfigError if not."""

        missing = self.missing_metadata()
        if missing:
            raise ConfigError(
                f"ETF metadata missing for: {', '.join(sorted(missing))}"
            )

    def to_records(self) -> list[Dict[str, Any]]:
        """Return universe entries as a list of dictionaries."""

        records: list[Dict[str, Any]] = []
        for ticker in self.tickers:
            meta = self.metadata.get(ticker) if self.metadata else None
            record: Dict[str, Any] = {"ticker": ticker}
            if meta:
                record.update(
                    {
                        "name": meta.name,
                        "sector": meta.sector,
                        "region": meta.region,
                        "issuer": meta.issuer,
                        "expense_ratio": meta.expense_ratio,
                        "benchmark": meta.benchmark,
                        "description": meta.description,
                        "inception": meta.inception,
                    }
                )
            records.append(record)
        return records


def _normalize_tickers(tickers: Iterable[str]) -> tuple[str, ...]:
    """Normalize a ticker sequence: uppercase, strip, and drop duplicates."""

    seen: set[str] = set()
    normalized: list[str] = []
    for raw in tickers:
        ticker = str(raw).strip().upper()
        if not ticker or ticker in seen:
            continue
        normalized.append(ticker)
        seen.add(ticker)
    if not normalized:
        raise ConfigError("Universe definition produced an empty ticker list")
    return tuple(normalized)


def load_etf_metadata(path: str | Path) -> Dict[str, ETFMetadata]:
    """Load ETF metadata from a YAML file."""

    raw = load_yaml_config(path)
    catalog = raw.get("etfs")
    if not isinstance(catalog, Mapping) or not catalog:
        raise ConfigError("ETF metadata file must expose an 'etfs' mapping")

    metadata: Dict[str, ETFMetadata] = {}
    for ticker, payload in catalog.items():
        if not isinstance(payload, Mapping):
            raise ConfigError(f"Metadata for {ticker} must be a mapping")
        entry = ETFMetadata.from_mapping(ticker, payload)
        metadata[entry.ticker] = entry
    return metadata


def resolve_universe(
    config: Mapping[str, Any],
    list_name: str | None = None,
    metadata: Mapping[str, ETFMetadata] | None = None,
) -> ETFUniverse:
    """Resolve a ticker universe from the project config."""

    universe_cfg = config.get("universe")
    if not isinstance(universe_cfg, Mapping):
        raise ConfigError("Config missing 'universe' section")

    lists = universe_cfg.get("lists")
    selected_name = list_name or universe_cfg.get("default_list")

    if selected_name and isinstance(lists, Mapping) and selected_name in lists:
        entry = lists[selected_name]
        if not isinstance(entry, Mapping):
            raise ConfigError(f"Universe list '{selected_name}' must be a mapping")
        tickers = _normalize_tickers(entry.get("tickers", []))
        description = entry.get("description")
        sectors = tuple(entry.get("sectors", [])) or None
        universe = ETFUniverse(
            name=str(selected_name),
            tickers=tickers,
            description=description,
            sectors=sectors,
            metadata=metadata,
        )
    elif "etfs" in universe_cfg:
        tickers = _normalize_tickers(universe_cfg.get("etfs", []))
        universe = ETFUniverse(
            name=str(selected_name or "legacy"),
            tickers=tickers,
            description=None,
            sectors=None,
            metadata=metadata,
        )
    else:
        raise ConfigError("Universe config must define either 'lists' or 'etfs'")

    if metadata:
        universe.require_metadata()

    return universe


def load_configured_universe(
    config_path: str | Path,
    list_name: str | None = None,
    metadata_path: str | Path | None = None,
) -> ETFUniverse:
    """Load both config and metadata in a single helper."""

    config = load_yaml_config(config_path)
    metadata_section = config.get("metadata")
    if not isinstance(metadata_section, Mapping):
        metadata_section = {}

    metadata_file: str | Path | None = metadata_path or metadata_section.get("etf_info_path")

    metadata = load_etf_metadata(metadata_file) if metadata_file else None

    return resolve_universe(config, list_name=list_name, metadata=metadata)
