"""
Live foreign-exchange rates via api.frankfurter.app.

Provides cached conversion utilities for the planning agent.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Dict, Optional

import httpx

logger = logging.getLogger(__name__)

EXCHANGE_API_URL = "https://api.frankfurter.app/latest"
CACHE_TTL_SECONDS = 3600  # Refresh rates hourly


class FXService:
    """Fetch and cache exchange rates using Frankfurter API."""

    def __init__(self) -> None:
        self._rates_cache: Dict[str, Dict[str, float]] = {}
        self._timestamp_cache: Dict[str, float] = {}
        self._lock = threading.Lock()

    def convert(self, amount: float, source_currency: str, target_currency: str) -> Optional[float]:
        """Convert amount from source_currency to target_currency."""
        if amount is None:
            return None
        source_currency = source_currency.upper()
        target_currency = target_currency.upper()
        if source_currency == target_currency:
            return float(amount)

        rates = self._get_rates(source_currency)
        if not rates:
            return None

        rate = rates.get(target_currency)
        if rate is None or rate <= 0:
            return None

        return float(amount) * rate

    def get_rate(self, source_currency: str, target_currency: str) -> Optional[float]:
        """Return exchange rate for converting source_currency→target_currency."""
        source_currency = source_currency.upper()
        target_currency = target_currency.upper()
        if source_currency == target_currency:
            return 1.0

        rates = self._get_rates(source_currency)
        if not rates:
            return None
        return rates.get(target_currency)

    def _get_rates(self, base_currency: str) -> Optional[Dict[str, float]]:
        base_currency = base_currency.upper()
        now = time.time()
        cached = self._rates_cache.get(base_currency)
        if cached and (now - self._timestamp_cache.get(base_currency, 0.0)) < CACHE_TTL_SECONDS:
            return cached

        with self._lock:
            cached = self._rates_cache.get(base_currency)
            if cached and (now - self._timestamp_cache.get(base_currency, 0.0)) < CACHE_TTL_SECONDS:
                return cached
            try:
                params = {"from": base_currency}
                with httpx.Client(timeout=5.0) as client:
                    response = client.get(EXCHANGE_API_URL, params=params)
                    response.raise_for_status()
                    payload = response.json()

                rates = payload.get("rates")
                if isinstance(rates, dict):
                    self._rates_cache[base_currency] = rates
                    self._timestamp_cache[base_currency] = now
                    logger.debug("FX rates refreshed for base %s", base_currency)
                    return rates

                logger.warning("Unexpected FX payload for base %s: %s", base_currency, payload)
                return None
            except Exception as exc:
                logger.error("Failed to fetch FX rates for base %s: %s", base_currency, exc)
                return cached  # fall back to stale cache if available


_fx_service: Optional[FXService] = None


def get_fx_service() -> FXService:
    """Return FX service singleton."""
    global _fx_service
    if _fx_service is None:
        _fx_service = FXService()
    return _fx_service
