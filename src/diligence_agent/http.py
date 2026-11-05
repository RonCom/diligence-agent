"""HTTP client for the tools: per-host rate limits, timeouts, and retries with full-jitter backoff.

Retries cover transport errors and 429/5xx. A Retry-After header (seconds or an HTTP date) takes the place of the
backoff, capped at max_retry_after_s. Other 4xx responses raise at once, except 404, which openFDA uses for
"no matches" and is returned to the caller.
"""

from __future__ import annotations

import email.utils
import math
import random
import threading
import time
from collections.abc import Callable
from pathlib import Path
from urllib.parse import urlsplit

import httpx

RETRY_STATUS = frozenset({429, 500, 502, 503, 504})
DEFAULT_USER_AGENT = "diligence-agent/0.1 (https://github.com/RonCom/diligence-agent)"


class HttpError(RuntimeError):
    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


def retry_after_seconds(value: str | None, now: float | None = None) -> float | None:
    if not value:
        return None
    value = value.strip()
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        when = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    now = time.time() if now is None else now
    return max(0.0, when.timestamp() - now)


def backoff_seconds(attempt: int, base: float, cap: float, rng: Callable[[], float] = random.random) -> float:
    """Full jitter: uniform on [0, min(cap, base * 2**attempt)]."""
    return rng() * min(cap, base * 2**attempt)


class HttpClient:
    def __init__(
        self,
        user_agent: str = DEFAULT_USER_AGENT,
        rate_limits: dict[str, float] | None = None,
        default_rate: float = 2.0,
        timeout_s: float = 20.0,
        max_retries: int = 3,
        backoff_base_s: float = 1.0,
        backoff_cap_s: float = 30.0,
        max_retry_after_s: float = 60.0,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        rng: Callable[[], float] = random.random,
    ) -> None:
        self.rate_limits = dict(rate_limits or {})
        self.default_rate = default_rate
        self.max_retries = max_retries
        self.backoff_base_s = backoff_base_s
        self.backoff_cap_s = backoff_cap_s
        self.max_retry_after_s = max_retry_after_s
        self.sleep = sleep
        self.rng = rng
        self.requests = 0  # HTTP requests sent, retries included
        self._last: dict[str, float] = {}
        self._lock = threading.Lock()
        self.client = httpx.Client(
            headers={"User-Agent": user_agent}, timeout=timeout_s, follow_redirects=True, transport=transport
        )

    @classmethod
    def from_config(cls, cfg: dict, **kwargs) -> HttpClient:
        http = cfg.get("http", {})
        return cls(
            rate_limits=http.get("rate_limits"),
            timeout_s=http.get("timeout_s", 20.0),
            max_retries=http.get("max_retries", 3),
            backoff_base_s=http.get("backoff_base_s", 1.0),
            backoff_cap_s=http.get("backoff_cap_s", 30.0),
            **kwargs,
        )

    def _throttle(self, host: str) -> None:
        interval = 1.0 / self.rate_limits.get(host, self.default_rate)
        with self._lock:
            now = time.monotonic()
            wait = self._last.get(host, -math.inf) + interval - now
            if wait > 0:
                self.sleep(wait)
                now += wait
            self._last[host] = now

    def get(self, url: str, params: dict | None = None, headers: dict | None = None) -> httpx.Response:
        host = urlsplit(url).hostname or ""
        for attempt in range(self.max_retries + 1):
            self._throttle(host)
            self.requests += 1
            try:
                resp = self.client.get(url, params=params, headers=headers)
            except httpx.TransportError as exc:
                if attempt == self.max_retries:
                    raise HttpError(f"{type(exc).__name__} after {attempt + 1} tries: {url}") from exc
                self.sleep(backoff_seconds(attempt, self.backoff_base_s, self.backoff_cap_s, self.rng))
                continue
            if resp.status_code in RETRY_STATUS and attempt < self.max_retries:
                wait = retry_after_seconds(resp.headers.get("Retry-After"))
                if wait is None:
                    wait = backoff_seconds(attempt, self.backoff_base_s, self.backoff_cap_s, self.rng)
                self.sleep(min(wait, self.max_retry_after_s))
                continue
            if resp.status_code >= 400 and resp.status_code != 404:
                raise HttpError(f"HTTP {resp.status_code} after {attempt + 1} tries: {resp.url}", resp.status_code)
            return resp
        raise AssertionError("unreachable")

    def download(self, url: str, dest: Path, headers: dict | None = None) -> Path:
        """Fetch url to dest once; later calls reuse the file."""
        if dest.exists():
            return dest
        resp = self.get(url, headers=headers)
        if resp.status_code != 200:
            raise HttpError(f"HTTP {resp.status_code}: {url}", resp.status_code)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        tmp.write_bytes(resp.content)
        tmp.replace(dest)
        return dest
