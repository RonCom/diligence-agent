from __future__ import annotations

import email.utils

import httpx
import pytest

from diligence_agent.http import HttpClient, HttpError, backoff_seconds, retry_after_seconds


def client(handler, sleeps: list[float], **kw) -> HttpClient:
    return HttpClient(
        transport=httpx.MockTransport(handler), sleep=sleeps.append, rng=lambda: 1.0, default_rate=float("inf"), **kw
    )


def scripted(*responses):
    calls = iter(responses)

    def handler(request):
        r = next(calls)
        if isinstance(r, Exception):
            raise r
        return r

    return handler


def test_retry_after_seconds_and_date():
    assert retry_after_seconds("7") == 7.0
    assert retry_after_seconds(None) is None
    assert retry_after_seconds("garbage") is None
    when = email.utils.formatdate(1_000_030, usegmt=True)
    assert retry_after_seconds(when, now=1_000_000) == pytest.approx(30.0)


def test_backoff_full_jitter_bounds():
    assert backoff_seconds(0, 1.0, 30.0, rng=lambda: 1.0) == 1.0
    assert backoff_seconds(3, 1.0, 30.0, rng=lambda: 1.0) == 8.0
    assert backoff_seconds(10, 1.0, 30.0, rng=lambda: 1.0) == 30.0
    assert backoff_seconds(3, 1.0, 30.0, rng=lambda: 0.0) == 0.0


def test_honors_retry_after_then_succeeds():
    sleeps: list[float] = []
    c = client(scripted(httpx.Response(429, headers={"Retry-After": "5"}), httpx.Response(200, json={"a": 1})), sleeps)
    assert c.get("https://x.test/").json() == {"a": 1}
    assert sleeps == [5.0]
    assert c.requests == 2


def test_backoff_without_retry_after():
    sleeps: list[float] = []
    c = client(scripted(httpx.Response(503), httpx.Response(502), httpx.Response(200)), sleeps)
    assert c.get("https://x.test/").status_code == 200
    assert sleeps == [1.0, 2.0]


def test_gives_up_after_three_retries():
    sleeps: list[float] = []
    c = client(scripted(*[httpx.Response(503)] * 4), sleeps)
    with pytest.raises(HttpError) as err:
        c.get("https://x.test/")
    assert err.value.status == 503
    assert c.requests == 4 and len(sleeps) == 3


def test_transport_errors_retry_then_raise():
    sleeps: list[float] = []
    c = client(scripted(*[httpx.ConnectTimeout("slow")] * 4), sleeps)
    with pytest.raises(HttpError, match="ConnectTimeout after 4 tries"):
        c.get("https://x.test/")


def test_client_errors_raise_at_once_but_404_returns():
    sleeps: list[float] = []
    c = client(scripted(httpx.Response(403)), sleeps)
    with pytest.raises(HttpError):
        c.get("https://x.test/")
    assert sleeps == []
    c = client(scripted(httpx.Response(404)), sleeps)
    assert c.get("https://x.test/").status_code == 404


def test_rate_limit_spaces_requests_per_host():
    sleeps: list[float] = []
    c = HttpClient(
        transport=httpx.MockTransport(lambda r: httpx.Response(200)),
        sleep=sleeps.append,
        rate_limits={"slow.test": 0.5},
        default_rate=float("inf"),
    )
    c.get("https://slow.test/a")
    c.get("https://fast.test/a")
    c.get("https://slow.test/b")
    assert len(sleeps) == 1 and sleeps[0] == pytest.approx(2.0, abs=0.05)
