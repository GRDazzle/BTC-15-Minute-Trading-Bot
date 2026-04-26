"""Tests for core.tick_window_slicer.

Every backtest path (live, replay, sweep, training) will depend on this
primitive, so exhaustive coverage matters — a regression here silently
corrupts every feature distribution downstream.
"""
from datetime import datetime, timedelta, timezone

import pytest

from core.tick_window_slicer import TickWindowSlicer


UTC = timezone.utc


def mk_tick(ts_iso: str, price: float = 100.0, qty: float = 1.0, is_buyer: bool = True) -> dict:
    return {
        "ts": datetime.fromisoformat(ts_iso),
        "price": price,
        "qty": qty,
        "is_buyer": is_buyer,
    }


# -- append + in-order ---------------------------------------------------------


def test_append_in_order_returns_all_within_window():
    s = TickWindowSlicer()
    anchor = datetime(2026, 4, 24, 12, 0, 0, tzinfo=UTC)
    for i in range(5):
        s.append("coinbase", mk_tick((anchor - timedelta(seconds=i)).isoformat(), price=100 + i))

    win = s.get_merged_window(anchor, lookback_seconds=60, sources=("coinbase",))
    assert len(win) == 5
    assert [t["price"] for t in win] == [104, 103, 102, 101, 100]  # sorted oldest -> newest


def test_append_out_of_order_still_sorted():
    s = TickWindowSlicer()
    anchor = datetime(2026, 4, 24, 12, 0, 0, tzinfo=UTC)
    # Insert intentionally out-of-order
    s.append("coinbase", mk_tick((anchor - timedelta(seconds=10)).isoformat(), price=90))
    s.append("coinbase", mk_tick((anchor - timedelta(seconds=30)).isoformat(), price=70))  # older
    s.append("coinbase", mk_tick((anchor - timedelta(seconds=5)).isoformat(), price=95))

    win = s.get_merged_window(anchor, lookback_seconds=60, sources=("coinbase",))
    assert [t["price"] for t in win] == [70, 90, 95]


def test_append_rejects_naive_ts():
    s = TickWindowSlicer()
    with pytest.raises(ValueError, match="tz-aware"):
        s.append("coinbase", {"ts": datetime(2026, 4, 24, 12, 0, 0), "price": 1.0})


# -- window bounds (inclusive on both ends) ------------------------------------


def test_window_bounds_are_inclusive():
    s = TickWindowSlicer()
    anchor = datetime(2026, 4, 24, 12, 0, 0, tzinfo=UTC)
    # Tick exactly at start boundary
    s.append("coinbase", mk_tick((anchor - timedelta(seconds=60)).isoformat(), price=1))
    # Tick exactly at anchor
    s.append("coinbase", mk_tick(anchor.isoformat(), price=2))
    # Tick just outside (older than start)
    s.append("coinbase", mk_tick((anchor - timedelta(seconds=61)).isoformat(), price=999))

    win = s.get_merged_window(anchor, lookback_seconds=60, sources=("coinbase",))
    assert [t["price"] for t in win] == [1, 2]


def test_no_lookahead():
    s = TickWindowSlicer()
    anchor = datetime(2026, 4, 24, 12, 0, 0, tzinfo=UTC)
    s.append("coinbase", mk_tick((anchor + timedelta(seconds=1)).isoformat(), price=999))
    s.append("coinbase", mk_tick(anchor.isoformat(), price=1))
    win = s.get_merged_window(anchor, lookback_seconds=60, sources=("coinbase",))
    assert [t["price"] for t in win] == [1]


# -- multi-source merge --------------------------------------------------------


def test_multi_source_merge_sorted_by_ts():
    s = TickWindowSlicer()
    anchor = datetime(2026, 4, 24, 12, 0, 0, tzinfo=UTC)
    s.append("coinbase", mk_tick((anchor - timedelta(seconds=30)).isoformat(), price=100))
    s.append("kraken", mk_tick((anchor - timedelta(seconds=25)).isoformat(), price=200))
    s.append("coinbase", mk_tick((anchor - timedelta(seconds=20)).isoformat(), price=101))
    s.append("kraken", mk_tick((anchor - timedelta(seconds=10)).isoformat(), price=201))

    win = s.get_merged_window(anchor, lookback_seconds=60, sources=("coinbase", "kraken"))
    assert [t["price"] for t in win] == [100, 200, 101, 201]


def test_sources_param_filters():
    s = TickWindowSlicer()
    anchor = datetime(2026, 4, 24, 12, 0, 0, tzinfo=UTC)
    s.append("coinbase", mk_tick((anchor - timedelta(seconds=10)).isoformat(), price=100))
    s.append("kraken", mk_tick((anchor - timedelta(seconds=5)).isoformat(), price=200))

    win_cb_only = s.get_merged_window(anchor, lookback_seconds=60, sources=("coinbase",))
    assert [t["price"] for t in win_cb_only] == [100]

    win_kr_only = s.get_merged_window(anchor, lookback_seconds=60, sources=("kraken",))
    assert [t["price"] for t in win_kr_only] == [200]

    win_both = s.get_merged_window(anchor, lookback_seconds=60, sources=("coinbase", "kraken"))
    assert [t["price"] for t in win_both] == [100, 200]


# -- deterministic tie-breaking ------------------------------------------------


def test_ts_tie_same_source_breaks_on_insertion_order():
    s = TickWindowSlicer()
    anchor = datetime(2026, 4, 24, 12, 0, 0, tzinfo=UTC)
    t = anchor - timedelta(seconds=10)
    s.append("coinbase", mk_tick(t.isoformat(), price=1))  # idx=0
    s.append("coinbase", mk_tick(t.isoformat(), price=2))  # idx=1
    s.append("coinbase", mk_tick(t.isoformat(), price=3))  # idx=2

    win = s.get_merged_window(anchor, lookback_seconds=60, sources=("coinbase",))
    assert [t["price"] for t in win] == [1, 2, 3]


def test_ts_tie_cross_source_breaks_on_sources_arg_position():
    s = TickWindowSlicer()
    anchor = datetime(2026, 4, 24, 12, 0, 0, tzinfo=UTC)
    t = anchor - timedelta(seconds=10)
    s.append("kraken", mk_tick(t.isoformat(), price=100))
    s.append("coinbase", mk_tick(t.isoformat(), price=200))

    # coinbase listed first → coinbase tick comes first despite being appended later
    win_cb_first = s.get_merged_window(anchor, lookback_seconds=60, sources=("coinbase", "kraken"))
    assert [t["price"] for t in win_cb_first] == [200, 100]

    # reverse order → kraken comes first
    win_kr_first = s.get_merged_window(anchor, lookback_seconds=60, sources=("kraken", "coinbase"))
    assert [t["price"] for t in win_kr_first] == [100, 200]


# -- determinism (insertion order independence when ts/source same) -----------


def test_repeated_extract_is_idempotent():
    s = TickWindowSlicer()
    anchor = datetime(2026, 4, 24, 12, 0, 0, tzinfo=UTC)
    for i in range(10):
        s.append("coinbase", mk_tick((anchor - timedelta(seconds=i)).isoformat(), price=100 + i))

    win1 = s.get_merged_window(anchor, 60)
    win2 = s.get_merged_window(anchor, 60)
    assert win1 == win2  # list equality, including object identity of the tick dicts


def test_extend_and_append_equivalent():
    s_ext = TickWindowSlicer()
    s_app = TickWindowSlicer()
    anchor = datetime(2026, 4, 24, 12, 0, 0, tzinfo=UTC)
    ticks = [
        mk_tick((anchor - timedelta(seconds=i)).isoformat(), price=100 + i)
        for i in range(10, 0, -1)
    ]
    s_ext.extend("coinbase", ticks)
    for t in ticks:
        s_app.append("coinbase", t)

    w1 = s_ext.get_merged_window(anchor, 60, ("coinbase",))
    w2 = s_app.get_merged_window(anchor, 60, ("coinbase",))
    assert [t["price"] for t in w1] == [t["price"] for t in w2]


# -- prune ---------------------------------------------------------------------


def test_prune_drops_old_ticks():
    s = TickWindowSlicer()
    base = datetime(2026, 4, 24, 12, 0, 0, tzinfo=UTC)
    for i in range(10):
        s.append("coinbase", mk_tick((base - timedelta(seconds=i)).isoformat(), price=100 + i))

    cutoff = base - timedelta(seconds=5)  # keep ts >= base-5s
    s.prune_before(cutoff)

    win = s.get_merged_window(base, lookback_seconds=60, sources=("coinbase",))
    # Should retain ticks at base-5, -4, -3, -2, -1, 0 (ts >= cutoff; prune_before
    # drops strictly-older, keeps equal-to-cutoff)
    assert len(win) == 6
    assert all(t["ts"] >= cutoff for t in win)


def test_prune_is_idempotent():
    s = TickWindowSlicer()
    base = datetime(2026, 4, 24, 12, 0, 0, tzinfo=UTC)
    for i in range(5):
        s.append("coinbase", mk_tick((base - timedelta(seconds=i)).isoformat(), price=100 + i))

    cutoff = base - timedelta(seconds=2)
    s.prune_before(cutoff)
    count1 = s.counts()
    s.prune_before(cutoff)  # second call
    count2 = s.counts()
    assert count1 == count2


def test_prune_rejects_naive_ts():
    s = TickWindowSlicer()
    with pytest.raises(ValueError, match="tz-aware"):
        s.prune_before(datetime(2026, 4, 24, 12, 0, 0))


# -- lookback variants ---------------------------------------------------------


def test_short_lookback_is_suffix_of_long_lookback():
    """Property: get_merged_window(t, 180) is a suffix of get_merged_window(t, 900)."""
    s = TickWindowSlicer()
    anchor = datetime(2026, 4, 24, 12, 0, 0, tzinfo=UTC)
    for i in range(1, 900, 3):  # every 3s
        s.append("coinbase", mk_tick((anchor - timedelta(seconds=i)).isoformat(), price=100.0 + i))

    long_win = s.get_merged_window(anchor, 900, ("coinbase",))
    short_win = s.get_merged_window(anchor, 180, ("coinbase",))

    assert len(short_win) < len(long_win)
    # short_win should equal the suffix of long_win with ts >= anchor-180
    cutoff = anchor - timedelta(seconds=180)
    expected = [t for t in long_win if t["ts"] >= cutoff]
    assert short_win == expected


def test_empty_slicer_returns_empty_window():
    s = TickWindowSlicer()
    anchor = datetime(2026, 4, 24, 12, 0, 0, tzinfo=UTC)
    assert s.get_merged_window(anchor, 60, ("coinbase", "kraken")) == []


def test_unknown_source_is_ignored():
    s = TickWindowSlicer()
    anchor = datetime(2026, 4, 24, 12, 0, 0, tzinfo=UTC)
    s.append("coinbase", mk_tick((anchor - timedelta(seconds=10)).isoformat(), price=100))
    win = s.get_merged_window(anchor, 60, ("coinbase", "bitstamp"))  # bitstamp absent
    assert [t["price"] for t in win] == [100]


# -- get_first_at_or_after -----------------------------------------------------


def test_first_at_or_after_returns_earliest():
    s = TickWindowSlicer()
    base = datetime(2026, 4, 24, 12, 0, 0, tzinfo=UTC)
    s.append("coinbase", mk_tick((base - timedelta(seconds=5)).isoformat(), price=1))
    s.append("coinbase", mk_tick((base + timedelta(seconds=5)).isoformat(), price=2))
    s.append("coinbase", mk_tick((base + timedelta(seconds=10)).isoformat(), price=3))

    t = s.get_first_at_or_after(base, sources=("coinbase",))
    assert t is not None
    assert t["price"] == 2


def test_first_at_or_after_inclusive():
    s = TickWindowSlicer()
    base = datetime(2026, 4, 24, 12, 0, 0, tzinfo=UTC)
    s.append("coinbase", mk_tick(base.isoformat(), price=42))

    t = s.get_first_at_or_after(base, sources=("coinbase",))
    assert t is not None
    assert t["price"] == 42


def test_first_at_or_after_multi_source():
    s = TickWindowSlicer()
    base = datetime(2026, 4, 24, 12, 0, 0, tzinfo=UTC)
    # Coinbase tick 10s after base; Kraken tick 5s after base (earlier)
    s.append("coinbase", mk_tick((base + timedelta(seconds=10)).isoformat(), price=100))
    s.append("kraken", mk_tick((base + timedelta(seconds=5)).isoformat(), price=200))

    t = s.get_first_at_or_after(base, sources=("coinbase", "kraken"))
    # Earliest ts across sources wins
    assert t["price"] == 200


def test_first_at_or_after_tie_resolves_by_source_order():
    s = TickWindowSlicer()
    base = datetime(2026, 4, 24, 12, 0, 0, tzinfo=UTC)
    # Both sources have tick at exactly base
    s.append("coinbase", mk_tick(base.isoformat(), price=100))
    s.append("kraken", mk_tick(base.isoformat(), price=200))

    t = s.get_first_at_or_after(base, sources=("coinbase", "kraken"))
    assert t["price"] == 100  # coinbase listed first → wins

    t = s.get_first_at_or_after(base, sources=("kraken", "coinbase"))
    assert t["price"] == 200  # kraken listed first → wins


def test_first_at_or_after_none_when_no_future_ticks():
    s = TickWindowSlicer()
    base = datetime(2026, 4, 24, 12, 0, 0, tzinfo=UTC)
    s.append("coinbase", mk_tick((base - timedelta(seconds=10)).isoformat(), price=1))
    assert s.get_first_at_or_after(base, sources=("coinbase",)) is None


# -- diagnostics ---------------------------------------------------------------


def test_counts_reflects_current_state():
    s = TickWindowSlicer()
    base = datetime(2026, 4, 24, 12, 0, 0, tzinfo=UTC)
    s.append("coinbase", mk_tick(base.isoformat()))
    s.append("coinbase", mk_tick((base - timedelta(seconds=1)).isoformat()))
    s.append("kraken", mk_tick(base.isoformat()))
    assert s.counts() == {"coinbase": 2, "kraken": 1}

    s.prune_before(base)  # drops the base-1s coinbase tick
    assert s.counts() == {"coinbase": 1, "kraken": 1}
