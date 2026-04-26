"""Multi-source tick window slicer.

Single authoritative primitive for building a merged, time-bounded tick window
across multiple exchanges (Coinbase, Kraken, ...). Used by:

  - Live strategy (`strategies/kalshi_strategy.py`)
  - Replay harness (`scripts/replay_live_inference.py`)
  - Sweep simulator (`backtester/simulator.py`)
  - Training-data generators (`scripts/generate_*.py`)

Identical semantics across all four paths: same ticks in, same merged window
out. This removes the structural drift where each path had its own buffer
sizing / merge / sort code.

Design contract:

  - Time-based, not count-based. `get_merged_window(anchor, lookback_seconds)`
    returns ticks in `[anchor - lookback, anchor]` (inclusive). Caller chooses
    the lookback for its use case (e.g., 180 for LSTM, 900 for velocity
    features). No surprise FIFO truncation at peak tick rates.

  - Deterministic. Same appends in the same order produce the same output.
    Stable tie-break on equal ts via (source_position_in_sources_arg,
    per_source_insertion_idx).

  - No lookahead. Ticks with ts > anchor are never returned.

  - No implicit memory bound. Live calls `prune_before()` periodically; batch
    consumers (backtest / training) just let GC reclaim when the slicer
    falls out of scope.
"""
from __future__ import annotations

import bisect
import heapq
from datetime import datetime, timedelta
from typing import Iterable


class TickWindowSlicer:
    """Multi-source tick store with deterministic time-window extraction.

    Per-source parallel arrays (`_ts`, `_idx`, `_data`) keep ticks sorted by
    ts within each source. `append()` is O(1) amortized for in-order ticks
    and O(log n) bisect-insert for out-of-order. `get_merged_window()` runs
    a two-sided bisect per source then a single k-way merge via `sorted()`
    on `(ts, src_pos, idx)` tuples.
    """

    __slots__ = ("_ts", "_idx", "_data", "_next_idx")

    def __init__(self) -> None:
        self._ts: dict[str, list[datetime]] = {}
        self._idx: dict[str, list[int]] = {}
        self._data: dict[str, list[dict]] = {}
        self._next_idx: dict[str, int] = {}

    # -- insertion -----------------------------------------------------------

    def append(self, source: str, tick: dict) -> None:
        """Append a single tick to the given source.

        Tick must have a tz-aware `ts` field. In-order inserts (the common
        case for live WS streams and pre-sorted bulk loads) are O(1). Out-
        of-order inserts trigger a bisect to preserve sort order.
        """
        ts = tick["ts"]
        if ts.tzinfo is None:
            raise ValueError(f"tick ts must be tz-aware, got naive: {ts}")

        if source not in self._ts:
            self._ts[source] = []
            self._idx[source] = []
            self._data[source] = []
            self._next_idx[source] = 0

        ts_arr = self._ts[source]
        idx_arr = self._idx[source]
        data_arr = self._data[source]

        my_idx = self._next_idx[source]
        self._next_idx[source] = my_idx + 1

        if ts_arr and ts < ts_arr[-1]:
            # Out-of-order arrival; bisect-insert.
            # bisect_right: place after any existing equal-ts entries so
            # insertion-order is preserved on ties (older ts-equal entries
            # keep their lower idx, hence sort first).
            pos = bisect.bisect_right(ts_arr, ts)
            ts_arr.insert(pos, ts)
            idx_arr.insert(pos, my_idx)
            data_arr.insert(pos, tick)
        else:
            ts_arr.append(ts)
            idx_arr.append(my_idx)
            data_arr.append(tick)

    def extend(self, source: str, ticks: Iterable[dict]) -> None:
        """Bulk append. For pre-sorted input this is O(n) amortized."""
        for tick in ticks:
            self.append(source, tick)

    # -- pruning (for long-running live process) -----------------------------

    def prune_before(self, cutoff: datetime) -> None:
        """Drop ticks with ts < cutoff across all sources.

        Backtest / training consumers can ignore this; live should call it
        periodically (e.g. every 60s with cutoff = now - 1800s) to bound
        memory.
        """
        if cutoff.tzinfo is None:
            raise ValueError(f"cutoff must be tz-aware, got naive: {cutoff}")
        for source in self._ts:
            ts_arr = self._ts[source]
            if not ts_arr:
                continue
            drop_to = bisect.bisect_left(ts_arr, cutoff)
            if drop_to > 0:
                del ts_arr[:drop_to]
                del self._idx[source][:drop_to]
                del self._data[source][:drop_to]

    # -- extraction ---------------------------------------------------------

    def get_merged_window(
        self,
        anchor: datetime,
        lookback_seconds: float,
        sources: tuple[str, ...] = ("coinbase", "kraken"),
    ) -> list[dict]:
        """Return ticks from [anchor - lookback_seconds, anchor] across `sources`,
        merged and deterministically sorted.

        Determinism: per-source arrays are already sorted by ts with ties broken
        by insertion order. `heapq.merge` is stable — for equal keys across
        inputs it emits items in argument order — so cross-source ties break
        by position in `sources`. Same input -> same output.

        Implementation: O(M log K) k-way merge where K = len(sources). Fast
        path for single source: return the slice directly (no merge needed,
        no allocations beyond the slice).
        """
        if anchor.tzinfo is None:
            raise ValueError(f"anchor must be tz-aware, got naive: {anchor}")
        start = anchor - timedelta(seconds=lookback_seconds)

        # Collect per-source slices of the raw tick dicts (no intermediate tuples)
        slices: list[list[dict]] = []
        for src in sources:
            ts_arr = self._ts.get(src)
            if not ts_arr:
                continue
            lo = bisect.bisect_left(ts_arr, start)
            hi = bisect.bisect_right(ts_arr, anchor)
            if lo < hi:
                slices.append(self._data[src][lo:hi])

        if not slices:
            return []
        if len(slices) == 1:
            # Single source — slice is already sorted, nothing to merge.
            return slices[0]

        # Multi-source k-way merge. Key is tick["ts"]; heapq.merge is stable
        # so equal ts across sources resolves to sources-arg order.
        return list(heapq.merge(*slices, key=lambda t: t["ts"]))

    def get_first_at_or_after(
        self,
        ts: datetime,
        sources: tuple[str, ...] = ("coinbase", "kraken"),
    ) -> dict | None:
        """Return the earliest tick with `ts >= given ts` across `sources`.

        Ties on ts resolve by source order in `sources`. Returns None if no
        source has a tick at/after the given ts. Used for window-open-price
        lookup in replay (first tick at window_start).
        """
        if ts.tzinfo is None:
            raise ValueError(f"ts must be tz-aware, got naive: {ts}")
        best: tuple[tuple[datetime, int], dict] | None = None
        for src_pos, src in enumerate(sources):
            ts_arr = self._ts.get(src)
            if not ts_arr:
                continue
            lo = bisect.bisect_left(ts_arr, ts)
            if lo >= len(ts_arr):
                continue
            candidate_key = (ts_arr[lo], src_pos)
            if best is None or candidate_key < best[0]:
                best = (candidate_key, self._data[src][lo])
        return best[1] if best else None

    # -- diagnostics --------------------------------------------------------

    def counts(self) -> dict[str, int]:
        """Per-source tick count. For logs / startup diagnostics."""
        return {src: len(ts) for src, ts in self._ts.items()}

    def sources(self) -> list[str]:
        """List of sources with at least one tick."""
        return [src for src, ts in self._ts.items() if ts]
