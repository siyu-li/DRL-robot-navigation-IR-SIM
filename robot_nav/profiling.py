"""
Nesting-aware wall-clock profiler (no dependencies, no source changes).

Built for the CAPSwitcher stack, where the interesting quantities are nested:
one decision contains 22 coarse vets and a precise rollout, and the precise
rollout contains ``selection_interval × n_unreached`` GAT forwards.  Two
numbers are therefore reported per label:

* **total** — inclusive wall time (the whole call, children included);
* **self**  — exclusive wall time (children timed under *other* labels
  subtracted), so the self column sums to the profiled wall time without
  double counting.

Timers are installed by monkey-patching (:meth:`Profiler.patch`), so the hot
path carries no profiling code when you are not profiling.

CUDA note: torch GPU calls are async, but every instrumented backbone call
here ends in a device→host transfer (``.cpu()`` / numpy), which synchronises.
Timings are therefore real; a new call site that returns a GPU tensor without
touching it would need an explicit ``torch.cuda.synchronize()``.

Recursion note: a label that nests inside *itself* would double count its
inclusive total (self time stays correct).  Nothing instrumented here recurses.
"""

from __future__ import annotations

import functools
import time
from contextlib import contextmanager
from dataclasses import dataclass, replace


@dataclass
class Stat:
    """Aggregated timings for one label."""

    calls: int = 0
    total: float = 0.0       # inclusive seconds
    self_time: float = 0.0   # exclusive seconds

    def __sub__(self, other: "Stat") -> "Stat":
        return Stat(
            calls=self.calls - other.calls,
            total=self.total - other.total,
            self_time=self.self_time - other.self_time,
        )

    @property
    def ms_per_call(self) -> float:
        return 1e3 * self.total / self.calls if self.calls else 0.0


class Profiler:
    """Collect nested wall-clock timings under string labels."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self.stats: dict[str, Stat] = {}
        self.counters: dict[str, int] = {}
        self._child_time: list[float] = []      # per active frame
        self._patches: list[tuple[object, str, object]] = []

    # ------------------------------------------------------------------
    # Measurement
    # ------------------------------------------------------------------

    @contextmanager
    def timed(self, name: str):
        """Time the block, attributing nested timers to their own labels."""
        if not self.enabled:
            yield
            return
        self._child_time.append(0.0)
        t0 = time.perf_counter()
        try:
            yield
        finally:
            dt = time.perf_counter() - t0
            child = self._child_time.pop()
            if self._child_time:                # charge parent's child bucket
                self._child_time[-1] += dt
            s = self.stats.get(name)
            if s is None:
                s = self.stats[name] = Stat()
            s.calls += 1
            s.total += dt
            s.self_time += dt - child

    def count(self, name: str, k: int = 1) -> None:
        """Bump a plain counter (things worth counting but not timing)."""
        self.counters[name] = self.counters.get(name, 0) + k

    def wrap(self, fn, name):
        """
        Return ``fn`` wrapped in a timer.  ``name`` is a label, or a callable
        ``(args, kwargs) -> label`` for labels that depend on the call.
        """
        if not self.enabled:
            return fn

        if callable(name):
            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                with self.timed(name(args, kwargs)):
                    return fn(*args, **kwargs)
        else:
            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                with self.timed(name):
                    return fn(*args, **kwargs)
        return wrapper

    def patch(self, target: object, attr: str, name) -> None:
        """Replace ``target.attr`` with a timed wrapper (undo via :meth:`unpatch`)."""
        original = getattr(target, attr)
        self._patches.append((target, attr, original))
        setattr(target, attr, self.wrap(original, name))

    def unpatch(self) -> None:
        """Restore every patched attribute, most recent first."""
        for target, attr, original in reversed(self._patches):
            setattr(target, attr, original)
        self._patches.clear()

    # ------------------------------------------------------------------
    # Snapshots (per-cycle deltas)
    # ------------------------------------------------------------------

    def snapshot(self) -> dict[str, Stat]:
        return {k: replace(v) for k, v in self.stats.items()}

    @staticmethod
    def delta(
        now: dict[str, Stat], before: dict[str, Stat]
    ) -> dict[str, Stat]:
        """Per-label ``now − before`` (labels absent in ``before`` count as 0)."""
        zero = Stat()
        return {k: v - before.get(k, zero) for k, v in now.items()}

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    @staticmethod
    def table(
        stats: dict[str, Stat],
        wall: float | None = None,
        title: str = "",
        order: list[str] | None = None,
        indent: dict[str, int] | None = None,
    ) -> str:
        """
        Render ``stats`` as a table.

        Args:
            wall:   Reference wall time for the ``%wall`` column.
            order:  Label order; unlisted labels follow, sorted by total desc.
            indent: Per-label indent levels (purely visual nesting hints).
        """
        rows = [(k, v) for k, v in stats.items() if v.calls]
        if not rows:
            return "(nothing recorded)"
        rank = {k: i for i, k in enumerate(order or [])}
        rows.sort(key=lambda kv: (rank.get(kv[0], len(rank)), -kv[1].total))
        indent = indent or {}

        w = max(len(k) + 2 * indent.get(k, 0) for k, _ in rows) + 2
        head = (f"{'section':<{w}}{'calls':>10}{'total s':>11}{'%wall':>8}"
                f"{'self s':>11}{'ms/call':>11}")
        out = []
        if title:
            out += ["", "=" * len(head), title, "=" * len(head)]
        out += [head, "-" * len(head)]
        for k, v in rows:
            label = " " * (2 * indent.get(k, 0)) + k
            pct = f"{100.0 * v.total / wall:>7.1f}%" if wall else " " * 8
            out.append(
                f"{label:<{w}}{v.calls:>10d}{v.total:>11.2f}{pct}"
                f"{v.self_time:>11.2f}{v.ms_per_call:>11.3f}"
            )
        out.append("-" * len(head))
        if wall:
            acct = sum(v.self_time for _, v in rows)
            out.append(
                f"{'accounted (Σ self)':<{w}}{'':>10}{acct:>11.2f}"
                f"{100.0 * acct / wall:>7.1f}%"
            )
            out.append(f"{'wall':<{w}}{'':>10}{wall:>11.2f}{100.0:>7.1f}%")
        return "\n".join(out)

    def to_dict(self, wall: float | None = None) -> dict:
        """JSON-serialisable dump of the current stats and counters."""
        return {
            "wall": wall,
            "counters": dict(self.counters),
            "stats": {
                k: {"calls": v.calls, "total_s": v.total, "self_s": v.self_time,
                    "ms_per_call": v.ms_per_call}
                for k, v in self.stats.items()
            },
        }
